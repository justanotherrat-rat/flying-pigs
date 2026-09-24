/*
 * uav_mesh_eval.cc   
 * ============================================================================
 * WHAT THIS PROGRAM DOES
 * ----------------------------------------------------------------------------
 * It replays, inside ns-3, the ship + UAV line-of-sight (LoS) graph that
 * losGeneration.py computed offline. ns-3 then simulates real 802.11 MAC
 * behaviour, real routing (OLSR/AODV), and real UDP traffic on top of that
 * graph, and reports packet delivery ratio, delay, hop count, throughput, 
 * and how the network behaves as the topology changes over time.
 * 
 * For every edge the notebook says is clear, this program computes a
 * maritime PATH LOSS for that link and writes it into the matching band's 
 * ns-3 MatrixPropagationLossModel. Antenna gain, tx power and noise figure 
 * are then applied by the ns-3 PHY as usual, so ns-3 decides whether a 
 * given LoS link actually has enough SNR to be usable. Edges the notebook 
 * says are NOT clear get an arbitrarily huge loss, i.e. they are inaudible.
 *
 * 
 * RADIO MODEL
 * ----------------------------------------------------------------------------
 * Ship <-> ship : L-band  (1.5 GHz default), 30 W tx power, ships-only channel
 * Ship <-> UAV  : L-band  (same channel as above - the UAV is now on it too)
 *                 AND Ku-band (16 GHz default), 50 W tx power, own channel
 *
 * losGeneration.py's LoS graph decides WHICH links can exist at all, per band
 * (it already accounts for radio horizon and Fresnel-zone terrain
 * obstruction, checked separately per band for any pair touching the UAV).
 * 
 * 
 * MARITIME PATH LOSS MODEL
 * ----------------------------------------------------------------------------
 * This file implements the classical two-ray (direct + one specular sea-surface 
 * bounce) propagation model used throughout the naval radar/comms literature (Blake, 
 * "Radar Range-Performance Analysis"; Kerr, "Propagation of Short Radio Waves"):
 * 
 *   1. Geometry: direct ray length r1 and sea-reflected ray length r2 are
 *      computed from the flat-earth image geometry (antenna heights h1, h2,
 *      horizontal separation d).
 *   2. Reflection coefficient: the complex Fresnel reflection coefficient of
 *      seawater is computed from its complex relative permittivity (a
 *      conductivity-dominated dielectric at these frequencies), separately
 *      for the two rays' grazing angle, band, and polarisation.
 *   3. Roughness: Ament's (1953) roughness factor attenuates the coherent 
 *      reflection based on RMS wave height, which is set from a Douglas 
 *      sea-state number.
 *   4. Combine: the direct and (reflection-coefficient and roughness-weighted) 
 *      reflected rays are summed coherently, giving a total path loss 
 *
 * The loss model is still an idealisation: it assumes a flat sea (no evaporation 
 * duct, no atmospheric refraction beyond losGeneration.py's radio-horizon check, 
 * no swell shadowing, single-bounce only). See the CAVEATS block further down for 
 * what it does and doesn't capture.
 *
 * TRAFFIC MODEL
 * ----------------------------------------------------------------------------
 *   Telemetry    : every ship broadcasts a small status packet on L-band
 *                  every --telemetryPeriodS seconds (default 5s), heard by
 *                  every other ship AND the UAV.
 *   Data transfer: every ship sends a burst of --dataBytes worth of packets
 *                  to the UAV every --dataPeriodMin minutes (default 6 min),
 *                  addressed to the UAV's Ku-band interface - though routing
 *                  can divert it over L-band during a Ku outage; see
 *                  PeriodicDataTransfer's call site for the full caveat.
 *
 * USAGE
 * ----------------------------------------------------------------------------
 * Place in ns-3's scratch/ and run from the ns-3 root, e.g.
 *   ./ns3 run "scratch/uav_mesh_eval --dir=ns3_export --useUav=1 --out=with_uav"
 *   ./ns3 run "scratch/uav_mesh_eval --dir=ns3_export --useUav=0 --out=no_uav"
 * Run --PrintHelp for the full list of --flags (radio powers/gains per band
 * and per platform, band frequencies, sea state, polarisation, routing
 * protocol, telemetry/data-transfer periods and sizes, ...).
 */

#include "ns3/aodv-helper.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/propagation-module.h"
#include "ns3/tag.h"
#include "ns3/wifi-module.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UavMeshEval");

// One LoS edge for one step, as read from links.csv. h1/h2 (antenna heights
// above the sea surface, metres) are filled in after nodes.csv is parsed,
// since links.csv only gives node indices, band, and distance, not height.
// isKu comes directly from links.csv's "band" column: a ship-UAV pair can 
// appear as TWO separate Edge entries for the same (i, j) in the same step, 
// one L and one Ku, if both bands are clear simultaneously.
struct Edge
{
    uint32_t i;
    uint32_t j;
    bool isKu;   // true = Ku-band link, false = L-band link
    double dist; // horizontal-equivalent distance between i and j, metres
    double h1;   // antenna height of node i, metres above sea level
    double h2;   // antenna height of node j, metres above sea level
};

// ---------------------------------------------------------------------------
// Global simulation state. These are "global" (rather than passed as
// parameters) only because RefreshLoss() is invoked by the ns-3 scheduler
// as a bare callback and can't carry extra context through Simulator::Schedule
// without more boilerplate; everything here is set once during setup in
// main() and only read afterwards.
// ---------------------------------------------------------------------------
static NodeContainer g_nodes;                    // ships 0..nShips-1, then the UAV last
static Ptr<MatrixPropagationLossModel> g_lossL;  // L-band channel's loss matrix (ships AND UAV)
static Ptr<MatrixPropagationLossModel> g_lossKu; // Ku-band channel's loss matrix (ships AND UAV)
static std::vector<std::vector<Edge>> g_edges;   // g_edges[step] = LoS edges active in that step
static bool g_useUav = true;                     // false => any edge touching the UAV is skipped (baseline run)
static double g_freqLGhz = 1.5;                  // L-band centre frequency
static double g_freqKuGhz = 16.0;                // Ku-band centre frequency
static double g_noLinkDb = 250.0;                // loss used for "no LoS" pairs (effectively infinite)
static double g_stepS = 300.0;                   // seconds between notebook steps (topology snapshots)
static int g_seaState = 3;                       // Douglas sea state 0 (calm) .. 9 (phenomenal), for roughness
static bool g_polV = true;                       // true = vertical polarisation, false = horizontal
static std::vector<uint64_t> g_tx;               // telemetry packets sent, bucketed by step
static std::vector<uint64_t> g_rx;               // telemetry packets received (summed over all receivers), bucketed by step
static std::vector<std::vector<uint64_t>> g_telemetryRx; // receptions by sender and step

// --- continuous loss refresh (see RefreshLoss() below) ---------------------
static double g_refreshS = 5.0; // how often we recompute path loss from live position
static double g_rampS = 20.0;   // how long a link fades in/out when it starts/stops having LoS
static double g_simTimeGlobal = 0.0; // simTime, so RefreshLoss knows when to stop rescheduling itself
// Per (node pair, band) state used to detect on/off transitions and time the fade. 
// Keyed on (i, j, isKu) rather than just (i, j), because a ship-UAV pair's L-band and Ku-band links are independent and can be in different states.
using LinkKey = std::tuple<uint32_t, uint32_t, bool>;
static std::map<LinkKey, bool> g_linkOn;
static std::map<LinkKey, double> g_transitionT;

// Split a CSV line on a delimiter, trimming a trailing '\r' if present (for Windows-style line endings).
static std::vector<std::string>
Split(const std::string& s, char d)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, d))
    {
        if (!tok.empty() && tok.back() == '\r')
        {
            tok.pop_back();
        }
        out.push_back(tok);
    }
    return out;
}

// ===========================================================================
// MARITIME (TWO-RAY SEA-REFLECTION) PATH LOSS MODEL
// ===========================================================================

// Free-space path loss in dB at distance d (metres) and frequency fGhz (GHz).
static double
Fspl(double d, double fGhz)
{
    return 20.0 * std::log10(4.0 * M_PI * d * fGhz * 1e9 / 299792458.0);
}

static double
WattsToDbm(double w)
{
    return 10.0 * std::log10(w * 1000.0);
}

// Complex relative permittivity of seawater, eps_r = eps' - j*eps'', bundled as 
// (eps', sigma) since the imaginary part is frequency-dependent through the 
// conductivity term (see FresnelReflectionCoefficient). Salinity ~35 ppt, ~20 C 
// are representative values distilled from the standard references (Ulaby, Moore & 
// Fung, "Microwave Remote Sensing", vol. 3; Klein & Swift 1977) and are accurate
// to maybe +/-15%
struct SeaDielectric
{
    double epsR;  // real relative permittivity
    double sigma; // conductivity, S/m
};

static SeaDielectric
SeaWaterDielectric(double freqGhz)
{
    if (freqGhz < 4.0)
    {
        return {70.0, 4.3}; // L/S band
    }
    if (freqGhz < 12.0)
    {
        return {55.0, 8.0}; // X band (interpolated; not used by the two bands in this scenario)
    }
    return {35.0, 12.0}; // Ku band
}

// Complex Fresnel reflection coefficient of a flat seawater surface at
// grazing angle grazingRad (radians, measured from the surface), for the 
// given frequency and polarisation. This is the air over seawater Fresnel formula:
//
//   R_h = (sin(psi) - sqrt(eps - cos^2(psi))) / (sin(psi) + sqrt(eps - cos^2(psi)))   [horizontal]
//   R_v = (eps*sin(psi) - sqrt(eps - cos^2(psi))) / (eps*sin(psi) + sqrt(eps - cos^2(psi)))  [vertical]
//
// Most simple ship/UAV whip or blade antennas are vertically polarised, hence g_polV defaults to true.
static std::complex<double>
FresnelReflectionCoefficient(double grazingRad, double freqGhz, bool vertical)
{
    SeaDielectric sw = SeaWaterDielectric(freqGhz);
    const double eps0 = 8.8541878128e-12; // vacuum permittivity, F/m
    const double omega = 2.0 * M_PI * freqGhz * 1e9;
    // eps'' = sigma / (omega * eps0); permittivity has a negative imaginary
    // part under the e^{+jwt} convention used implicitly below.
    std::complex<double> eps(sw.epsR, -sw.sigma / (omega * eps0));

    const double sinPsi = std::sin(grazingRad);
    const double cosPsi = std::cos(grazingRad);
    const std::complex<double> root = std::sqrt(eps - cosPsi * cosPsi);

    const std::complex<double> Rh = (sinPsi - root) / (sinPsi + root);
    const std::complex<double> Rv = (eps * sinPsi - root) / (eps * sinPsi + root);
    return vertical ? Rv : Rh;
}

// Approximate RMS sea-surface wave height (metres) for a given Douglas sea
// state (0 = glassy calm, 9 = phenomenal/hurricane). This is a coarse,
// commonly-cited lookup (wave heights vary a great deal with fetch and wind
// duration for the same "sea state" label); if you have measured/forecast
// significant wave height Hs for your scenario, replace this with
// sigmaH = Hs / 4 (the standard relationship between RMS surface elevation
// and significant wave height for a narrow-band sea).
static double
SeaStateRmsHeight(int seaState)
{
    static const double table[] = {0.0, 0.05, 0.15, 0.4, 1.0, 1.75, 3.0, 5.0, 8.0, 12.0};
    const int s = std::min(std::max(seaState, 0), 9);
    return table[s];
}

// Ament's (1953) specular scattering / roughness factor: attenuates the
// coherent (specular) reflection because a rough sea scatters part of the
// reflected energy away from the specular direction rather than back toward
// the receiver. rho_s = exp(-2*(2*pi*sigmaH*sin(psi)/lambda)^2), which is 1
// for a perfectly smooth surface (sigmaH -> 0) and falls toward 0 as the
// surface gets rough relative to a wavelength - consistent with Ku-band
// (short wavelength) losing its coherent sea bounce in a rougher sea much
// sooner than L-band does.
static double
RoughnessFactor(double grazingRad, double freqGhz, int seaState)
{
    const double lambda = 299792458.0 / (freqGhz * 1e9);
    const double sigmaH = SeaStateRmsHeight(seaState);
    const double g = (2.0 * M_PI * sigmaH * std::sin(grazingRad)) / lambda;
    return std::exp(-2.0 * g * g);
}

// Total two-ray (direct + sea-reflected) path loss in dB.
//   d           horizontal separation, metres
//   h1, h2      antenna heights above the sea surface, metres (clamped to a
//               10 cm floor - the model is undefined for an antenna at the
//               waterline and this scenario never has one that low anyway)
//   freqGhz     link frequency, GHz
//   seaState    Douglas sea state, drives the roughness factor
//   vertical    true = vertical polarisation, false = horizontal
//
// Method (Blake ch. 13 / Kerr ch. 2, the standard maritime two-ray formula):
//   r1 = direct-ray path length, r2 = sea-reflected-ray path length (flat-
//   earth image geometry). The two rays are summed coherently relative to
//   the direct ray's free-space field:
//
//     E_total / E_direct = 1 + rho * exp(-j*dPhi) * (r1/r2)
//
//   where rho = (Fresnel reflection coefficient) * (roughness factor), and
//   dPhi = 2*pi*(r2-r1)/lambda is the two rays' phase difference. Path loss
//   is FSPL(r1) minus 20*log10(|1 + rho*exp(-j dPhi)*(r1/r2)|): this is what
//   produces the classic maritime "lobing" - interference maxima where the
//   two rays add in phase (up to ~6 dB better than FSPL) and interference
//   nulls where they cancel, converging to a smooth ~d^4 flat-earth falloff
//   once you're many wavelengths past the last lobe.
//
// Numerical note: for d >> h1+h2 (essentially always, at these maritime
// ranges), computing dPhi as 2*pi*(r2-r1)/lambda subtracts two nearly-equal
// large numbers and loses precision in double arithmetic. The interference
// term instead uses the algebraically equivalent, numerically stable
// small-angle form dPhi ~= 4*pi*h1*h2/(lambda*d), which is the standard
// asymptotic expression used in exactly this regime.
static double
TwoRaySeaLoss(double d, double h1, double h2, double freqGhz, int seaState, bool vertical)
{
    d = std::max(d, 1.0);
    h1 = std::max(h1, 0.1);
    h2 = std::max(h2, 0.1);

    const double r1 = std::sqrt(d * d + (h1 - h2) * (h1 - h2)); // direct ray
    const double r2 = std::sqrt(d * d + (h1 + h2) * (h1 + h2)); // sea-reflected ray
    const double grazing = std::atan((h1 + h2) / d);            // flat-earth grazing angle
    const double lambda = 299792458.0 / (freqGhz * 1e9);

    double dPhi;
    if (d > 20.0 * (h1 + h2))
    {
        dPhi = (4.0 * M_PI * h1 * h2) / (lambda * d); // stable asymptotic form (see note above)
    }
    else
    {
        dPhi = 2.0 * M_PI * (r2 - r1) / lambda; // fine at short range, where r2-r1 isn't tiny
    }

    const std::complex<double> reflCoef = FresnelReflectionCoefficient(grazing, freqGhz, vertical);
    const double rough = RoughnessFactor(grazing, freqGhz, seaState);
    const std::complex<double> rho = reflCoef * rough;

    const std::complex<double> sumTerm = 1.0 + rho * std::polar(1.0, -dPhi) * (r1 / r2);
    // Guard against the theoretical exact-cancellation null (|sumTerm| -> 0,
    // loss -> infinity): clamp so a single lobe null doesn't turn into a
    // literal "no LoS" edge, since real seas are never perfectly specular.
    const double factor = std::max(std::abs(sumTerm), 1.0e-3);

    return Fspl(r1, freqGhz) - 20.0 * std::log10(factor);
}

// CAVEATS - what the two-ray model above does NOT capture, for anyone citing
// these results:
//   - Single-bounce only. Real multipath can include multiple bounces,
//     diffuse (non-specular) sea clutter, and, closer to shore, land clutter;
//     none of that is modelled.
//   - Flat-earth geometry. Fine at these ranges (a few tens of km) given the
//     notebook's own radio-horizon gate already excludes over-the-horizon
//     links; a curved-earth (4/3-Earth-radius) correction would only matter
//     near the extreme edge of that horizon.
//   - No evaporation duct / anomalous propagation. Marine evaporation ducts
//     can extend VHF/UHF and low-microwave range well beyond geometric LoS,
//     which this model (and the notebook's LoS graph) does not represent.
//   - Sea state is a single global knob (g_seaState / --seaState), not
//     spatially or temporally varying, and the RMS-height table is a coarse
//     approximation - see SeaStateRmsHeight's comment for how to replace it
//     with a measured significant wave height.
//   - Antenna heights are taken directly from the notebook's ENU z-coordinate
//     (altitude above its local sea-level reference), which assumes that
//     reference is close enough to true sea level for the geometry to be
//     valid; ship deck motion (roll/pitch/heave) is not modelled.

// Convert simulation time to a step index, clamped to the last valid bucket
// (guards the tail end of the run, where Simulator::Now() can be a hair past
// the final step's nominal start due to floating-point scheduling).
static uint32_t
Bucket()
{
    const uint32_t k = static_cast<uint32_t>(Simulator::Now().GetSeconds() / g_stepS);
    return std::min<uint32_t>(k, g_tx.size() - 1);
}

class TelemetrySenderTag : public Tag
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::TelemetrySenderTag").SetParent<Tag>().AddConstructor<TelemetrySenderTag>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return sizeof(uint32_t);
    }

    void Serialize(TagBuffer buffer) const override
    {
        buffer.WriteU32(m_sender);
    }

    void Deserialize(TagBuffer buffer) override
    {
        m_sender = buffer.ReadU32();
    }

    void Print(std::ostream& os) const override
    {
        os << "sender=" << m_sender;
    }

    void SetSender(uint32_t sender)
    {
        m_sender = sender;
    }

    uint32_t GetSender() const
    {
        return m_sender;
    }

  private:
    uint32_t m_sender = 0;
};

static void
OnTelemetryRx(Ptr<const Packet> packet, const Address& from)
{
    const uint32_t step = Bucket();
    g_rx[step]++;
    TelemetrySenderTag tag;
    if (packet->PeekPacketTag(tag) && tag.GetSender() < g_telemetryRx.size())
    {
        g_telemetryRx[tag.GetSender()][step]++;
    }
}

// ===========================================================================
// CONTINUOUS LOSS REFRESH (replaces a once-per-step "jump")
// ===========================================================================
// The notebook only re-checks LoS/Fresnel obstruction every g_stepS seconds
// (5 min by default) because that check involves DEM raycasting and is
// expensive. But WHICH pairs have LoS and HOW FAR APART they are are two
// different things: distance changes continuously as ships move (and
// WaypointMobilityModel already interpolates each node's position smoothly
// between the notebook's snapshots, for free), even though the go/no-go LoS
// decision is only re-evaluated once per step.
//
// RefreshLoss() exploits that: instead of freezing each link's loss at the
// value computed from the step's *snapshot* distance (the old, once-per-step
// behaviour), it re-derives loss every g_refreshS seconds from each node's
// *live, interpolated* position - a handful of trig/log ops per active link,
// nowhere near the cost of another DEM pass. On top of that, when a link's
// LoS state flips (appears or disappears at a step boundary), instead of an
// instant jump between "audible" and "inaudible" it fades linearly in dB
// over g_rampS seconds. Two things this buys you:
//   1. Smoother received power - SNR tracks real ship motion within a step
//      instead of stepping in 5-minute jumps.
//   2. A more realistic (and, per the step-by-step debugging done earlier in
//      this scenario, materially less lossy) transition when a link comes up
//      or down: OLSR/AODV get a beat of ramping signal to react to instead of
//      a cliff, similar to how a real link's margin degrades gradually as
//      geometry or obstruction changes, rather than teleporting.
// This is all "free" in the sense that it needs nothing new from the
// notebook: g_edges (the per-step LoS graph) is read exactly as before, at
// the same 5-minute resolution; only what ns-3 does *between* those snapshots
// changes.
static void
RefreshLoss()
{
    const uint32_t n = g_nodes.GetN();
    const uint32_t uav = n - 1;
    const double now = Simulator::Now().GetSeconds();
    const uint32_t step = std::min<uint32_t>(static_cast<uint32_t>(now / g_stepS),
                                             static_cast<uint32_t>(g_edges.size() - 1));

    // "target" = the set of (i, j, band) triples the notebook says should
    // have LoS right now, regardless of how far apart they currently are.
    // The band comes straight from links.csv (Edge::isKu), not from whether
    // the pair touches the UAV, since a ship-UAV pair can be a target on
    // both bands at once (two independent Edge rows for the same i, j).
    std::set<LinkKey> target;
    for (const auto& e : g_edges[step])
    {
        const bool touchesUav = (e.i == uav || e.j == uav);
        if (touchesUav && !g_useUav)
        {
            continue; // ships-only baseline: the UAV is never a valid target on either band
        }
        target.insert({std::min(e.i, e.j), std::max(e.i, e.j), e.isKu});
    }

    // Touch every (pair, band) that should be on now, PLUS every one that
    // was on a moment ago (so a just-removed link gets caught and its
    // fade-out kicked off), PLUS every one still mid-fade per transitionT
    // (so a fade-out keeps getting revisited after g_linkOn has already
    // flipped to false, all the way to its terminal g_noLinkDb value - one
    // extra g_refreshS of grace beyond g_rampS guards against a fade
    // otherwise freezing one tick short of "fully off" on a boundary where
    // tSince lands exactly on g_rampS).
    std::set<LinkKey> touch = target;
    for (const auto& kv : g_linkOn)
    {
        if (kv.second)
        {
            touch.insert(kv.first);
        }
    }
    for (const auto& kv : g_transitionT)
    {
        if (now - kv.second < g_rampS + g_refreshS)
        {
            touch.insert(kv.first);
        }
    }

    uint32_t nOnL = 0;
    uint32_t nOnKu = 0;
    for (const auto& key : touch)
    {
        const uint32_t i = std::get<0>(key);
        const uint32_t j = std::get<1>(key);
        const bool isKu = std::get<2>(key);
        const bool shouldBeOn = target.count(key) > 0;
        const bool wasOn = g_linkOn.count(key) ? g_linkOn[key] : false;

        if (shouldBeOn != wasOn)
        {
            g_transitionT[key] = now; // (re)start the fade from this moment
            g_linkOn[key] = shouldBeOn;
        }

        // Live geometry, not the snapshot distance from links.csv: both
        // endpoints' positions are continuously interpolated by
        // WaypointMobilityModel, so this tracks real motion within a step.
        // Note i/j are the SAME physical nodes regardless of band (e.g. ship
        // 2 and the UAV), so their live positions are identical for the L
        // and Ku evaluations of the same pair - only the frequency differs.
        auto mi = g_nodes.Get(i)->GetObject<MobilityModel>();
        auto mj = g_nodes.Get(j)->GetObject<MobilityModel>();
        const Vector pi = mi->GetPosition();
        const Vector pj = mj->GetPosition();
        const double dx = pi.x - pj.x;
        const double dy = pi.y - pj.y;
        const double dist = std::sqrt(dx * dx + dy * dy);
        const double freq = isKu ? g_freqKuGhz : g_freqLGhz;
        const double physicalLoss = TwoRaySeaLoss(dist, pi.z, pj.z, freq, g_seaState, g_polV);

        const double tSince = now - (g_transitionT.count(key) ? g_transitionT.at(key) : -1e9);
        double loss;
        if (tSince < g_rampS)
        {
            // Mid-fade: blend linearly in dB between "inaudible" and the
            // current physical loss, in whichever direction this link is
            // headed (appearing vs. disappearing).
            const double frac = std::clamp(tSince / g_rampS, 0.0, 1.0);
            const double from = shouldBeOn ? g_noLinkDb : physicalLoss;
            const double to = shouldBeOn ? physicalLoss : g_noLinkDb;
            loss = from + (to - from) * frac;
        }
        else
        {
            loss = shouldBeOn ? physicalLoss : g_noLinkDb;
        }

        // isKu picks which channel's loss matrix this (i, j) pair's link
        // lives on - the L-band and Ku-band matrices are entirely separate
        // MatrixPropagationLossModel objects, so setting one never touches
        // the other, even for the same node pair.
        auto& lossModel = isKu ? g_lossKu : g_lossL;
        lossModel->SetLoss(mi, mj, loss, true);
        if (shouldBeOn)
        {
            (isKu ? nOnKu : nOnL)++;
        }
    }

    NS_LOG_UNCOND("t=" << now << "s  step " << step << ": " << touch.size()
                       << " (pair,band) links tracked, " << nOnL << " L-band + " << nOnKu
                       << " Ku-band currently in the notebook's LoS graph");

    // Keep ticking until just before the run ends; no point scheduling a tick
    // that will never fire before Simulator::Stop().
    if (now + g_refreshS < g_simTimeGlobal)
    {
        Simulator::Schedule(Seconds(g_refreshS), &RefreshLoss);
    }
}

// ===========================================================================
// TRAFFIC MODEL: TELEMETRY (broadcast, L-band) + DATA TRANSFER (unicast, Ku-band)
// ===========================================================================
// Neither of these uses OnOffHelper. Data transfer is a plain unicast UDP
// socket, which OnOffHelper could do too, but bursty periodic sends aren't
// its shape (it models a continuous on/off duty cycle, not "N packets every
// few minutes"). Telemetry is genuine IP broadcast, which OnOffHelper's
// socket cannot do at all: ns-3's UdpSocket defaults to REFUSING sends to a
// broadcast address (Socket::SetAllowBroadcast defaults false) and
// OnOffHelper exposes no attribute to flip that - so real broadcast needs a
// socket we create and configure ourselves, the same pattern ns-3's own
// examples/routing/manet-routing-compare.cc uses for MANET broadcast
// traffic. Both senders below are simple self-rescheduling functions rather
// than full ns3::Application subclasses, since neither needs Application's
// Start/Stop lifecycle beyond what a single initial Simulator::Schedule call
// already gives us.

// Sends one telemetry packet on the given (already broadcast-enabled,
// already Connect()-ed to the L-band broadcast address) socket, then
// reschedules itself. Runs for the lifetime of the simulation once started -
// there's no explicit stop, mirroring the always-on telemetry beacon this is
// meant to represent.
static void
PeriodicTelemetry(Ptr<Socket> socket, uint32_t sender, uint32_t bytes, Time period)
{
    Ptr<Packet> packet = Create<Packet>(bytes);
    TelemetrySenderTag tag;
    tag.SetSender(sender);
    packet->AddPacketTag(tag);
    socket->Send(packet);
    g_tx[Bucket()]++; // no OnOffApplication Tx trace to hook here, so count directly
    Simulator::Schedule(period, &PeriodicTelemetry, socket, sender, bytes, period);
}

static void
SendOnePacket(Ptr<Socket> socket, uint32_t pktSize)
{
    socket->Send(Create<Packet>(pktSize));
}

// Fires off burstPkts back-to-back sends (spaced by gap) on the given
// (already Connect()-ed, unicast) socket - one ship's "data transfer" burst.
static void
StartDataBurst(Ptr<Socket> socket, uint32_t pktSize, uint32_t burstPkts, Time gap)
{
    for (uint32_t k = 0; k < burstPkts; ++k)
    {
        Simulator::Schedule(gap * k, &SendOnePacket, socket, pktSize);
    }
}

// Kicks off one burst immediately, then reschedules itself after `period` -
// i.e. "burstPkts packets, every `period`", forever once started.
static void
PeriodicDataTransfer(Ptr<Socket> socket, uint32_t pktSize, uint32_t burstPkts, Time gap,
                     Time period)
{
    StartDataBurst(socket, pktSize, burstPkts, gap);
    Simulator::Schedule(period, &PeriodicDataTransfer, socket, pktSize, burstPkts, gap, period);
}

int
main(int argc, char* argv[])
{
    // ----------------------------------------------------------- CLI defaults
    std::string dir = "ns3_export";
    std::string out = "result";
    std::string routing = "olsr"; // olsr | aodv
    std::string modeL = "OfdmRate6Mbps";
    std::string modeKu = "OfdmRate6Mbps";
    double warmup = 30.0;
    double simTime = 0.0; // 0 -> derived from the CSV (nSteps * stepS)
    uint32_t run = 1;
    bool useUav = true;

    // ---- fixed traffic model (per the scenario spec) ----------------------
    // Telemetry: every ship broadcasts a small status packet, on L-band, to
    // everyone (all other ships AND the UAV, since the UAV is now on the
    // L-band subnet too), every telemetryPeriodS seconds.
    double telemetryPeriodS = 5.0;
    uint32_t telemetryBytes = 128; // small status message: position/heading/speed/id/time, say

    // Data transfer: every ship sends a burst of "bulk" data to the UAV, on
    // Ku-band, every dataPeriodMin minutes. Sizes/rates are not specified in
    // the brief, so these are placeholders - override via the matching flags.
    double dataPeriodMin = 6.0;
    uint32_t dataBytes = 500000;  // 500 KB per transfer (placeholder bulk size)
    uint32_t dataPktSize = 1024;
    double dataPktGapMs = 5.0; // spacing between packets within one burst
    bool printRoutes = false;
    bool rtscts = false;

    // ---- radio parameters (tx power values given in the scenario brief; the
    // rest - antenna gains, noise figure, and the UAV's L-band tx power
    // (never specified - assumed equal to the ships') - are PLACEHOLDERS. Set
    // them to your actual hardware figures via the matching --flag before
    // trusting results.)
    double txLW = 30.0;       // L-band ship radio power (W)
    double txLUavW = 30.0;    // L-band UAV radio power (W) - ASSUMED equal to ships'; not specified
    double txKuW = 50.0;      // Ku-band radio power, ships and UAV (W)
    double gainL = 3.0;       // L-band ship antenna gain, dBi (assumed)
    double gainLUav = 3.0;    // L-band UAV antenna gain, dBi (assumed equal to a ship's)
    double gainKuShip = 30.0; // Ku ship antenna gain, dBi (assumed)
    double gainKuUav = 25.0;  // Ku UAV antenna gain, dBi (assumed)
    double noiseFigure = 7.0; // receiver noise figure, dB (ns-3 default)

    CommandLine cmd(__FILE__);
    cmd.AddValue("dir", "Folder with nodes.csv and links.csv", dir);
    cmd.AddValue("out", "Prefix for result files", out);
    cmd.AddValue("routing", "olsr | aodv", routing);
    cmd.AddValue("useUav", "1 = keep UAV links (both bands), 0 = ships-only baseline", useUav);
    cmd.AddValue("telemetryPeriodS", "Seconds between each ship's telemetry broadcast",
                telemetryPeriodS);
    cmd.AddValue("telemetryBytes", "Telemetry packet size (bytes)", telemetryBytes);
    cmd.AddValue("dataPeriodMin", "Minutes between each ship's data-transfer burst to the UAV",
                dataPeriodMin);
    cmd.AddValue("dataBytes", "Bytes transferred per ship->UAV burst", dataBytes);
    cmd.AddValue("dataPktSize", "Packet size within a data-transfer burst (bytes)", dataPktSize);
    cmd.AddValue("dataPktGapMs", "Spacing between packets within one data-transfer burst (ms)",
                dataPktGapMs);
    cmd.AddValue("modeL", "Fixed 802.11a data rate on L-band", modeL);
    cmd.AddValue("modeKu", "Fixed 802.11a data rate on Ku-band", modeKu);
    cmd.AddValue("freqL", "L-band centre frequency (GHz)", g_freqLGhz);
    cmd.AddValue("freqKu", "Ku-band centre frequency (GHz)", g_freqKuGhz);
    cmd.AddValue("txL", "L-band ship transmit power (W)", txLW);
    cmd.AddValue("txLUav", "L-band UAV transmit power (W)", txLUavW);
    cmd.AddValue("txKu", "Ku-band transmit power (W)", txKuW);
    cmd.AddValue("gainL", "L-band ship antenna gain (dBi)", gainL);
    cmd.AddValue("gainLUav", "L-band UAV antenna gain (dBi)", gainLUav);
    cmd.AddValue("gainKuShip", "Ku-band ship antenna gain (dBi)", gainKuShip);
    cmd.AddValue("gainKuUav", "Ku-band UAV antenna gain (dBi)", gainKuUav);
    cmd.AddValue("nf", "Receiver noise figure (dB)", noiseFigure);
    cmd.AddValue("seaState",
                "Douglas sea state 0 (calm) .. 9 (phenomenal); drives sea-surface roughness in "
                "the two-ray maritime path loss model",
                g_seaState);
    cmd.AddValue("polV",
                "1 = vertical polarisation (default; typical for whip/blade antennas), "
                "0 = horizontal - changes the sea-reflection coefficient",
                g_polV);
    cmd.AddValue("warmup", "Seconds before traffic starts (routing convergence)", warmup);
    cmd.AddValue("stepS", "Seconds between notebook steps", g_stepS);
    cmd.AddValue("simTime", "Total sim seconds (0 = steps * stepS)", simTime);
    cmd.AddValue("printRoutes", "Dump routing tables 60 s into every step", printRoutes);
    cmd.AddValue("rtscts", "Enable RTS/CTS on all radios (hidden-terminal test)", rtscts);
    cmd.AddValue("refreshS",
                "Seconds between continuous path-loss recomputes from live position "
                "(smaller = smoother SNR tracking, more scheduler events; the notebook's "
                "5-min LoS graph is unaffected either way)",
                g_refreshS);
    cmd.AddValue("rampS",
                "Seconds over which a link fades in/out (in dB) when it appears/disappears "
                "in the notebook's LoS graph, instead of switching instantly",
                g_rampS);
    cmd.AddValue("run", "RNG run number", run);
    cmd.Parse(argc, argv);
    g_useUav = useUav;
    RngSeedManager::SetRun(run);
    if (rtscts)
    {
        // Attribute name is version-dependent; if this aborts on your ns-3, tell me the message.
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", UintegerValue(0));
    }

    const double txLDbm = WattsToDbm(txLW);
    const double txLUavDbm = WattsToDbm(txLUavW);
    const double txKuDbm = WattsToDbm(txKuW);

    // ---------------------------------------------------------------- load CSVs
    // nodes.csv: one row per (step, node) giving its ENU position. We only
    // need x,y (for the horizontal distance already supplied by links.csv -
    // recomputing it isn't necessary) and z (antenna height above the sea
    // surface, needed by the two-ray model below). Positions also drive each
    // node's WaypointMobilityModel purely for visualisation (NetAnim); actual
    // connectivity comes entirely from the loss matrices.
    std::vector<std::vector<Vector>> pos; // pos[node][step]
    std::vector<double> stepTime;         // stepTime[step], seconds
    uint32_t nNodes = 0;
    uint32_t nSteps = 0;
    {
        std::ifstream f(dir + "/nodes.csv");
        NS_ABORT_MSG_IF(!f.is_open(), "cannot open " << dir << "/nodes.csv");
        std::string line;
        std::getline(f, line); // header
        struct Row
        {
            uint32_t step, node;
            double t, x, y, z;
        };
        std::vector<Row> rows;
        while (std::getline(f, line))
        {
            auto c = Split(line, ',');
            if (c.size() < 6)
            {
                continue;
            }
            Row r{static_cast<uint32_t>(std::stoul(c[0])),
                  static_cast<uint32_t>(std::stoul(c[2])),
                  std::stod(c[1]),
                  std::stod(c[3]),
                  std::stod(c[4]),
                  std::stod(c[5])};
            nNodes = std::max(nNodes, r.node + 1);
            nSteps = std::max(nSteps, r.step + 1);
            rows.push_back(r);
        }
        pos.assign(nNodes, std::vector<Vector>(nSteps));
        stepTime.assign(nSteps, 0.0);
        for (const auto& r : rows)
        {
            pos[r.node][r.step] = Vector(r.x, r.y, r.z);
            stepTime[r.step] = r.t;
        }
    }

    // links.csv: one row per (step, i, j, band, distance) for each LoS edge
    // the notebook found clear at that band. Antenna heights (h1, h2) aren't
    // in this file, so they're filled in here from the nodes.csv positions
    // just loaded, using each endpoint's z-coordinate at the same step. A
    // ship-UAV pair can appear as two separate rows in the same step (one L,
    // one Ku) if both bands are clear simultaneously.
    g_edges.assign(nSteps, {});
    {
        std::ifstream f(dir + "/links.csv");
        NS_ABORT_MSG_IF(!f.is_open(), "cannot open " << dir << "/links.csv");
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line))
        {
            auto c = Split(line, ',');
            if (c.size() < 6)
            {
                continue;
            }
            const uint32_t s = std::stoul(c[0]);
            const uint32_t ni = std::stoul(c[2]);
            const uint32_t nj = std::stoul(c[3]);
            const bool isKu = (c[4] == "Ku" || c[4] == "ku" || c[4] == "KU");
            const double dist = std::stod(c[5]);
            const double h1 = (ni < nNodes && s < nSteps) ? pos[ni][s].z : 0.0;
            const double h2 = (nj < nNodes && s < nSteps) ? pos[nj][s].z : 0.0;
            g_edges[s].push_back({ni, nj, isKu, dist, h1, h2});
        }
    }
    const uint32_t nShips = nNodes - 1; // UAV is the last node
    if (simTime <= 0.0)
    {
        simTime = nSteps * g_stepS;
    }
    g_tx.assign(nSteps, 0);
    g_rx.assign(nSteps, 0);
    g_telemetryRx.assign(nShips, std::vector<uint64_t>(nSteps, 0));
    NS_LOG_UNCOND("Loaded " << nShips << " ships + 1 UAV, " << nSteps << " steps, simTime="
                            << simTime << "s, UAV links " << (g_useUav ? "ON" : "OFF")
                            << ", sea state " << g_seaState << ", "
                            << (g_polV ? "vertical" : "horizontal") << " polarisation");

    // ----------------------------------------------- link-budget report (offline)
    // Precomputes, for every LoS edge at every step, both the plain
    // free-space loss and the maritime two-ray loss actually used in the
    // simulation, plus the resulting received power and SNR. This runs before
    // any ns-3 objects are created, so it's a good sanity check to read
    // *before* trusting the simulated PDR/delay numbers: if the worst-case
    // SNR here is already negative, no amount of MAC/routing tuning will fix
    // that link, and if two_ray_db and fspl_db differ wildly for a link, that
    // link is deep in an interference lobe (see TwoRaySeaLoss's header comment).
    // 20 MHz noise bandwidth (802.11a) -> thermal noise floor + noise figure.
    const double noiseDbm = -174.0 + 10.0 * std::log10(20e6) + noiseFigure;
    {
        std::ofstream lb(out + "_linkbudget.csv");
        lb << "step,i,j,band,dist_km,h1_m,h2_m,fspl_db,two_ray_db,rx_dbm,snr_db\n";
        double minSnrL = 1e9;
        double minSnrKu = 1e9;
        for (uint32_t k = 0; k < nSteps; ++k)
        {
            for (const auto& e : g_edges[k])
            {
                const bool touchesUav = (e.i == nNodes - 1 || e.j == nNodes - 1);
                if (touchesUav && !g_useUav)
                {
                    continue;
                }
                const double f = e.isKu ? g_freqKuGhz : g_freqLGhz;
                const double fsplDb = Fspl(e.dist, f);
                const double twoRayDb = TwoRaySeaLoss(e.dist, e.h1, e.h2, f, g_seaState, g_polV);
                // Per-endpoint antenna gain: the UAV has its own gain figure
                // on each band (may differ from a ship's), so look each side
                // up individually rather than assuming "one end is always a
                // ship" - which no longer holds for L-band ship-UAV links.
                auto gainOf = [&](uint32_t node) {
                    const bool isUav = (node == nNodes - 1);
                    if (e.isKu)
                    {
                        return isUav ? gainKuUav : gainKuShip;
                    }
                    return isUav ? gainLUav : gainL;
                };
                // Link budget assumes symmetric tx power on a band (fine by
                // default since txLUav == txL); if you set them differently,
                // this reports the ship-side tx power for L-band links.
                const double txDbm = e.isKu ? txKuDbm : txLDbm;
                const double rx = txDbm + gainOf(e.i) + gainOf(e.j) - twoRayDb;
                const double snr = rx - noiseDbm;
                (e.isKu ? minSnrKu : minSnrL) = std::min(e.isKu ? minSnrKu : minSnrL, snr);
                lb << k << "," << e.i << "," << e.j << "," << (e.isKu ? "Ku" : "L") << ","
                   << std::fixed << std::setprecision(2) << e.dist / 1000.0 << "," << e.h1 << ","
                   << e.h2 << "," << fsplDb << "," << twoRayDb << "," << rx << "," << snr << "\n";
            }
        }
        NS_LOG_UNCOND("Noise floor " << noiseDbm << " dBm. Worst-case LoS-link SNR (two-ray): "
                                     << "L-band " << (minSnrL > 1e8 ? NAN : minSnrL)
                                     << " dB, Ku-band " << (minSnrKu > 1e8 ? NAN : minSnrKu)
                                     << " dB  (see " << out << "_linkbudget.csv)");
    }

    // -------------------------------------------------------------------- nodes
    g_nodes.Create(nNodes);
    NodeContainer ships;
    for (uint32_t s = 0; s < nShips; ++s)
    {
        ships.Add(g_nodes.Get(s));
    }
    NodeContainer uavNode(g_nodes.Get(nShips));

    // Every node's position over time is a series of waypoints, one per
    // notebook step. This is only used for visualisation (e.g. NetAnim) -
    // simulated connectivity comes entirely from the loss matrices set in
    // RefreshLoss(), not from ns-3 computing distance/loss from these
    // positions itself.
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::WaypointMobilityModel");
    mob.Install(g_nodes);
    for (uint32_t n = 0; n < nNodes; ++n)
    {
        auto wp = g_nodes.Get(n)->GetObject<WaypointMobilityModel>();
        for (uint32_t s = 0; s < nSteps; ++s)
        {
            wp->AddWaypoint(Waypoint(Seconds(stepTime[s]), pos[n][s]));
        }
    }

    // ------------------------------------------------------ channels + radios
    // Two independent 802.11 (Yans) channels, one per band, each carrying its
    // own MatrixPropagationLossModel (rewritten continuously by RefreshLoss).
    // A shared little helper builds a channel + loss model pair so the two
    // bands are configured identically apart from which loss matrix is theirs.
    auto makeChannel = [](Ptr<MatrixPropagationLossModel>& lossOut) -> Ptr<YansWifiChannel> {
        Ptr<YansWifiChannel> ch = CreateObject<YansWifiChannel>();
        lossOut = CreateObject<MatrixPropagationLossModel>();
        lossOut->SetDefaultLoss(g_noLinkDb); // any pair we never explicitly set is "no LoS"
        ch->SetPropagationLossModel(lossOut);
        // 802.11 MAC timeouts (ACK/CTS windows) assume short range; at the
        // 10s-of-km ranges here the *real* propagation delay (<0.5 ms) would
        // still break those timeouts if used directly, so we use a small
        // fixed 1 us delay instead. This means the simulation does NOT model
        // real maritime propagation delay's contribution to end-to-end
        // latency - only queuing/MAC/routing delay is captured.
        Ptr<PropagationDelayModel> d = CreateObjectWithAttributes<RandomPropagationDelayModel>(
            "Variable",
            StringValue("ns3::ConstantRandomVariable[Constant=0.000001]"));
        ch->SetPropagationDelayModel(d);
        return ch;
    };
    // Builds a WifiHelper fixed to one 802.11a data rate (no rate adaptation),
    // so throughput/PDR differences reflect the loss model and MAC behaviour,
    // not the rate-control algorithm's own decisions.
    auto makeWifi = [](const std::string& mode) -> WifiHelper {
        WifiHelper w;
        w.SetStandard(WIFI_STANDARD_80211a);
        w.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                  "DataMode",
                                  StringValue(mode),
                                  "ControlMode",
                                  StringValue(mode));
        return w;
    };
    // Applies tx power (as dBm, both Start and End so there's no ramping) and
    // antenna gain to a PHY helper. TxGain/RxGain here are lumped antenna
    // gains; the maritime propagation loss already written into the channel's
    // loss matrix is separate and is not affected by these.
    auto setPhy = [&](YansWifiPhyHelper& p, Ptr<YansWifiChannel> ch, double txDbm, double gain) {
        p.SetChannel(ch);
        p.Set("TxPowerStart", DoubleValue(txDbm));
        p.Set("TxPowerEnd", DoubleValue(txDbm));
        p.Set("TxGain", DoubleValue(gain));
        p.Set("RxGain", DoubleValue(gain));
        p.Set("RxNoiseFigure", DoubleValue(noiseFigure));
    };

    Ptr<YansWifiChannel> chL = makeChannel(g_lossL);
    Ptr<YansWifiChannel> chKu = makeChannel(g_lossKu);

    YansWifiPhyHelper phyLShip;
    YansWifiPhyHelper phyLUav;   // UAV's L PHY: its own tx power/gain (gainLUav, txLUavW)
    YansWifiPhyHelper phyKuShip;
    YansWifiPhyHelper phyKuUav;  // UAV's Ku PHY has its own antenna gain (gainKuUav != gainKuShip)
    setPhy(phyLShip, chL, txLDbm, gainL);
    setPhy(phyLUav, chL, txLUavDbm, gainLUav);
    setPhy(phyKuShip, chKu, txKuDbm, gainKuShip);
    setPhy(phyKuUav, chKu, txKuDbm, gainKuUav);

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac"); // no AP - every radio is a peer

    // Both bands are now installed on ships, then the UAV, in that order, so
    // their device/interface indices line up with ships followed by the UAV
    // throughout (index nShips is always the UAV, on either container).
    NetDeviceContainer devL = makeWifi(modeL).Install(phyLShip, mac, ships);
    devL.Add(makeWifi(modeL).Install(phyLUav, mac, uavNode));
    NetDeviceContainer devKu = makeWifi(modeKu).Install(phyKuShip, mac, ships);
    devKu.Add(makeWifi(modeKu).Install(phyKuUav, mac, uavNode));

    // ---------------------------------------------------------------- internet
    InternetStackHelper internet;
    if (routing == "aodv")
    {
        AodvHelper aodv;
        internet.SetRoutingHelper(aodv);
    }
    else
    {
        OlsrHelper olsr;
        internet.SetRoutingHelper(olsr);
    }
    internet.Install(g_nodes);
    Ipv4AddressHelper addr;
    addr.SetBase("10.1.1.0", "255.255.255.0"); // L-band subnet: ships AND the UAV now
    Ipv4InterfaceContainer ifL = addr.Assign(devL);
    addr.SetBase("10.1.2.0", "255.255.255.0"); // Ku-band subnet: ships then UAV
    Ipv4InterfaceContainer ifKu = addr.Assign(devKu);
    // Both subnets are 10.1.x.NODE+1, so (last octet - 1) recovers the node
    // index from either address. On both containers the UAV is index nShips
    // (last), since both were installed ships-then-UAV.

    // ------------------------------------------------------------------ traffic
    const uint16_t telemetryPort = 9001;
    const uint16_t dataPort = 9002;

    // --- Telemetry sinks: every node (ships AND the UAV, now that the UAV is
    // on the L-band subnet) listens for broadcast telemetry. A UDP socket
    // bound to (Any, port) receives both unicast and broadcast packets
    // addressed to that port, so no special receive-side flag is needed -
    // only the SENDER needs SetAllowBroadcast(true) (see below).
    for (uint32_t n = 0; n < nNodes; ++n)
    {
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), telemetryPort));
        auto app = sink.Install(g_nodes.Get(n));
        app.Get(0)->TraceConnectWithoutContext("Rx", MakeCallback(&OnTelemetryRx));
        app.Start(Seconds(0.0));
    }
    // --- Telemetry senders: one raw socket per ship, connected to the
    // limited broadcast address (255.255.255.255) and explicitly bound to
    // the ship's L-band NetDevice. Both of these matter because each ship
    // now has TWO devices (L and Ku): without BindToNetDevice, a broadcast
    // send has no unambiguous egress interface to resolve to and can be
    // silently dropped at the IP layer before it ever reaches the radio -
    // and subnet-directed broadcast (10.1.1.255) isn't guaranteed to be
    // locally delivered the way the limited broadcast address always is.
    // This mirrors the pattern ns-3's own examples/routing/manet-routing-
    // compare.cc uses for MANET broadcast traffic.
    for (uint32_t s = 0; s < nShips; ++s)
    {
        // Spread each ship's first beacon evenly across one telemetry period
        // so all ships do not contend at the same simulation instant.
        const double telemetryPhaseS = telemetryPeriodS * double(s) / nShips;
        Ptr<Socket> sock = Socket::CreateSocket(g_nodes.Get(s), UdpSocketFactory::GetTypeId());
        sock->BindToNetDevice(devL.Get(s));
        sock->SetAllowBroadcast(true);
        sock->Connect(InetSocketAddress(Ipv4Address::GetBroadcast(), telemetryPort));
        Simulator::Schedule(Seconds(warmup + telemetryPhaseS), &PeriodicTelemetry, sock, s,
                    telemetryBytes, Seconds(telemetryPeriodS));
    }

    // --- Data-transfer sink: only the UAV needs to receive these (that's
    // the whole point of the flow), on its Ku-band address. Its stats are
    // read from FlowMonitor below rather than a manual per-step counter,
    // since (unlike broadcast telemetry) it's a clean single-destination
    // unicast flow that FlowMonitor already classifies cleanly per ship.
    {
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), dataPort));
        auto app = sink.Install(uavNode);
        app.Start(Seconds(0.0));
    }
    // --- Data-transfer senders: one unicast socket per ship, connected to
    // the UAV's Ku-band address. A plain UdpSocket needs no special flag for
    // unicast, unlike the broadcast telemetry sockets above.
    //
    // IMPORTANT: this is addressed to the UAV's KU address, but OLSR/AODV
    // will happily route it over L-band instead if the direct Ku link is
    // down and an L-band path (now that the UAV has an L radio too) is up -
    // the routing protocol has no notion of "this traffic is meant to stay
    // on the high-rate band". So a "data transfer" flow can, in periods when
    // Ku is out, end up crossing the lower-capacity L-band channel instead.
    // That's a real, expected consequence of adding UAV L-band connectivity,
    // not a bug - but it means "Ku-band data transfer" numbers can include
    // some packets that didn't actually travel over Ku. *_datatransfer.csv doesn't
    // currently break that out; if it matters for your analysis, cross-check
    // against *_routes.txt (--printRoutes=1) for the periods in question.
    const Ipv4Address uavKuAddr = ifKu.GetAddress(nShips);
    const uint32_t burstPkts = std::max<uint32_t>(1, dataBytes / dataPktSize);
    for (uint32_t s = 0; s < nShips; ++s)
    {
        // Use a separate deterministic phase for each ship's bulk-transfer
        // burst, avoiding synchronized Ku-band bursts as well.
        const double dataPhaseMin = dataPeriodMin * double(s) / nShips;
        Ptr<Socket> sock = Socket::CreateSocket(g_nodes.Get(s), UdpSocketFactory::GetTypeId());
        sock->Connect(InetSocketAddress(uavKuAddr, dataPort));
        Simulator::Schedule(Minutes(warmup / 60.0 + dataPhaseMin), &PeriodicDataTransfer,
                            sock, dataPktSize, burstPkts,
                            MilliSeconds(static_cast<int64_t>(dataPktGapMs)),
                            Minutes(dataPeriodMin));
    }

    // --------------------------------------------------- topology schedule
    // Kick off the continuous loss refresh (see RefreshLoss()'s header
    // comment): it reads g_edges/g_stepS to know which pairs the notebook
    // currently says have LoS, recomputes their loss from live position every
    // g_refreshS seconds, fades transitions over g_rampS seconds, and
    // reschedules itself until just before simTime.
    g_simTimeGlobal = simTime;
    Simulator::Schedule(Seconds(0.0), &RefreshLoss);
    // Optional: dump every node's routing table 60 s into each step (i.e.
    // well after that step's transitions and any resulting reconvergence),
    // which is useful for diagnosing whether traffic is taking the route you
    // expect (e.g. whether it's using a direct L-band link or detouring
    // through the UAV even when a direct link exists).
    if (printRoutes)
    {
        Ptr<OutputStreamWrapper> rs = Create<OutputStreamWrapper>(out + "_routes.txt", std::ios::out);
        for (uint32_t k = 0; k < nSteps; ++k)
        {
            if (stepTime[k] + 60.0 < simTime)
            {
                Ipv4RoutingHelper::PrintRoutingTableAllAt(Seconds(stepTime[k] + 60.0), rs);
            }
        }
    }

    FlowMonitorHelper fmh;
    Ptr<FlowMonitor> mon = fmh.InstallAll();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // ==========================================================================
    // RESULTS
    // ==========================================================================
    mon->CheckForLostPackets();
    auto cls = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());

    // Expected ("offered") counts, computed from the traffic *settings*
    // rather than from what actually got sent - so a packet that never even
    // made it onto the wire (e.g. socket-level failure) still counts as
    // offered-but-lost rather than silently vanishing from PDR.
    const double avail = std::max(0.0, simTime - warmup);
    const auto telemetryOfferedFor = [&](uint32_t ship) {
        const double phase = telemetryPeriodS * double(ship) / nShips;
        return phase < avail
                   ? static_cast<uint64_t>(std::floor((avail - phase) / telemetryPeriodS)) + 1
                   : uint64_t{0};
    };
    const uint64_t dataBurstCount =
        static_cast<uint64_t>(std::floor(avail / (dataPeriodMin * 60.0))) + 1;
    const uint64_t dataOfferedPkts = dataBurstCount * burstPkts;

    // --- TELEMETRY (broadcast, L-band) --------------------------------------
    // Do not use FlowMonitor for broadcast reception totals: its broadcast
    // flow accounting is not a reliable "one receive per listening node"
    // counter. The sink Rx trace counts each actual reception, and the packet
    // tag lets us retain the originating ship for the per-ship report.
    const uint32_t possibleReceivers = nNodes - 1; // everyone except the sender itself
    std::ofstream tf(out + "_telemetry.csv");
    tf << "ship,offered_broadcasts,sent_broadcasts,rx_receptions,possible_receptions,pdr,"
          "mean_delay_ms\n";
    uint64_t telTotRx = 0;
    uint64_t telTotPossible = 0;
    for (uint32_t s = 0; s < nShips; ++s)
    {
        const uint64_t telemetryCount = telemetryOfferedFor(s);
        uint64_t receptions = 0;
        for (uint32_t k = 0; k < nSteps; ++k)
        {
            receptions += g_telemetryRx[s][k];
        }
        const uint64_t possible = telemetryCount * possibleReceivers;
        const double pdr = possible ? double(receptions) / possible : 0.0;
        tf << s << "," << telemetryCount << "," << telemetryCount << "," << receptions << ","
           << possible << "," << std::fixed << std::setprecision(4) << pdr << ",0\n";
        telTotRx += receptions;
        telTotPossible += possible;
    }
    tf.close();

    // --- DATA TRANSFER (unicast, ship -> UAV, nominally Ku-band) -----------
    // A clean single-destination unicast flow per ship (destination is
    // always the UAV), so unlike telemetry this can be aggregated the
    // straightforward way: one flow per ship, no broadcast fan-out.
    // g_edges/g_useUav still gate whether a Ku (or, per the caveat above the
    // sender loop, sometimes L) path to the UAV exists at all.
    struct DataAgg
    {
        uint64_t tx = 0;
        uint64_t rx = 0;
        uint64_t rxBytes = 0;
        uint64_t fwd = 0;
        double delay = 0.0;
    };
    std::map<int, DataAgg> dataAgg;
    for (const auto& kv : mon->GetFlowStats())
    {
        auto t = cls->FindFlow(kv.first);
        if (t.destinationPort != dataPort)
        {
            continue;
        }
        const int src = (t.sourceAddress.Get() & 0xFF) - 1;
        DataAgg& a = dataAgg[src];
        a.tx += kv.second.txPackets;
        a.rx += kv.second.rxPackets;
        a.rxBytes += kv.second.rxBytes;
        a.fwd += kv.second.timesForwarded;
        a.delay += kv.second.delaySum.GetSeconds();
    }
    std::ofstream df(out + "_datatransfer.csv");
    df << "ship,offered_pkts,sent_pkts,noroute_pkts,rx_pkts,pdr,mean_delay_ms,mean_hops,"
          "throughput_kbps\n";
    uint64_t dataTotOff = 0;
    uint64_t dataTotRx = 0;
    double dataTotDelay = 0.0;
    double dataTotFwd = 0.0;
    for (uint32_t s = 0; s < nShips; ++s)
    {
        const DataAgg& a =
            dataAgg.count(static_cast<int>(s)) ? dataAgg[static_cast<int>(s)] : DataAgg{};
        const uint64_t noRoute = dataOfferedPkts > a.tx ? dataOfferedPkts - a.tx : 0;
        const double pdr = dataOfferedPkts ? double(a.rx) / dataOfferedPkts : 0.0;
        const double dms = a.rx ? a.delay * 1000.0 / a.rx : 0.0;
        const double hops = a.rx ? 1.0 + double(a.fwd) / a.rx : 0.0;
        const double kbps = a.rxBytes * 8.0 / avail / 1000.0;
        df << s << "," << dataOfferedPkts << "," << a.tx << "," << noRoute << "," << a.rx << ","
           << std::fixed << std::setprecision(4) << pdr << "," << dms << "," << hops << ","
           << kbps << "\n";
        dataTotOff += dataOfferedPkts;
        dataTotRx += a.rx;
        dataTotDelay += a.delay;
        dataTotFwd += a.fwd;
    }
    df.close();

    // --- Per-step breakdown (telemetry only - see TelAgg's comment above for
    // why data-transfer doesn't get the same manual-bucket treatment: its
    // FlowMonitor aggregation above is already clean per-ship). Link counts
    // now come from Edge::isKu directly instead of "touches the UAV".
    std::ofstream sf(out + "_steps.csv");
    sf << "step,t_start_s,links_L,links_Ku,telemetry_tx_pkts,telemetry_rx_receptions,"
          "telemetry_pdr\n";
    for (uint32_t k = 0; k < nSteps; ++k)
    {
        uint32_t nL = 0;
        uint32_t nKu = 0;
        for (const auto& e : g_edges[k])
        {
            const bool touchesUav = (e.i == nNodes - 1 || e.j == nNodes - 1);
            if (touchesUav && !g_useUav)
            {
                continue;
            }
            (e.isKu ? nKu : nL)++;
        }
        sf << k << "," << stepTime[k] << "," << nL << "," << nKu << "," << g_tx[k] << ","
           << g_rx[k] << "," << std::fixed << std::setprecision(4)
           << ((g_tx[k] != 0 && possibleReceivers != 0)
               ? double(g_rx[k]) / (g_tx[k] * possibleReceivers)
               : 0.0)
           << "\n";
    }
    sf.close();

    std::cout << "\n===== SUMMARY (" << (g_useUav ? "with UAV" : "ships only") << ", " << routing
              << ", sea state " << g_seaState << ") =====\n"
              << "Telemetry PDR (broadcast, L-band)   : "
              << (telTotPossible ? 100.0 * telTotRx / telTotPossible : 0.0) << " %\n"
              << "Data-transfer PDR (ship->UAV)       : "
              << (dataTotOff ? 100.0 * dataTotRx / dataTotOff : 0.0) << " %\n"
              << "Data-transfer mean e2e delay        : "
              << (dataTotRx ? 1000.0 * dataTotDelay / dataTotRx : 0.0) << " ms\n"
              << "Data-transfer mean hop count        : "
              << (dataTotRx ? 1.0 + dataTotFwd / dataTotRx : 0.0) << "\n"
              << "Files  : " << out << "_telemetry.csv, " << out << "_datatransfer.csv, " << out
              << "_steps.csv, " << out << "_linkbudget.csv\n";

    Simulator::Destroy();
    return 0;
}