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
 * RADIO MODEL
 * ----------------------------------------------------------------------------
 *   Ship <-> ship : L-band  (1.5 GHz default),  30 W tx power, own 802.11 channel
 *   Ship <-> UAV  : Ku-band (16  GHz default),  50 W tx power, own 802.11 channel
 * Every ship therefore carries two radios; the UAV carries one (Ku only).
 *
 * The losGeneration.py's LoS graph decides WHICH links can exist at all (it already
 * accounts for radio horizon and Fresnel-zone terrain obstruction). For every
 * edge the losGeneration.py says is clear, this program computes a maritime PATH LOSS
 * for that link (see "MARITIME PATH LOSS MODEL" below) and writes it into an
 * ns-3 MatrixPropagationLossModel. Antenna gain, tx power and noise figure are
 * then applied by the ns-3 PHY as usual, so ns-3 - not us - decides whether a
 * given LoS link actually has enough SNR to be usable. Edges the losGeneration.py
 * says are NOT clear get an arbitrarily huge loss, i.e. they are inaudible.
 *
 * MARITIME PATH LOSS MODEL
 * ----------------------------------------------------------------------------
 * Over water, free-space path loss (FSPL) alone is a poor model: a large
 * fraction of the energy also reaches the receiver via a specular reflection
 * off the sea surface, and that reflected ray interferes with the direct ray.
 * This produces the well-known maritime "lobing" pattern (constructive and
 * destructive interference rings as range or height changes), which is very
 * different from FSPL's smooth 1/d^2 falloff. This file implements the
 * classical two-ray (direct + one specular sea-surface bounce) propagation
 * model used throughout the naval radar/comms literature (Blake, "Radar
 * Range-Performance Analysis"; Kerr, "Propagation of Short Radio Waves"):
 *
 *   1. Geometry: direct ray length r1 and sea-reflected ray length r2 are
 *      computed from the flat-earth image geometry (antenna heights h1, h2,
 *      horizontal separation d).
 *   2. Reflection coefficient: the complex Fresnel reflection coefficient of
 *      seawater is computed from its complex relative permittivity (a
 *      conductivity-dominated dielectric at these frequencies), separately
 *      for the two rays' grazing angle, band, and polarisation.
 *   3. Roughness: the sea is not a perfect mirror. Ament's (1953) roughness
 *      factor attenuates the coherent reflection based on RMS wave height,
 *      which is set from a Douglas sea-state number.
 *   4. Combine: the direct and (reflection-coefficient- and roughness-
 *      weighted) reflected rays are summed coherently (with their relative
 *      phase), giving a total path loss that oscillates around FSPL at short
 *      range and asymptotes to the classic ~d^4 "flat-earth" law at long
 *      range, exactly as the physics literature describes.
 *
 * This is the most physically faithful loss model reasonably implementable
 * without a full ray-tracer or an empirical clutter database. It is still an
 * idealisation: it assumes a flat sea (no evaporation duct, no atmospheric
 * refraction beyond the losGeneration.py's radio-horizon check, no swell shadowing,
 * single-bounce only). See the CAVEATS block further down for what it does
 * and doesn't capture.
 *
 * USAGE
 * ----------------------------------------------------------------------------
 * Place in ns-3's scratch/ and run from the ns-3 root, e.g.
 *   ./ns3 run "scratch/uav_mesh_eval --dir=ns3_export --useUav=1 --out=with_uav"
 *   ./ns3 run "scratch/uav_mesh_eval --dir=ns3_export --useUav=0 --out=no_uav"
 * Run --PrintHelp for the full list of --flags (radio powers/gains, band
 * frequencies, sea state, polarisation, routing protocol, traffic rate, ...).
 *
 * Not compiled by the author against your ns-3 checkout - the maritime-loss
 * math (TwoRaySeaLoss and its helpers) was unit-tested standalone with g++,
 * but the ns-3 API calls around it were not; expect minor version-specific
 * fixes, as with earlier revisions of this file.
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
#include <utility>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UavMeshEval");

// One LoS edge for one step, as read from links.csv. h1/h2 (antenna heights
// above the sea surface, metres) are filled in after nodes.csv is parsed,
// since links.csv only gives node indices and distance, not height.
struct Edge
{
    uint32_t i;
    uint32_t j;
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
static Ptr<MatrixPropagationLossModel> g_lossL;  // ship<->ship (L-band) channel's loss matrix
static Ptr<MatrixPropagationLossModel> g_lossKu; // ship<->UAV (Ku-band) channel's loss matrix
static std::vector<std::vector<Edge>> g_edges;   // g_edges[step] = LoS edges active in that step
static bool g_useUav = true;                     // false => Ku links are never turned on (baseline run)
static double g_freqLGhz = 1.5;                  // L-band centre frequency
static double g_freqKuGhz = 16.0;                // Ku-band centre frequency
static double g_noLinkDb = 250.0;                // loss used for "no LoS" pairs (effectively infinite)
static double g_stepS = 300.0;                   // seconds between notebook steps (topology snapshots)
static int g_seaState = 3;                       // Douglas sea state 0 (calm) .. 9 (phenomenal), for roughness
static bool g_polV = true;                       // true = vertical polarisation, false = horizontal
static std::vector<uint64_t> g_tx;               // app-layer packets sent, bucketed by step
static std::vector<uint64_t> g_rx;               // app-layer packets received, bucketed by step

// --- continuous loss refresh (see RefreshLoss() below) ---------------------
static double g_refreshS = 5.0; // how often we recompute path loss from live position
static double g_rampS = 20.0;   // how long a link fades in/out when it starts/stops having LoS
static double g_simTimeGlobal = 0.0; // simTime, so RefreshLoss knows when to stop rescheduling itself
// Per node-pair (i<j) state used to detect on/off transitions and time the fade.
static std::map<std::pair<uint32_t, uint32_t>, bool> g_linkOn;
static std::map<std::pair<uint32_t, uint32_t>, double> g_transitionT;

// Split a CSV line on a delimiter, trimming a trailing '\r' (files written on
// Windows, or by Python's csv module without an explicit line terminator,
// often carry one; without this the last field on each line fails to parse).
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
// Used both directly (Fspl(r1,...) is the "carrier" term the two-ray model
// modulates) and as a quick reference figure in the link-budget report.
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

// Complex relative permittivity of seawater, eps_r = eps' - j*eps'', bundled
// as (eps', sigma) since the imaginary part is frequency-dependent through
// the conductivity term (see FresnelReflectionCoefficient). Seawater at
// microwave frequencies is dominated by its high conductivity, so its
// effective loss tangent is large and roughly frequency-independent over an
// octave or so; these figures (salinity ~35 ppt, ~20 C) are representative
// values distilled from the standard references (Ulaby, Moore & Fung,
// "Microwave Remote Sensing", vol. 3; Klein & Swift 1977) and are accurate
// to maybe +/-15%, not laboratory-grade. Replace with a full Klein-Swift (or
// Debye double-pole) fit if you need better than that.
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
// grazing angle grazingRad (radians, measured from the surface - NOT from
// the normal), for the given frequency and polarisation. This is the
// standard two-medium (air over seawater) Fresnel formula:
//
//   R_h = (sin(psi) - sqrt(eps - cos^2(psi))) / (sin(psi) + sqrt(eps - cos^2(psi)))   [horizontal]
//   R_v = (eps*sin(psi) - sqrt(eps - cos^2(psi))) / (eps*sin(psi) + sqrt(eps - cos^2(psi)))  [vertical]
//
// Horizontal polarisation reflects strongly (|R| approx 1) at all practical
// maritime grazing angles. Vertical polarisation has a Brewster-angle dip:
// its reflectivity falls sharply once the grazing angle exceeds roughly
// sqrt(1/eps_r) radians (a few degrees for seawater), which is why elevated
// links (e.g. to a UAV) see much less multipath than low grazing ship-ship
// links once they're not near-grazing. Most simple ship/UAV whip or blade
// antennas are vertically polarised, hence g_polV defaults to true.
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
//     losGeneration.py's own radio-horizon gate already excludes over-the-horizon
//     links; a curved-earth (4/3-Earth-radius) correction would only matter
//     near the extreme edge of that horizon.
//   - No evaporation duct / anomalous propagation. Marine evaporation ducts
//     can extend VHF/UHF and low-microwave range well beyond geometric LoS,
//     which this model (and the losGeneration.py's LoS graph) does not represent.
//   - Sea state is a single global knob (g_seaState / --seaState), not
//     spatially or temporally varying, and the RMS-height table is a coarse
//     approximation - see SeaStateRmsHeight's comment for how to replace it
//     with a measured significant wave height.
//   - Antenna heights are taken directly from the losGeneration.py's ENU z-coordinate
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

// Trace callbacks: count application-layer packets sent/received, bucketed by
// which notebook step was active when they occurred. Used for the per-step
// PDR breakdown in *_steps.csv.
static void
OnTx(Ptr<const Packet> p)
{
    g_tx[Bucket()]++;
}

static void
OnRx(Ptr<const Packet> p, const Address& from)
{
    g_rx[Bucket()]++;
}

// ===========================================================================
// CONTINUOUS LOSS REFRESH (replaces a once-per-step "jump")
// ===========================================================================
// The notebook only re-checks LoS/Fresnel obstruction every g_stepS seconds
// (5 min by default) because that check involves DEM raycasting and is
// expensive. But WHICH pairs have LoS and HOW FAR APART they are are two
// different things: distance changes continuously as ships move (and
// WaypointMobilityModel already interpolates each node's position smoothly
// between the losGeneration.py's snapshots, for free), even though the go/no-go LoS
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

    // "target" = the set of pairs the notebook says should have LoS right now
    // (i.e. during this step), regardless of how far apart they currently are.
    std::set<std::pair<uint32_t, uint32_t>> target;
    for (const auto& e : g_edges[step])
    {
        const bool isKu = (e.i == uav || e.j == uav);
        if (isKu && !g_useUav)
        {
            continue; // ships-only baseline: the UAV is never a valid target
        }
        target.insert({std::min(e.i, e.j), std::max(e.i, e.j)});
    }

    // We need to touch every pair that should be on now, AND every pair that
    // was on a moment ago (so a just-removed link gets its fade-out tick
    // instead of vanishing silently).
    std::set<std::pair<uint32_t, uint32_t>> touch = target;
    for (const auto& kv : g_linkOn)
    {
        if (kv.second)
        {
            touch.insert(kv.first);
        }
    }

    uint32_t nOn = 0;
    for (const auto& pr : touch)
    {
        const uint32_t i = pr.first;
        const uint32_t j = pr.second;
        const bool isKu = (i == uav || j == uav);
        const bool shouldBeOn = target.count(pr) > 0;
        const bool wasOn = g_linkOn.count(pr) ? g_linkOn[pr] : false;

        if (shouldBeOn != wasOn)
        {
            g_transitionT[pr] = now; // (re)start the fade from this moment
            g_linkOn[pr] = shouldBeOn;
        }

        // Live geometry, not the snapshot distance from links.csv: both
        // endpoints' positions are continuously interpolated by
        // WaypointMobilityModel, so this tracks real motion within a step.
        auto mi = g_nodes.Get(i)->GetObject<MobilityModel>();
        auto mj = g_nodes.Get(j)->GetObject<MobilityModel>();
        const Vector pi = mi->GetPosition();
        const Vector pj = mj->GetPosition();
        const double dx = pi.x - pj.x;
        const double dy = pi.y - pj.y;
        const double dist = std::sqrt(dx * dx + dy * dy);
        const double freq = isKu ? g_freqKuGhz : g_freqLGhz;
        const double physicalLoss = TwoRaySeaLoss(dist, pi.z, pj.z, freq, g_seaState, g_polV);

        const double tSince = now - (g_transitionT.count(pr) ? g_transitionT.at(pr) : -1e9);
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

        auto& lossModel = isKu ? g_lossKu : g_lossL;
        lossModel->SetLoss(mi, mj, loss, true);
        if (shouldBeOn)
        {
            ++nOn;
        }
    }

    NS_LOG_UNCOND("t=" << now << "s  step " << step << ": " << touch.size()
                       << " links tracked, " << nOn << " currently in the losGeneration.py's LoS graph");

    // Keep ticking until just before the run ends; no point scheduling a tick
    // that will never fire before Simulator::Stop().
    if (now + g_refreshS < g_simTimeGlobal)
    {
        Simulator::Schedule(Seconds(g_refreshS), &RefreshLoss);
    }
}

int
main(int argc, char* argv[])
{
    // ----------------------------------------------------------- CLI defaults
    std::string dir = "ns3_export";
    std::string out = "result";
    std::string routing = "olsr"; // olsr | aodv
    std::string flowSpec = "all"; // "all" (every ship pair) or "0:4,1:3,..."
    std::string modeL = "OfdmRate6Mbps";
    std::string modeKu = "OfdmRate6Mbps";
    std::string rate = "100kbps";
    uint32_t pktSize = 512;
    double warmup = 30.0;
    double simTime = 0.0; // 0 -> derived from the CSV (nSteps * stepS)
    uint32_t run = 1;
    bool useUav = true;
    bool printRoutes = false;
    bool rtscts = false;

    // ---- radio parameters (tx power values given in the scenario brief; the
    // rest - antenna gains, noise figure - are PLACEHOLDERS. Set them to your
    // actual hardware figures via the matching --flag before trusting results.)
    double txLW = 30.0;       // L-band ship radio power (W)
    double txKuW = 50.0;      // Ku-band radio power, ships and UAV (W)
    double gainL = 3.0;       // L-band antenna gain, dBi (assumed)
    double gainKuShip = 30.0; // Ku ship antenna gain, dBi (assumed)
    double gainKuUav = 25.0;  // Ku UAV antenna gain, dBi (assumed)
    double noiseFigure = 7.0; // receiver noise figure, dB (ns-3 default)

    CommandLine cmd(__FILE__);
    cmd.AddValue("dir", "Folder with nodes.csv and links.csv", dir);
    cmd.AddValue("out", "Prefix for result files", out);
    cmd.AddValue("routing", "olsr | aodv", routing);
    cmd.AddValue("flows", "'all' or comma list of src:dst", flowSpec);
    cmd.AddValue("useUav", "1 = keep UAV links, 0 = ships-only baseline", useUav);
    cmd.AddValue("rate", "Per-flow CBR rate", rate);
    cmd.AddValue("pktSize", "Packet size (bytes)", pktSize);
    cmd.AddValue("modeL", "Fixed 802.11a data rate on L-band", modeL);
    cmd.AddValue("modeKu", "Fixed 802.11a data rate on Ku-band", modeKu);
    cmd.AddValue("freqL", "L-band centre frequency (GHz)", g_freqLGhz);
    cmd.AddValue("freqKu", "Ku-band centre frequency (GHz)", g_freqKuGhz);
    cmd.AddValue("txL", "L-band transmit power (W)", txLW);
    cmd.AddValue("txKu", "Ku-band transmit power (W)", txKuW);
    cmd.AddValue("gainL", "L-band antenna gain (dBi)", gainL);
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
                "(smaller = smoother SNR tracking, more scheduler events; the losGeneration.py's "
                "5-min LoS graph is unaffected either way)",
                g_refreshS);
    cmd.AddValue("rampS",
                "Seconds over which a link fades in/out (in dB) when it appears/disappears "
                "in the losGeneration.py's LoS graph, instead of switching instantly",
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

    // links.csv: one row per (step, i, j, distance) for each LoS edge the
    // notebook found clear. Antenna heights (h1, h2) aren't in this file, so
    // they're filled in here from the nodes.csv positions just loaded, using
    // each endpoint's z-coordinate at the same step.
    g_edges.assign(nSteps, {});
    {
        std::ifstream f(dir + "/links.csv");
        NS_ABORT_MSG_IF(!f.is_open(), "cannot open " << dir << "/links.csv");
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line))
        {
            auto c = Split(line, ',');
            if (c.size() < 5)
            {
                continue;
            }
            const uint32_t s = std::stoul(c[0]);
            const uint32_t ni = std::stoul(c[2]);
            const uint32_t nj = std::stoul(c[3]);
            const double dist = std::stod(c[4]);
            const double h1 = (ni < nNodes && s < nSteps) ? pos[ni][s].z : 0.0;
            const double h2 = (nj < nNodes && s < nSteps) ? pos[nj][s].z : 0.0;
            g_edges[s].push_back({ni, nj, dist, h1, h2});
        }
    }
    const uint32_t nShips = nNodes - 1; // UAV is the last node
    if (simTime <= 0.0)
    {
        simTime = nSteps * g_stepS;
    }
    g_tx.assign(nSteps, 0);
    g_rx.assign(nSteps, 0);
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
                const bool isKu = (e.i == nNodes - 1 || e.j == nNodes - 1);
                if (isKu && !g_useUav)
                {
                    continue;
                }
                const double f = isKu ? g_freqKuGhz : g_freqLGhz;
                const double fsplDb = Fspl(e.dist, f);
                const double twoRayDb = TwoRaySeaLoss(e.dist, e.h1, e.h2, f, g_seaState, g_polV);
                const double rx = isKu ? (txKuDbm + gainKuShip + gainKuUav - twoRayDb)
                                       : (txLDbm + 2.0 * gainL - twoRayDb);
                const double snr = rx - noiseDbm;
                (isKu ? minSnrKu : minSnrL) = std::min(isKu ? minSnrKu : minSnrL, snr);
                lb << k << "," << e.i << "," << e.j << "," << (isKu ? "Ku" : "L") << ","
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

    YansWifiPhyHelper phyL;
    YansWifiPhyHelper phyKuShip;
    YansWifiPhyHelper phyKuUav; // UAV's Ku PHY has its own antenna gain (gainKuUav != gainKuShip)
    setPhy(phyL, chL, txLDbm, gainL);
    setPhy(phyKuShip, chKu, txKuDbm, gainKuShip);
    setPhy(phyKuUav, chKu, txKuDbm, gainKuUav);

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac"); // no AP - every radio is a peer

    // Install L first so it becomes interface 1 (OLSR/AODV's "main address")
    // on ships; devKu is installed on ships and then the UAV, in that order,
    // so its indices line up with ships followed by the UAV throughout.
    NetDeviceContainer devL = makeWifi(modeL).Install(phyL, mac, ships);
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
    addr.SetBase("10.1.1.0", "255.255.255.0"); // L-band subnet: ships only
    Ipv4InterfaceContainer ifL = addr.Assign(devL);
    addr.SetBase("10.1.2.0", "255.255.255.0"); // Ku-band subnet: ships then UAV
    Ipv4InterfaceContainer ifKu = addr.Assign(devKu);
    // Both subnets are 10.1.x.NODE+1, so (last octet - 1) recovers the node
    // index from either address; used below when labelling FlowMonitor flows.

    // ------------------------------------------------------------------ traffic
    const uint16_t port = 9000;
    std::vector<std::pair<uint32_t, uint32_t>> flows;
    if (flowSpec == "all")
    {
        // Default: one flow per unordered ship pair (a full mesh of traffic
        // demands). The UAV never originates or sinks traffic - it only relays.
        for (uint32_t i = 0; i < nShips; ++i)
        {
            for (uint32_t j = i + 1; j < nShips; ++j)
            {
                flows.emplace_back(i, j);
            }
        }
    }
    else
    {
        for (const auto& tok : Split(flowSpec, ','))
        {
            auto sd = Split(tok, ':');
            NS_ABORT_MSG_IF(sd.size() != 2, "bad flow spec: " << tok);
            flows.emplace_back(std::stoul(sd[0]), std::stoul(sd[1]));
        }
    }

    // Every ship runs a UDP sink so it can receive any flow addressed to it.
    for (uint32_t s = 0; s < nShips; ++s)
    {
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        auto app = sink.Install(g_nodes.Get(s));
        app.Get(0)->TraceConnectWithoutContext("Rx", MakeCallback(&OnRx));
        app.Start(Seconds(0.0));
    }
    // Each flow is a constant-bit-rate UDP stream. It's addressed to the
    // destination's L-band interface address; OLSR/AODV then routes to it
    // over whichever band(s) currently connect the two ships (possibly via
    // the UAV's Ku hop, possibly multi-hop over L-band, possibly both).
    for (const auto& fl : flows)
    {
        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(ifL.GetAddress(fl.second), port));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate(rate)));
        onoff.SetAttribute("PacketSize", UintegerValue(pktSize));
        auto app = onoff.Install(g_nodes.Get(fl.first));
        app.Get(0)->TraceConnectWithoutContext("Tx", MakeCallback(&OnTx));
        app.Start(Seconds(warmup));        // give routing time to converge first
        app.Stop(Seconds(simTime - 1.0));  // stop just before the sim ends
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

    // FlowMonitor identifies flows by (src addr, dst addr, ports, protocol).
    // Because a logical ship-to-ship flow can be routed over either band, its
    // *source* address can change (L-band vs Ku-band interface) if the route
    // changes mid-run - FlowMonitor then reports that as a second, separate
    // flow. Agg merges everything back down to one row per logical
    // (ship, ship) pair regardless of which interface carried it.
    struct Agg
    {
        uint64_t tx = 0;
        uint64_t rx = 0;
        uint64_t rxBytes = 0;
        uint64_t fwd = 0;
        double delay = 0.0;
    };
    std::map<std::pair<int, int>, Agg> agg;
    for (const auto& kv : mon->GetFlowStats())
    {
        auto t = cls->FindFlow(kv.first);
        if (t.destinationPort != port)
        {
            continue; // skip OLSR/AODV control traffic, which FlowMonitor also sees
        }
        const int src = (t.sourceAddress.Get() & 0xFF) - 1;
        const int dst = (t.destinationAddress.Get() & 0xFF) - 1;
        Agg& a = agg[{src, dst}];
        a.tx += kv.second.txPackets;
        a.rx += kv.second.rxPackets;
        a.rxBytes += kv.second.rxBytes;
        a.fwd += kv.second.timesForwarded;
        a.delay += kv.second.delaySum.GetSeconds();
    }

    // Offered load is computed from the traffic *settings* (rate, packet
    // size, on/off window), NOT from what the apps or FlowMonitor actually
    // observed. This matters: OnOffApplication only fires its Tx trace once
    // the socket has accepted a packet, and FlowMonitor only sees packets
    // that reached the IP layer - so a packet addressed to a destination
    // with NO ROUTE AT ALL is invisible to both, and would silently vanish
    // from a naive PDR calculation instead of counting as a loss. Comparing
    // against this independently-computed offered load catches that case.
    const double trafficStart = warmup;
    const double trafficStop = simTime - 1.0;
    const double pktRate = double(DataRate(rate).GetBitRate()) / (8.0 * pktSize);
    auto offeredIn = [&](double t0, double t1) {
        const double lo = std::max(t0, trafficStart);
        const double hi = std::min(t1, trafficStop);
        return hi > lo ? (hi - lo) * pktRate : 0.0;
    };
    const uint64_t offeredPerFlow = static_cast<uint64_t>(std::llround(offeredIn(0.0, simTime)));

    // Per-flow results.
    std::ofstream ff(out + "_flows.csv");
    ff << "src,dst,offered_pkts,sent_pkts,noroute_pkts,rx_pkts,pdr,mean_delay_ms,mean_hops,"
          "throughput_kbps\n";
    uint64_t totOff = 0;
    uint64_t totSent = 0;
    uint64_t totRx = 0;
    double totDelay = 0.0;
    double totFwd = 0.0;
    for (const auto& fl : flows)
    {
        Agg a; // stays all-zero if FlowMonitor never saw this flow (no route, ever)
        auto it = agg.find({static_cast<int>(fl.first), static_cast<int>(fl.second)});
        if (it != agg.end())
        {
            a = it->second;
        }
        const uint64_t off = offeredPerFlow;
        const uint64_t noRoute = off > a.tx ? off - a.tx : 0; // offered but never even sent
        const double pdr = off ? double(a.rx) / off : 0.0;
        const double dms = a.rx ? a.delay * 1000.0 / a.rx : 0.0;
        const double hops = a.rx ? 1.0 + double(a.fwd) / a.rx : 0.0; // approximate: 1 + forwards/rx
        const double kbps = a.rxBytes * 8.0 / (simTime - warmup) / 1000.0;
        ff << fl.first << "," << fl.second << "," << off << "," << a.tx << "," << noRoute << ","
           << a.rx << "," << std::fixed << std::setprecision(4) << pdr << "," << dms << ","
           << hops << "," << kbps << "\n";
        totOff += off;
        totSent += a.tx;
        totRx += a.rx;
        totDelay += a.delay;
        totFwd += a.fwd;
    }
    ff.close();

    // Per-step results: how many L/Ku links were active, and offered vs.
    // delivered packets, for each notebook step. This is the file to read
    // when diagnosing *when* loss happens (e.g. concentrated right after a
    // topology change, vs. spread evenly through a step).
    std::ofstream sf(out + "_steps.csv");
    sf << "step,t_start_s,links_L,links_Ku,offered_pkts,sent_pkts,noroute_pkts,rx_pkts,pdr\n";
    for (uint32_t k = 0; k < nSteps; ++k)
    {
        uint32_t nL = 0;
        uint32_t nKu = 0;
        for (const auto& e : g_edges[k])
        {
            const bool isKu = (e.i == nNodes - 1 || e.j == nNodes - 1);
            if (isKu)
            {
                nKu += g_useUav ? 1 : 0;
            }
            else
            {
                ++nL;
            }
        }
        const double t1 = (k + 1 < nSteps) ? stepTime[k + 1] : simTime;
        const uint64_t off =
            static_cast<uint64_t>(std::llround(flows.size() * offeredIn(stepTime[k], t1)));
        const uint64_t noRoute = off > g_tx[k] ? off - g_tx[k] : 0;
        sf << k << "," << stepTime[k] << "," << nL << "," << nKu << "," << off << "," << g_tx[k]
           << "," << noRoute << "," << g_rx[k] << "," << std::fixed << std::setprecision(4)
           << (off ? double(g_rx[k]) / off : 0.0) << "\n";
    }
    sf.close();

    std::cout << "\n===== SUMMARY (" << (g_useUav ? "with UAV" : "ships only") << ", " << routing
              << ", sea state " << g_seaState << ") =====\n"
              << "PDR (rx / offered)      : " << (totOff ? 100.0 * totRx / totOff : 0.0) << " %\n"
              << "Offered with no route   : "
              << (totOff ? 100.0 * (totOff - std::min(totOff, totSent)) / totOff : 0.0) << " %\n"
              << "PDR of routed packets   : " << (totSent ? 100.0 * totRx / totSent : 0.0)
              << " %  (old metric; hides unreachable flows)\n"
              << "Mean e2e delay          : " << (totRx ? 1000.0 * totDelay / totRx : 0.0)
              << " ms (delivered pkts)\n"
              << "Mean hop count          : " << (totRx ? 1.0 + totFwd / totRx : 0.0) << "\n"
              << "Files                   : " << out << "_flows.csv, " << out << "_steps.csv, "
              << out << "_linkbudget.csv\n";

    Simulator::Destroy();
    return 0;
}