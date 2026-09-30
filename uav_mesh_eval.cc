/*
 * WHAT THIS PROGRAM DOES
 * ----------------------------------------------------------------------------
 * It replays, inside ns-3, the ship + UAV line-of-sight (LoS) graph that was 
 * computed offline (nodes.csv / links.csv). ns-3 then simulates real 802.11 
 * MAC behaviour, real routing (OLSR/AODV), and real UDP traffic on top of that
 * graph, and reports packet delivery ratio, delay, hop count, throughput, and 
 * how the network behaves as the topology changes over time.
 *
 * For every edge the notebook says is clear, this program computes a
 * maritime PATH LOSS for that link (see "MARITIME PATH LOSS MODEL" below)
 * and writes it into the matching band's ns-3 MatrixPropagationLossModel.
 * Antenna gain, tx power and noise figure are then applied by the ns-3 PHY
 * as usual, so ns-3 decides whether a given LoS link actually has enough 
 * SNR to be usable. Edges the notebook says are NOT clear get an 
 * arbitrarily huge loss, i.e. they are inaudible.
 * 
 *  RADIO MODEL
 * ----------------------------------------------------------------------------
 *   Ship <-> ship : L-band  (1.5 GHz default), 30 W tx power, ships-only channel
 *   Ship <-> UAV  : L-band  (same channel as above - the UAV is now on it too)
 *                   AND/OR Ku-band (16 GHz default), 50 W tx power, own channel
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
 *      phase), giving a total path loss.
 *
 * This is still an idealisation: it assumes a flat sea (no evaporation duct, 
 * no atmospheric refraction beyond the notebook's radio-horizon check, no 
 * swell shadowing, single-bounce only). See the CAVEATS block further down for 
 * what it does and doesn't capture.
 *
 * SMALL-SCALE FADING AND ANTENNA POINTING LOSS
 * ----------------------------------------------------------------------------
 * The two-ray model gives each link's MEAN received power. Two further
 * effects scatter real links around that mean, on different time scales:
 * -> Fading (per packet): a SeaFadingLossModel is chained after the matrix loss
 *     on both channels and multiplies each packet's received power by a
 *     unit-mean Nakagami-m (gamma) draw, so the mean is unchanged and packets
 *     spread above and below it. m comes from a Rician K-factor (--fadeKssDb
 *     for ship-ship, --fadeKsuDb for links touching the UAV); K = -100 dB
 *     gives Rayleigh, a large K gives almost no fading. --fading=0 disables it.
 * -> Pointing loss (slow): only the Ku dishes are directional. Each link keeps
 *     a 2-axis pointing error at the ship end and at the UAV end, each a
 *     Gauss-Markov process (RMS --pointSigmaShipDeg / --pointSigmaUavDeg per
 *     axis, correlation time --pointTauS) applied through RefreshLoss(). Loss
 *     = 12 (error / beamwidth)^2 dB per end, beamwidth from the antenna gain,
 *     capped at 30 dB. L-band antennas are treated as omnidirectional.
 * The link-budget CSV reports the mean pointing loss and the probability that
 * fading pushes a packet below --reqSnrDb.
 *
 * TRAFFIC MODEL
 * ----------------------------------------------------------------------------
 *   Telemetry    : every ship sends a small status packet to every other
 *                  node (ships AND the UAV) every --telemetryPeriodS seconds
 *                  (default 5s). One unicast copy per destination, addressed
 *                  to its L-band address; routing relays each across as many
 *                  hops as needed. 
 *                  Telemetry nPDR = receptions / (rounds * (nNodes - 1)) per ship.
 *   Data transfer: every ship sends a burst of --dataBytes worth of packets
 *                  to the UAV every --dataPeriodMin minutes (default 6 min),
 *                  addressed to the UAV's Ku-band interface.
 * See TRAFFIC MODEL functions below (PeriodicTelemetry / PeriodicDataTransfer)
 * for why these use raw sockets rather than OnOffHelper.
 *
 * USAGE
 * ----------------------------------------------------------------------------
 * Place in ns-3's scratch/ and run from the ns-3 root:
 * -> ./ns3 run "scratch/uav_mesh_eval --dir=ns3_export --useUav=1 --out=with_uav"
 * -> --PrintHelp for the full list of --flags
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
#include <tuple>
#include <utility>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UavMeshEval");

// One LoS edge for each (i, j) per step, as read from links.csv. 
// TWO separate Edge entries for the same (i, j) in the same step, one L and one Ku, if both bands are clear simultaneously.
struct Edge
{
    uint32_t i;
    uint32_t j;
    bool isKu;   // true = Ku-band link, false = L-band link
    double dist; // horizontal-equivalent distance between i and j, metres
    double h1;   // antenna height of node i, metres above sea level
    double h2;   // antenna height of node j, metres above sea level
};

// These are "global" - everything here is set once during setup in main() and only read afterwards.
static NodeContainer g_nodes;                    // ships 0 to nShips-1, then the UAV 
static Ptr<MatrixPropagationLossModel> g_lossL;  // L-band channel's loss matrix (ships AND UAV)
static Ptr<MatrixPropagationLossModel> g_lossKu; // Ku-band channel's loss matrix (ships AND UAV)
static std::vector<std::vector<Edge>> g_edges;   // g_edges[step] = LoS edges active in that step
static bool g_useUav = true;                     // false => any edge touching the UAV is skipped
static double g_freqLGhz = 1.5;                  // L-band centre frequency
static double g_freqKuGhz = 16.0;                // Ku-band centre frequency
static double g_noLinkDb = 250.0;                // loss used for "no LoS" pairs (effectively infinite)
static double g_stepS = 300.0;                   // seconds between notebook steps (topology snapshots)
static int g_seaState = 3;                       // Douglas sea state 0 (calm) .. 9 (phenomenal), for roughness
static bool g_polV = true;                       // Most simple ship/UAV whip or blade antennas are vertically polarised so g_polV defaults to true.
static std::vector<uint64_t> g_tx;               // telemetry packets sent, bucketed by step
static std::vector<uint64_t> g_rx;               // telemetry packets received (summed over all receivers), bucketed by step

// --- continuous loss refresh (see RefreshLoss() below) ---------------------
static double g_refreshS = 5.0; // how often we recompute path loss from live position
static double g_rampS = 20.0;   // how long a link fades in/out when it starts/stops having LoS
static double g_simTimeGlobal = 0.0; // simTime, so RefreshLoss knows when to stop rescheduling itself

// --- state used to detect on/off transitions and time the fade -------------
// (i, j, isKu) because a ship-UAV pair's L-band and Ku-band links are independent.
using LinkKey = std::tuple<uint32_t, uint32_t, bool>;
static std::map<LinkKey, bool> g_linkOn;
static std::map<LinkKey, double> g_transitionT;

// --- small-scale fading and antenna pointing loss (see the models below) ---
static bool g_fading = true;          // chain a per-packet fading model onto both channels
static double g_fadeKssDb = 6.0;      // Rician K-factor, ship<->ship links (dB) - PLACEHOLDER
static double g_fadeKsuDb = 10.0;     // Rician K-factor, ship<->UAV links (dB) - PLACEHOLDER
static double g_pointSigmaShipDeg = 1.0; // ship Ku dish RMS pointing error PER AXIS (deg) - PLACEHOLDER
static double g_pointSigmaUavDeg = 1.5;  // UAV Ku dish RMS pointing error PER AXIS (deg) - PLACEHOLDER
static double g_pointTauS = 20.0;     // pointing-error correlation time (s)
static double g_gainKuShipDb = 30.0;  // Ku ship antenna gain, sets its beamwidth (set from --gainKuShip)
static double g_gainKuUavDb = 25.0;   // Ku UAV antenna gain, sets its beamwidth (set from --gainKuUav)
static Ptr<NormalRandomVariable> g_pointRv; // N(0,1) draws for the pointing-error process

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

// Convert Watts to dBm. 0 dBm = 1 mW, so dBm = 10 log10(W * 1000).
static double
WattsToDbm(double w)
{
    return 10.0 * std::log10(w * 1000.0);
}

// Complex relative permittivity of seawater, eps_r = eps' - j*eps' --> (eps', sigma) since the imaginary part is frequency-dependent
// salinity ~35 ppt, ~20 C are representative values from (Ulaby, Moore & Fung, "Microwave Remote Sensing", vol. 3; Klein & Swift 1977)
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

// Complex Fresnel reflection coefficient of a flat seawater surface at grazing angle 
// This is the standard two-medium (air over seawater) Fresnel formula:
//   R_h = (sin(psi) - sqrt(eps - cos^2(psi))) / (sin(psi) + sqrt(eps - cos^2(psi)))   [horizontal]
//   R_v = (eps*sin(psi) - sqrt(eps - cos^2(psi))) / (eps*sin(psi) + sqrt(eps - cos^2(psi)))  [vertical]

// Horizontal polarisation reflects strongly (|R| approx 1) at all practical maritime 
// grazing angles. Vertical polarisation has a Brewster-angle dip: its reflectivity 
// falls sharply once the grazing angle exceeds roughly sqrt(1/eps_r) radians.

static std::complex<double>
FresnelReflectionCoefficient(double grazingRad, double freqGhz, bool vertical)
{
    SeaDielectric sw = SeaWaterDielectric(freqGhz);
    const double eps0 = 8.8541878128e-12; // vacuum permittivity, F/m
    const double omega = 2.0 * M_PI * freqGhz * 1e9;
    // eps'' = sigma / (omega * eps0); permittivity has a negative imaginary part under the e^{+jwt} convention used.
    std::complex<double> eps(sw.epsR, -sw.sigma / (omega * eps0));

    const double sinPsi = std::sin(grazingRad);
    const double cosPsi = std::cos(grazingRad);
    const std::complex<double> root = std::sqrt(eps - cosPsi * cosPsi);

    const std::complex<double> Rh = (sinPsi - root) / (sinPsi + root);
    const std::complex<double> Rv = (eps * sinPsi - root) / (eps * sinPsi + root);
    return vertical ? Rv : Rh;
}

// Approximate RMS sea-surface wave height (metres) for a given Douglas sea state (0 = glassy calm, 9 = phenomenal/hurricane). 
// This is coarse (wave heights vary a great deal with fetch and wind duration for the same "sea state" label)
// Replace with sigmaH = Hs / 4 if you have measured/forecast significant wave height Hs.
static double
SeaStateRmsHeight(int seaState)
{
    static const double table[] = {0.0, 0.05, 0.15, 0.4, 1.0, 1.75, 3.0, 5.0, 8.0, 12.0};
    const int s = std::min(std::max(seaState, 0), 9);
    return table[s];
}

// Ament's (1953) roughness factor: the reflection because a rough sea scatters part of the reflected energy away from the receiver. 
//      rho_s = exp(-2*(2*pi*sigmaH*sin(psi)/lambda)^2)
// 1 for a perfectly smooth surface (sigmaH -> 0)
// 0 as the surface gets rough relative to a wavelength
static double
RoughnessFactor(double grazingRad, double freqGhz, int seaState)
{
    const double lambda = 299792458.0 / (freqGhz * 1e9);
    const double sigmaH = SeaStateRmsHeight(seaState);
    const double g = (2.0 * M_PI * sigmaH * std::sin(grazingRad)) / lambda;
    return std::exp(-2.0 * g * g);
}

// --- Total two-ray (direct + sea-reflected) path loss in dB -----------------------------------------
//   d           horizontal separation, metres
//   h1, h2      antenna heights above the sea surface, metres
//   freqGhz     link frequency, GHz
//   seaState    Douglas sea state
//   vertical    true = vertical polarisation, false = horizontal
//
// Method (Blake ch. 13 / Kerr ch. 2, the standard maritime two-ray formula):
//      r1 = direct-ray path length
//      r2 = sea-reflected-ray path length 
//   The two rays are summed relative to the direct ray's free-space field:
//      E_total / E_direct = 1 + rho * exp(-j*dPhi) * (r1/r2)
//   Where 
//      rho = (Fresnel reflection coefficient) * (roughness factor)
//      dPhi ~= 4*pi*h1*h2/(lambda*d) OR dPhi = 2*pi*(r2-r1)/lambda for short range
//   RETURN Path loss: FSPL(r1) - 20*log10(|1 + rho*exp(-j dPhi)*(r1/r2)|)
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
        dPhi = (4.0 * M_PI * h1 * h2) / (lambda * d); // stable asymptotic form 
    }
    else
    {
        dPhi = 2.0 * M_PI * (r2 - r1) / lambda; // fine at short range, where r2-r1 isn't tiny
    }

    const std::complex<double> reflCoef = FresnelReflectionCoefficient(grazing, freqGhz, vertical);
    const double rough = RoughnessFactor(grazing, freqGhz, seaState);
    const std::complex<double> rho = reflCoef * rough;

    const std::complex<double> sumTerm = 1.0 + rho * std::polar(1.0, -dPhi) * (r1 / r2);
    // Guard against |sumTerm| -> 0, loss -> infinity: clamp so a single lobe null doesn't turn into a "no LoS" edge, since real seas are never perfectly specular.
    const double factor = std::max(std::abs(sumTerm), 1.0e-3);

    return Fspl(r1, freqGhz) - 20.0 * std::log10(factor);
}
// CAVEATS - what the two-ray model above does NOT capture:
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
//   - Fading is per-packet i.i.d. Nakagami-m (see SeaFadingLossModel), not a
//     time-correlated Doppler process, and there is no shadowing/blockage
//     beyond what the notebook's LoS graph already excludes.
//   - Pointing error is Gaussian and independent per link; it does not depend
//     on ship motion or sea state, and the UAV is assumed to have one
//     steerable aperture per ship link.
//   - Sea state is a single global knob (g_seaState / --seaState), not
//     spatially or temporally varying, and the RMS-height table is a coarse
//     approximation - see SeaStateRmsHeight's comment for how to replace it
//     with a measured significant wave height.
//   - Antenna heights are taken directly from the notebook's ENU z-coordinate
//     (altitude above its local sea-level reference), which assumes that
//     reference is close enough to true sea level for the geometry to be
//     valid; ship deck motion (roll/pitch/heave) is not modelled.

// ===========================================================================
// SMALL-SCALE FADING (per packet) AND ANTENNA POINTING LOSS (slow)
// ===========================================================================
// The two-ray model above gives the MEAN received power of a link. Real links
// scatter around that mean, on two very different time scales, so they are
// modelled in two different places:
//
//  * Small-scale fading changes on the order of a millisecond (coherence time
//    ~ 1 / Doppler; at 1.5 GHz with a ~40 m/s UAV that is a couple of ms, at
//    16 GHz well under a millisecond). That is about the airtime of one
//    packet, so it is applied per packet by a PropagationLossModel chained
//    after the matrix loss model (SeaFadingLossModel below).
//  * Pointing error of a narrow-beam dish (ship roll/pitch/yaw residual after
//    stabilisation, UAV attitude, tracking-loop error) drifts over seconds.
//    It is applied in RefreshLoss(), which already runs every g_refreshS
//    seconds - the right time scale, and free of any per-packet cost.

// Nakagami-m parameter that matches a Rician K-factor (Kdb in dB).
// m = (K+1)^2 / (2K+1): K = 0 (no line-of-sight component) gives m = 1 (Rayleigh), and m grows as K grows (a pure LoS channel with no fading). 
// Nakagami-m is used instead of true Rician only because ns-3 ships a gamma random variable, and the two have near-identical fade-depth statistics for this purpose.
static double
KdbToNakagamiM(double kDb)
{
    const double k = std::pow(10.0, kDb / 10.0);
    return (k + 1.0) * (k + 1.0) / (2.0 * k + 1.0);
}

// Multiplies the received POWER by (Nakagami-m power fading), drawn 
// independently for every packet and receiver. The mean is preserved, 
// so the two-ray path loss stays the average; fading only spreads 
// packets above and below it. Two classes of link get different m: 
// those touching the UAV (strong LoS, high K) and ship-to-ship links 
// (grazing, sea-surface multipath, lower K).
//
// Limitation: draws are independent per packet, i.e. block fading with a
// block equal to one packet. Ok for the UAV links (coherence time about 
// one packet) but pessimistic about burst structure for ship-to-ship links, 
// whose relative speed is small enough that real fades would correlate 
// across many packets.
class SeaFadingLossModel : public PropagationLossModel
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::SeaFadingLossModel")
                                .SetParent<PropagationLossModel>()
                                .SetGroupName("Propagation")
                                .AddConstructor<SeaFadingLossModel>();
        return tid;
    }

    SeaFadingLossModel()
        : m_shipShipM(1.0),
          m_shipUavM(1.0)
    {
        m_rv = CreateObject<GammaRandomVariable>();
    }

    void Configure(double mShipShip, double mShipUav, Ptr<MobilityModel> uavMobility)
    {
        m_shipShipM = mShipShip;
        m_shipUavM = mShipUav;
        m_uav = uavMobility;
    }

  private:
    double DoCalcRxPower(double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override
    {
        const bool touchesUav = (a == m_uav || b == m_uav);
        const double m = touchesUav ? m_shipUavM : m_shipShipM;
        if (m > 1.0e3)
        {
            return txPowerDbm; // K so large the channel is effectively unfaded
        }
        const double meanW = std::pow(10.0, (txPowerDbm - 30.0) / 10.0);
        // Gamma(alpha = m, beta = mean / m) has mean exactly meanW.
        const double drawnW = std::max(m_rv->GetValue(m, meanW / m), 1.0e-40);
        return 10.0 * std::log10(drawnW) + 30.0;
    }

    int64_t DoAssignStreams(int64_t stream) override
    {
        m_rv->SetStream(stream);
        return 1;
    }

    double m_shipShipM;
    double m_shipUavM;
    Ptr<MobilityModel> m_uav;
    Ptr<GammaRandomVariable> m_rv;
};
NS_OBJECT_ENSURE_REGISTERED(SeaFadingLossModel);

// Half-power beamwidth (degrees) of a pencil-beam antenna from its gain.
// G ~= 27000 / theta^2
static double
BeamwidthDeg(double gainDbi)
{
    return std::sqrt(27000.0 / std::pow(10.0, gainDbi / 10.0));
}

// Pointing loss in dB for one antenna whose pointing error is (exDeg, eyDeg)
// off boresight: the usual parabolic main-lobe approximation
// 12 * (error / beamwidth)^2, capped at 30 dB because the approximation
// breaks down far off-axis, where the pattern is really a sidelobe floor.
static double
PointingLossDb(double exDeg, double eyDeg, double beamwidthDeg)
{
    const double e2 = exDeg * exDeg + eyDeg * eyDeg;
    return std::min(12.0 * e2 / (beamwidthDeg * beamwidthDeg), 30.0);
}

// Pointing-error state for one Ku link: an independent 2-axis error at the
// ship's dish and at the UAV's dish (the UAV is assumed to track each ship
// independently, i.e. one steerable aperture per link).
struct PointState
{
    double shipEx = 0.0, shipEy = 0.0;
    double uavEx = 0.0, uavEy = 0.0;
    double lastT = -1.0; // < 0 means "not initialised yet"
};
static std::map<LinkKey, PointState> g_pointState;

// Advances the link's pointing-error process to time `now` and returns the
// total pointing loss (ship end + UAV end) in dB. Each axis is a first-order
// Gauss-Markov process: x' = rho*x + sqrt(1 - rho^2) * sigma * N(0,1), with
// rho = exp(-dt / tau). That keeps the stationary standard deviation at
// exactly sigma while making the error drift smoothly with correlation time
// tau instead of jumping to an independent value every tick. Only Ku links
// are directional here; the L-band antennas are treated as omnidirectional.
static double
UpdatePointingLoss(const LinkKey& key, double now)
{
    if (!g_pointRv || (g_pointSigmaShipDeg <= 0.0 && g_pointSigmaUavDeg <= 0.0))
    {
        return 0.0;
    }
    PointState& st = g_pointState[key];
    if (st.lastT < 0.0)
    {
        // first touch: start from the stationary distribution
        st.shipEx = g_pointSigmaShipDeg * g_pointRv->GetValue();
        st.shipEy = g_pointSigmaShipDeg * g_pointRv->GetValue();
        st.uavEx = g_pointSigmaUavDeg * g_pointRv->GetValue();
        st.uavEy = g_pointSigmaUavDeg * g_pointRv->GetValue();
    }
    else
    {
        const double dt = std::max(0.0, now - st.lastT);
        const double rho = std::exp(-dt / g_pointTauS);
        const double q = std::sqrt(std::max(0.0, 1.0 - rho * rho));
        st.shipEx = rho * st.shipEx + q * g_pointSigmaShipDeg * g_pointRv->GetValue();
        st.shipEy = rho * st.shipEy + q * g_pointSigmaShipDeg * g_pointRv->GetValue();
        st.uavEx = rho * st.uavEx + q * g_pointSigmaUavDeg * g_pointRv->GetValue();
        st.uavEy = rho * st.uavEy + q * g_pointSigmaUavDeg * g_pointRv->GetValue();
    }
    st.lastT = now;
    return PointingLossDb(st.shipEx, st.shipEy, BeamwidthDeg(g_gainKuShipDb)) +
           PointingLossDb(st.uavEx, st.uavEy, BeamwidthDeg(g_gainKuUavDb));
}

// Expected (mean) total pointing loss of a Ku link, ignoring the 30 dB cap.
// For a 2-axis Gaussian error E[e^2] = 2 sigma^2 per end. 
static double
MeanPointingLossDb()
{
    const double bs = BeamwidthDeg(g_gainKuShipDb);
    const double bu = BeamwidthDeg(g_gainKuUavDb);
    return 12.0 * 2.0 * g_pointSigmaShipDeg * g_pointSigmaShipDeg / (bs * bs) +
           12.0 * 2.0 * g_pointSigmaUavDeg * g_pointSigmaUavDeg / (bu * bu);
}

// Regularised lower incomplete gamma function P(a, x), by its power series
// (converges quickly for the modest x used here). Needed only to turn the
// Nakagami-m power distribution into an outage probability.
static double
RegLowerIncGamma(double a, double x)
{
    if (x <= 0.0)
    {
        return 0.0;
    }
    double term = 1.0 / a;
    double sum = term;
    for (int n = 1; n < 2000; ++n)
    {
        term *= x / (a + n);
        sum += term;
        if (term < 1.0e-15 * sum)
        {
            break;
        }
    }
    return std::min(1.0, sum * std::exp(-x + a * std::log(x) - std::lgamma(a)));
}

// Probability that a single packet's instantaneous SNR falls below the decode
// threshold, given the link's mean SNR margin and its Nakagami-m parameter. 
// The fading power is Gamma(m, 1/m), so 
// P(power < 10^(-margin/10)) = P(m, m * 10^(-margin/10))
// Cross-checked against sampling. This is a link-budget diagnostic only. 
// The simulation itself draws fades packet by packet and lets the PHY decide.
static double
FadeOutageProb(double m, double marginDb)
{
    if (marginDb <= -10.0)
    {
        return 1.0; // mean SNR far below threshold: effectively always in outage
    }
    return RegLowerIncGamma(m, m * std::pow(10.0, -marginDb / 10.0));
}

// ===========================================================================
// MAC-LAYER LOSS COUNTERS
// ===========================================================================
// Count every failed transmission attempt (no ACK came back) and every frame 
// dropped after exhausting its retries. Devices are installed L first, then 
// Ku, on every node, so DeviceList index 0 = L-band and 1 = Ku-band
static uint64_t g_macFail[2] = {0, 0}; // failed attempts (no ACK), [0] = L-band, [1] = Ku-band
static uint64_t g_macDrop[2] = {0, 0}; // frames dropped after the retry limit, same indexing

static uint32_t
BandFromContext(const std::string& context)
{
    const std::string key = "DeviceList/";
    const size_t at = context.find(key);
    if (at == std::string::npos)
    {
        return 0;
    }
    return std::stoul(context.substr(at + key.size())) == 0 ? 0 : 1;
}

static void
OnMacDataFailed(std::string context, Mac48Address /*peer*/)
{
    g_macFail[BandFromContext(context)]++;
}

static void
OnMacFinalDataFailed(std::string context, Mac48Address /*peer*/)
{
    g_macDrop[BandFromContext(context)]++;
}

// Convert simulation time to a step index, clamped to the last valid bucket
// (guards the tail end of the run, where Simulator::Now() can be past the 
// final step's nominal start due to floating-point scheduling).
static uint32_t
Bucket()
{
    const uint32_t k = static_cast<uint32_t>(Simulator::Now().GetSeconds() / g_stepS);
    return std::min<uint32_t>(k, g_tx.size() - 1);
}

// Trace callback: count telemetry packets received (any node, any sender),
// bucketed by which notebook step was active when they arrived. Used for the
// per-step breakdown in *_steps.csv.
OnRx(Ptr<const Packet> p, const Address& from)
{
    g_rx[Bucket()]++;
}

// ===========================================================================
// CONTINUOUS LOSS REFRESH (replaces a once-per-step "jump")
// ===========================================================================
// The notebook only re-checks LoS/Fresnel obstruction every g_stepS seconds
// (5 min by default) because that check involves DEM raycasting and is
// expensive. RefreshLoss() exploits that: instead of freezing each link's 
// loss at the value computed from the step's snapshot distance, it re-derives 
// loss every g_refreshS seconds from each node's live, interpolated position.
// When a link's LoS state flips (appears or disappears at a step boundary), 
// instead of an instant jump between "audible" and "inaudible" it fades linearly 
// in dB over g_rampS seconds. Two things this buys:
//   1. Smoother received power - SNR tracks real ship motion within a step
//      instead of stepping in 5-minute jumps.
//   2. A more realistic transition when a link comes up or down
static void
RefreshLoss()
{
    const uint32_t n = g_nodes.GetN();
    const uint32_t uav = n - 1;
    const double now = Simulator::Now().GetSeconds();
    const uint32_t step = std::min<uint32_t>(static_cast<uint32_t>(now / g_stepS),
                                             static_cast<uint32_t>(g_edges.size() - 1));

    // "target" = the sets of (i, j, band) the notebook says should have LoS right now
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

        // Live geometry. Endpoint positions are continuously interpolated by WaypointMobilityModel, so this tracks real motion within a step.
        auto mi = g_nodes.Get(i)->GetObject<MobilityModel>();
        auto mj = g_nodes.Get(j)->GetObject<MobilityModel>();
        const Vector pi = mi->GetPosition();
        const Vector pj = mj->GetPosition();
        const double dx = pi.x - pj.x;
        const double dy = pi.y - pj.y;
        const double dist = std::sqrt(dx * dx + dy * dy);
        const double freq = isKu ? g_freqKuGhz : g_freqLGhz;
        // Mean path loss from the two-ray model, plus current antenna pointing  error which advances on every tick.
        const double pointLoss = isKu ? UpdatePointingLoss(key, now) : 0.0;
        const double physicalLoss =
            TwoRaySeaLoss(dist, pi.z, pj.z, freq, g_seaState, g_polV) + pointLoss;

        const double tSince = now - (g_transitionT.count(key) ? g_transitionT.at(key) : -1e9);
        double loss;
        if (tSince < g_rampS)
        {
            // Mid-fade: blend linearly in dB 
            const double frac = std::clamp(tSince / g_rampS, 0.0, 1.0);
            const double from = shouldBeOn ? g_noLinkDb : physicalLoss;
            const double to = shouldBeOn ? physicalLoss : g_noLinkDb;
            loss = from + (to - from) * frac;
        }
        else
        {
            loss = shouldBeOn ? physicalLoss : g_noLinkDb;
        }

        // isKu picks which channel's loss matrix this pair's link lives on 
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

    // Keep ticking until just before the run ends; no point scheduling a tick that will never fire before Simulator::Stop().
    if (now + g_refreshS < g_simTimeGlobal)
    {
        Simulator::Schedule(Seconds(g_refreshS), &RefreshLoss);
    }
}

// ===========================================================================
// TRAFFIC MODEL: TELEMETRY (unicast fan-out) + DATA TRANSFER (unicast burst)
// ===========================================================================
// Both flows are "N packets every so often" self-rescheduling functions
//
// TELEMETRY is "broadcast to everyone" in the sense that every ship's status
// must reach every other node (all other ships AND the UAV) even when they
// are several hops apart. Each ship instead sends one small UNICAST copy
// to each other node (fan-out) and the routing protocol carries each copy
// across however many hops it takes.
//
// Every ship's send schedule must be staggered (see the call sites in
// main()). If all ships fire at the same simulated instant every period,
// they collide on the shared channel every time.

// Sends one telemetry packet on a connected unicast socket and counts it
// toward the per-step "sent" tally (there is no Application Tx trace to hook).
static void
SendTelemetryPacket(Ptr<Socket> socket, uint32_t bytes)
{
    socket->Send(Create<Packet>(bytes));
    g_tx[Bucket()]++;
}

// One telemetry round for one ship: a copy to every destination, spaced
// `spacing` apart so the ship does not dump all copies into its own MAC queue
// in the same instant, then reschedules itself after `period`. Runs for the
// lifetime of the simulation once started.
static void
PeriodicTelemetry(std::vector<Ptr<Socket>> socks, uint32_t bytes, Time spacing, Time period)
{
    for (uint32_t k = 0; k < socks.size(); ++k)
    {
        Simulator::Schedule(spacing * k, &SendTelemetryPacket, socks[k], bytes);
    }
    Simulator::Schedule(period, &PeriodicTelemetry, socks, bytes, spacing, period);
}

static uint64_t g_dataSendFail = 0; // Socket::Send() returned -1 (see SendOnePacket)

static void
SendOnePacket(Ptr<Socket> socket, uint32_t pktSize)
{
    if (socket->Send(Create<Packet>(pktSize)) < 0)
    {
        g_dataSendFail++; // socket-level failure (e.g. no route found) - never delivered at all
    }
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
    // --- CLI defaults ----------------------------------------------------
    std::string dir = "ns3_export";
    std::string out = "result";
    std::string routing = "olsr"; // olsr | aodv
    std::string modeL = "OfdmRate6Mbps";
    std::string modeKu = "OfdmRate6Mbps";
    double warmup = 30.0;
    double simTime = 0.0; // 0 -> derived from the CSV (nSteps * stepS)
    uint32_t run = 1;
    bool useUav = true;

    // Telemetry: every ship sends a small status packet to every other node
    // (all other ships AND the UAV) every telemetryPeriodS seconds, as one
    // unicast copy per destination that routing relays across the mesh.
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

    // ---- radio parameters ------------------------------
    double txLW = 30.0;       // L-band ship radio power (W)
    double txLUavW = 30.0;    // L-band UAV radio power (W) - ASSUMED equal to ships'; not specified
    double txKuW = 50.0;      // Ku-band radio power, ships and UAV (W)
    double gainL = 3.0;       // L-band ship antenna gain, dBi (assumed)
    double gainLUav = 3.0;    // L-band UAV antenna gain, dBi (assumed equal to a ship's)
    double gainKuShip = 30.0; // Ku ship antenna gain, dBi (assumed)
    double gainKuUav = 25.0;  // Ku UAV antenna gain, dBi (assumed)
    double noiseFigure = 7.0; // receiver noise figure, dB (ns-3 default)
    double reqSnrDb = 5.0;    // SNR needed to decode the fixed data rate; ~5 dB for 6 Mbps 802.11a (BPSK 1/2)

    CommandLine cmd(__FILE__);
    cmd.AddValue("dir", "Folder with nodes.csv and links.csv", dir);
    cmd.AddValue("out", "Prefix for result files", out);
    cmd.AddValue("routing", "olsr | aodv", routing);
    cmd.AddValue("useUav", "1 = keep UAV links (both bands), 0 = ships-only baseline", useUav);
    cmd.AddValue("telemetryPeriodS", "Seconds between each ship's telemetry rounds",
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
    cmd.AddValue("reqSnrDb",
                "Approximate SNR needed to decode the chosen rate; used only by the link-budget "
                "outage-probability column, not by the simulation",
                reqSnrDb);
    cmd.AddValue("fading", "1 = per-packet small-scale fading on both bands, 0 = off", g_fading);
    cmd.AddValue("fadeKssDb",
                "Rician K-factor for ship<->ship links, dB (-100 ~ Rayleigh, large = no fading)",
                g_fadeKssDb);
    cmd.AddValue("fadeKsuDb", "Rician K-factor for ship<->UAV links, dB", g_fadeKsuDb);
    cmd.AddValue("pointSigmaShipDeg",
                "Ship Ku dish RMS pointing error per axis, degrees (0 = perfect pointing)",
                g_pointSigmaShipDeg);
    cmd.AddValue("pointSigmaUavDeg", "UAV Ku dish RMS pointing error per axis, degrees",
                g_pointSigmaUavDeg);
    cmd.AddValue("pointTauS", "Pointing-error correlation time, seconds", g_pointTauS);
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

    // Beamwidths (and so pointing loss) depend on the Ku antenna gains.
    g_gainKuShipDb = gainKuShip;
    g_gainKuUavDb = gainKuUav;
    g_pointRv = CreateObject<NormalRandomVariable>(); // default N(0, 1)
    NS_LOG_UNCOND("Fading " << (g_fading ? "ON" : "OFF") << ": Nakagami-m ship-ship "
                            << KdbToNakagamiM(g_fadeKssDb) << " (K=" << g_fadeKssDb
                            << " dB), ship-UAV " << KdbToNakagamiM(g_fadeKsuDb)
                            << " (K=" << g_fadeKsuDb << " dB). Ku pointing: sigma ship/UAV "
                            << g_pointSigmaShipDeg << "/" << g_pointSigmaUavDeg
                            << " deg per axis, beamwidth " << BeamwidthDeg(gainKuShip) << "/"
                            << BeamwidthDeg(gainKuUav) << " deg, mean loss "
                            << MeanPointingLossDb() << " dB");

    const double txLDbm = WattsToDbm(txLW);
    const double txLUavDbm = WattsToDbm(txLUavW);
    const double txKuDbm = WattsToDbm(txKuW);

    // --- load CSVs ---------------------------------------------------------
    // nodes.csv: one row per (step, node) giving its ENU position x, y, z. 
    // Positions also drive each node's WaypointMobilityModel for visualisation; 
    // actual connectivity comes entirely from the loss matrices.
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
    // the notebook found clear at that band. A ship-UAV pair can appear as 
    // two separate rows in the same step (one L, one Ku) if both bands are 
    // clear simultaneously.
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
    NS_LOG_UNCOND("Loaded " << nShips << " ships + 1 UAV, " << nSteps << " steps, simTime="
                            << simTime << "s, UAV links " << (g_useUav ? "ON" : "OFF")
                            << ", sea state " << g_seaState << ", "
                            << (g_polV ? "vertical" : "horizontal") << " polarisation");

    // --- link-budget report (offline) --------------------------------
    // Precomputes, for every LoS edge at every step, both the plain
    // free-space loss and the maritime two-ray loss actually used in the
    // simulation, plus the resulting received power and SNR.
    const double noiseDbm = -174.0 + 10.0 * std::log10(20e6) + noiseFigure;
    {
        std::ofstream lb(out + "_linkbudget.csv");
        // point_loss_db is the MEAN Ku pointing loss (0 on L-band); snr_db
        // already includes it. outage_prob is the probability that fading
        // drops a packet's SNR below reqSnrDb (see FadeOutageProb).
        lb << "step,i,j,band,dist_km,h1_m,h2_m,fspl_db,two_ray_db,point_loss_db,rx_dbm,snr_db,"
              "outage_prob\n";
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
                const double pointDb = e.isKu ? MeanPointingLossDb() : 0.0;
                const double rx = txDbm + gainOf(e.i) + gainOf(e.j) - twoRayDb - pointDb;
                const double snr = rx - noiseDbm;
                const bool linkTouchesUav = (e.i == nNodes - 1 || e.j == nNodes - 1);
                const double outage = g_fading
                                          ? FadeOutageProb(KdbToNakagamiM(linkTouchesUav ? g_fadeKsuDb
                                                                                        : g_fadeKssDb),
                                                           snr - reqSnrDb)
                                          : (snr < reqSnrDb ? 1.0 : 0.0);
                (e.isKu ? minSnrKu : minSnrL) = std::min(e.isKu ? minSnrKu : minSnrL, snr);
                lb << k << "," << e.i << "," << e.j << "," << (e.isKu ? "Ku" : "L") << ","
                   << std::fixed << std::setprecision(2) << e.dist / 1000.0 << "," << e.h1 << ","
                   << e.h2 << "," << fsplDb << "," << twoRayDb << "," << pointDb << "," << rx
                   << "," << snr << "," << std::setprecision(6) << outage << std::setprecision(2)
                   << "\n";
            }
        }
        NS_LOG_UNCOND("Noise floor " << noiseDbm << " dBm. Worst-case LoS-link SNR (two-ray): "
                                     << "L-band " << (minSnrL > 1e8 ? NAN : minSnrL)
                                     << " dB, Ku-band " << (minSnrKu > 1e8 ? NAN : minSnrKu)
                                     << " dB  (see " << out << "_linkbudget.csv)");
    }

    // --- nodes ---------------------------------------------------------
    g_nodes.Create(nNodes);
    NodeContainer ships;
    for (uint32_t s = 0; s < nShips; ++s)
    {
        ships.Add(g_nodes.Get(s));
    }
    NodeContainer uavNode(g_nodes.Get(nShips));

    // Every node's position over time is a series of waypoints, one per
    // notebook step. This is only used for visualisation. Simulated 
    // connectivity comes entirely from the loss matrices set in
    // RefreshLoss(), not from ns-3 computing distance/loss.
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

    // --- channels + radios ----------------------------------------------------
    // Two independent 802.11 (Yans) channels, one per band, each carrying its
    // own MatrixPropagationLossModel (rewritten continuously by RefreshLoss).
    // A shared little helper builds a channel + loss model pair so the two
    // bands are configured identically apart from which loss matrix is theirs.
    auto makeChannel = [&](Ptr<MatrixPropagationLossModel>& lossOut) -> Ptr<YansWifiChannel> {
        Ptr<YansWifiChannel> ch = CreateObject<YansWifiChannel>();
        lossOut = CreateObject<MatrixPropagationLossModel>();
        lossOut->SetDefaultLoss(g_noLinkDb); // any pair we never explicitly set is "no LoS"
        ch->SetPropagationLossModel(lossOut);
        // Chain per-packet small-scale fading AFTER the matrix loss: ns-3
        // feeds the matrix model's output power into the next model, so the
        // fading model sees the mean (two-ray + pointing) power and spreads
        // packets around it. Each channel gets its own instance, so the two
        // bands fade independently.
        if (g_fading)
        {
            Ptr<SeaFadingLossModel> fade = CreateObject<SeaFadingLossModel>();
            fade->Configure(KdbToNakagamiM(g_fadeKssDb),
                            KdbToNakagamiM(g_fadeKsuDb),
                            g_nodes.Get(g_nodes.GetN() - 1)->GetObject<MobilityModel>());
            lossOut->SetNext(fade);
        }
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

    // --- internet -----------------------------------------------------
    OlsrHelper olsr;

    InternetStackHelper internet;
    if (routing == "aodv")
    {
        AodvHelper aodv;
        internet.SetRoutingHelper(aodv);
    }
    else
    {
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

    // --- traffic -----------------------------------------------------
    const uint16_t telemetryPort = 9001;
    const uint16_t dataPort = 9002;

    // --- Telemetry sinks -------------------------------------------- 
    // Every node (ships AND the UAV) listens on the telemetry port, since telemetry is adressed to all.
    for (uint32_t n = 0; n < nNodes; ++n)
    {
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), telemetryPort));
        auto app = sink.Install(g_nodes.Get(n));
        app.Get(0)->TraceConnectWithoutContext("Rx", MakeCallback(&OnRx));
        app.Start(Seconds(0.0));
    }
    // --- Telemetry senders ---------------------------------------------
    // Start times are STAGGERED: ship s begins s * (period / nShips) seconds after warmup. 
    const Time telemetryPeriod = Seconds(telemetryPeriodS);
    for (uint32_t s = 0; s < nShips; ++s)
    {
        std::vector<Ptr<Socket>> socks;
        for (uint32_t d = 0; d < nNodes; ++d)
        {
            if (d == s)
            {
                continue;
            }
            Ptr<Socket> sock =
                Socket::CreateSocket(g_nodes.Get(s), UdpSocketFactory::GetTypeId());
            sock->Connect(InetSocketAddress(ifL.GetAddress(d), telemetryPort));
            socks.push_back(sock);
        }
        Simulator::Schedule(Seconds(warmup + s * telemetryPeriodS / nShips),
                            &PeriodicTelemetry,
                            socks,
                            telemetryBytes,
                            MilliSeconds(10),
                            telemetryPeriod);
    }

    // --- Data-transfer sink ---------------------------------------------
    // Only the UAV needs to receive these on its Ku-band address. Its stats 
    // are read from FlowMonitor below rather than a manual per-step counter,
    // since it's a clean single-destination unicast flow that FlowMonitor
    // classifies cleanly per ship.
    {
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), dataPort));
        auto app = sink.Install(uavNode);
        app.Start(Seconds(0.0));
    }
    // --- Data-transfer senders ------------------------------------------
    // one unicast socket per ship, connected to the UAV's Ku-band address.
    const Ipv4Address uavKuAddr = ifKu.GetAddress(nShips);
    const uint32_t burstPkts = std::max<uint32_t>(1, dataBytes / dataPktSize);
    for (uint32_t s = 0; s < nShips; ++s)
    {
        Ptr<Socket> sock = Socket::CreateSocket(g_nodes.Get(s), UdpSocketFactory::GetTypeId());
        if (sock->Connect(InetSocketAddress(uavKuAddr, dataPort)) != 0)
        {
            NS_LOG_UNCOND("WARNING: ship " << s << "'s data-transfer socket failed to Connect() "
                                          << "- it will never send anything");
        }
        // Staggered like telemetry: ship s starts s/nShips of a period late,
        // so the five ships' bursts never land on the channel together.
        Simulator::Schedule(Seconds(warmup + s * dataPeriodMin * 60.0 / nShips),
                            &PeriodicDataTransfer,
                            sock,
                            dataPktSize,
                            burstPkts,
                            MilliSeconds(static_cast<int64_t>(dataPktGapMs)),
                            Seconds(dataPeriodMin * 60.0));
    }

    // --- topology schedule --------------------------------------
    // Reads g_edges/g_stepS to know which pairs currently have LoS, 
    // recomputes their loss from live position every g_refreshS 
    // seconds, fades transitions over g_rampS seconds, and reschedules 
    // itself until just before simTime.
    g_simTimeGlobal = simTime;
    Simulator::Schedule(Seconds(0.0), &RefreshLoss);
    // Dump every node's routing table 60 s into each step.
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

    // Hook the MAC failure traces on every node's every Wi-Fi device. The
    // FailSafe variants return false instead of aborting if this ns-3 version
    // names the trace path differently so the run still completes.
    const bool macTraceOk =
        Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
                                "MacTxDataFailed",
                                MakeCallback(&OnMacDataFailed)) &&
        Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
                                "MacTxFinalDataFailed",
                                MakeCallback(&OnMacFinalDataFailed));
    if (!macTraceOk)
    {
        NS_LOG_UNCOND("NOTE: MAC failure traces could not be connected on this ns-3 version; "
                      "*_mac.csv will read zero and is NOT evidence of a loss-free channel.");
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

    // Expected ("offered") counts, computed from the traffic *settings* rather
    // than from what actually got sent - so a packet that never made it onto
    // the wire still counts as offered-but-lost instead of silently vanishing.
    // Start times are staggered per ship, so the number of rounds that fit 
    // before simTime is computed per ship.
    const double avail = std::max(0.0, simTime - warmup);
    auto roundsFor = [&](double startS, double periodS) -> uint64_t {
        return simTime >= startS ? static_cast<uint64_t>(std::floor((simTime - startS) / periodS)) + 1
                                 : 0;
    };
    const double dataPeriodS = dataPeriodMin * 60.0;

    // --- TELEMETRY (unicast fan-out, ship -> every other node) --------------
    // Each ship sends one unicast copy to each of the other (nNodes - 1)
    // nodes per round, so a ship's possible receptions = rounds * (nNodes - 1)
    // and PDR = receptions / possible. Copies are ordinary unicast flows, so
    // FlowMonitor tracks them correctly. Flows are aggregated per SENDING ship, 
    // whichever interface address routing is used as the source.
    struct TelAgg
    {
        uint64_t tx = 0;
        uint64_t rx = 0;
        uint64_t fwd = 0;
        double delay = 0.0;
    };
    std::map<int, TelAgg> telAgg;
    for (const auto& kv : mon->GetFlowStats())
    {
        auto t = cls->FindFlow(kv.first);
        if (t.destinationPort != telemetryPort)
        {
            continue;
        }
        const int src = (t.sourceAddress.Get() & 0xFF) - 1;
        TelAgg& a = telAgg[src];
        a.tx += kv.second.txPackets;
        a.rx += kv.second.rxPackets;
        a.fwd += kv.second.timesForwarded;
        a.delay += kv.second.delaySum.GetSeconds();
    }
    const uint32_t possibleReceivers = nNodes - 1; // everyone except the sender itself
    std::ofstream tf(out + "_telemetry.csv");
    tf << "ship,possible_receptions,sent_pkts,rx_receptions,pdr,mean_delay_ms,mean_hops\n";
    uint64_t telTotRx = 0;
    uint64_t telTotPossible = 0;
    double telTotDelay = 0.0;
    double telTotFwd = 0.0;
    for (uint32_t s = 0; s < nShips; ++s)
    {
        const TelAgg& a = telAgg.count(static_cast<int>(s)) ? telAgg[static_cast<int>(s)] : TelAgg{};
        const uint64_t possible =
            roundsFor(warmup + s * telemetryPeriodS / nShips, telemetryPeriodS) * possibleReceivers;
        const double pdr = possible ? double(a.rx) / possible : 0.0;
        const double dms = a.rx ? a.delay * 1000.0 / a.rx : 0.0;
        const double hops = a.rx ? 1.0 + double(a.fwd) / a.rx : 0.0;
        tf << s << "," << possible << "," << a.tx << "," << a.rx << "," << std::fixed
           << std::setprecision(4) << pdr << "," << dms << "," << hops << "\n";
        telTotRx += a.rx;
        telTotPossible += possible;
        telTotDelay += a.delay;
        telTotFwd += a.fwd;
    }
    tf.close();

    // --- DATA TRANSFER (unicast, ship -> UAV, nominally Ku-band) -----------
    // One unicast flow per ship (destination is always the UAV).
    // g_edges/g_useUav still gate whether a Ku path to the UAV exists.
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
        const uint64_t dataOfferedPkts =
            roundsFor(warmup + s * dataPeriodS / nShips, dataPeriodS) * burstPkts;
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

    // Per-step breakdown (telemetry only). tx/rx come from the manual
    // per-step counters, since FlowMonitor has no time axis. Link counts come
    // from Edge::isKu directly instead of "touches the UAV".
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
        const double t1 = (k + 1 < nSteps) ? stepTime[k + 1] : simTime;
        const double stepDur = std::max(0.0, std::min(t1, simTime) - std::max(stepTime[k], warmup));
        const uint64_t possibleThisStep =
            static_cast<uint64_t>(std::llround(nShips * (stepDur / telemetryPeriodS) *
                                               possibleReceivers));
        sf << k << "," << stepTime[k] << "," << nL << "," << nKu << "," << g_tx[k] << ","
           << g_rx[k] << "," << std::fixed << std::setprecision(4)
           << (possibleThisStep ? double(g_rx[k]) / possibleThisStep : 0.0) << "\n";
    }
    sf.close();

    std::ofstream mf(out + "_mac.csv");
    mf << "band,failed_attempts,frames_dropped_after_retries,trace_connected\n"
       << "L," << g_macFail[0] << "," << g_macDrop[0] << "," << macTraceOk << "\n"
       << "Ku," << g_macFail[1] << "," << g_macDrop[1] << "," << macTraceOk << "\n";
    mf.close();

    std::cout << "\n===== SUMMARY (" << (g_useUav ? "with UAV" : "ships only") << ", " << routing
              << ", sea state " << g_seaState << ") =====\n"
              << "Telemetry PDR (every ship -> every node): "
              << (telTotPossible ? 100.0 * telTotRx / telTotPossible : 0.0) << " %\n"
              << "Telemetry mean e2e delay            : "
              << (telTotRx ? 1000.0 * telTotDelay / telTotRx : 0.0) << " ms\n"
              << "Telemetry mean hop count            : "
              << (telTotRx ? 1.0 + telTotFwd / telTotRx : 0.0) << "\n"
              << "Data-transfer PDR (ship->UAV)       : "
              << (dataTotOff ? 100.0 * dataTotRx / dataTotOff : 0.0) << " %\n"
              << "Data-transfer mean e2e delay        : "
              << (dataTotRx ? 1000.0 * dataTotDelay / dataTotRx : 0.0) << " ms\n"
              << "Data-transfer mean hop count        : "
              << (dataTotRx ? 1.0 + dataTotFwd / dataTotRx : 0.0) << "\n"
              << "Data-transfer socket send failures  : " << g_dataSendFail
              << (g_dataSendFail ? "  <-- packets that never reached the wire at all, check "
                                   "Connect() warnings above"
                                : "") << "\n"
              << "MAC failed attempts L / Ku         : " << g_macFail[0] << " / " << g_macFail[1]
              << (macTraceOk ? "" : "  (trace NOT connected - ignore)") << "\n"
              << "MAC frames dropped after retries    : " << g_macDrop[0] << " / " << g_macDrop[1]
              << "\n"
              << "Files  : " << out << "_telemetry.csv, " << out << "_datatransfer.csv, " << out
              << "_steps.csv, " << out << "_linkbudget.csv, " << out << "_mac.csv\n";

    Simulator::Destroy();
    return 0;
}