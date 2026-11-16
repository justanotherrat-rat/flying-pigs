
### Table 2: DE optimiser vs brute-force grid, proxy scores (n = 50 scenarios)

|  | Score (mean) | UAV link length (mean) | Position rel. to fleet centroid (mean) | Compute time (mean) | Evaluations (mean) |
|---|---|---|---|---|---|
| Grid | 107.38 (median 105.30) | 107.38 km | dE=0.43 km, dN=-0.22 km, U=1546 m | 15.37 s | 406 |
| DE (1s) | 110.83 (median 108.55) | 110.83 km | dE=-0.45 km, dN=-0.44 km, U=3533 m | 1.02 s | 29 |
| DE (10s) | 106.57 (median 103.65) | 106.57 km | dE=-0.38 km, dN=-0.47 km, U=1706 m | 6.29 s | 203 |
| DE (60s) | 106.56 (median 103.65) | 106.56 km | dE=-0.39 km, dN=-0.45 km, U=1706 m | 6.30 s | 204 |

### table2_grid_vs_de_proxy: DE minus Grid score, paired

| DE budget | mean diff [95% CI] | Wilcoxon p | better / equal / worse (scenarios) |
|---|---|---|---|
| DE (1s) | +3.45 [+2.61, +4.28] | <0.001 | 0 / 1 / 49 |
| DE (10s) | -0.82 [-1.16, -0.48] | <0.001 | 38 / 0 / 12 |
| DE (60s) | -0.82 [-1.16, -0.48] | <0.001 | 38 / 0 / 12 |

### Table 4: DE optimiser vs midpoint placement, proxy scores (n = 50 scenarios)

|  | Score (mean) | UAV link length (mean) | Position rel. to fleet centroid (mean) | Compute time (mean) | Evaluations (mean) |
|---|---|---|---|---|---|
| Midpoint | 111.66 (median 109.31) | 111.66 km | dE=0.00 km, dN=0.00 km, U=3800 m | 0.03 s | 1 |
| DE (1s) | 110.83 (median 108.55) | 110.83 km | dE=-0.45 km, dN=-0.44 km, U=3533 m | 1.02 s | 29 |
| DE (10s) | 106.57 (median 103.65) | 106.57 km | dE=-0.38 km, dN=-0.47 km, U=1706 m | 6.29 s | 203 |
| DE (60s) | 106.56 (median 103.65) | 106.56 km | dE=-0.39 km, dN=-0.45 km, U=1706 m | 6.30 s | 204 |

### table4_midpoint_vs_de_proxy: DE minus Midpoint score, paired

| DE budget | mean diff [95% CI] | Wilcoxon p | better / equal / worse (scenarios) |
|---|---|---|---|
| DE (1s) | -0.84 [-1.32, -0.35] | <0.001 | 14 / 36 / 0 |
| DE (10s) | -5.10 [-6.36, -3.83] | <0.001 | 50 / 0 / 0 |
| DE (60s) | -5.10 [-6.36, -3.83] | <0.001 | 50 / 0 / 0 |

### Table 3: DE optimiser vs brute-force grid, ns-3 metrics (n = 50 paired scenarios)

|  | Signal strength (dBm) | Latency (ms) | Throughput (kbps) | ETX | PDR (%) | Ships linked at all steps |
|---|---|---|---|---|---|---|
| Grid | -42.3 | 1.518 | 11.49 | 1.000 | 99.86 | 100.0 % |
| DE (10s) | -41.9 | 1.519 | 11.47 | 1.000 | 99.74 | 100.0 % |

### table3_grid_vs_de_ns3: DE minus Grid, paired (mean diff [95% CI], Wilcoxon p)

|  | Signal strength (dBm) | Latency (ms) | Throughput (kbps) | ETX | PDR (%) | Link availability |
|---|---|---|---|---|---|---|
| DE (10s) | +0.394 [+0.191, +0.596] p=<0.001 | +0.00141 [-0.000153, +0.00298] p=0.166 | -0.0129 [-0.0447, +0.0189] p=0.508 | +0 [+0, +0] p=1.000 | -0.112 [-0.389, +0.164] p=0.508 | +0.0 pp p=1.000 |

### Table 5: DE optimiser vs midpoint placement, ns-3 metrics (n = 50 paired scenarios)

|  | Signal strength (dBm) | Latency (ms) | Throughput (kbps) | ETX | PDR (%) | Ships linked at all steps |
|---|---|---|---|---|---|---|
| Midpoint | -42.8 | 1.518 | 11.48 | 1.000 | 99.80 | 99.6 % |
| DE (10s) | -41.9 | 1.519 | 11.47 | 1.000 | 99.74 | 100.0 % |

### table5_midpoint_vs_de_ns3: DE minus Midpoint, paired (mean diff [95% CI], Wilcoxon p)

|  | Signal strength (dBm) | Latency (ms) | Throughput (kbps) | ETX | PDR (%) | Link availability |
|---|---|---|---|---|---|---|
| DE (10s) | +0.915 [+0.536, +1.29] p=<0.001 | +0.00155 [-8.82e-05, +0.00319] p=0.007 | -0.00658 [-0.0476, +0.0344] p=0.575 | +0 [+0, +0] p=1.000 | -0.0572 [-0.414, +0.299] p=0.575 | +0.4 pp p=0.317 |
