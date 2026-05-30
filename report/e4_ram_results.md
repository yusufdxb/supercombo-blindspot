# E4-RAM results: real-to-sim interpolation (RAM source)

Pixel alpha-blend of the RAM real sequence and the CARLA sequence (N=320 frames, 100 warmup discarded). alpha=0 is the real frame, alpha=1 is the CARLA frame.

**Verdict: gradient.** Output activity falls from 0.9 to 0.1 of the real baseline over alpha 0.666 to 0.940 (transition width 0.274; < 0.2 reads as a cliff).

## Comparison with Subaru E4

| Source | a90 | a10 | Transition width | E6 fires-at-alpha | E6 headroom | Verdict |
|---|---|---|---|---|---|---|
| Subaru | 0.784 | 0.799 | 0.015 | 0.550 | 0.234 | cliff |
| RAM | 0.666 | 0.940 | 0.274 | 0.850 | -0.184 | gradient |

## Per-alpha table

| alpha | output activity | feature collapse | feature spread | plan uncertainty |
|---|---|---|---|---|
| 0.0000 | 1.0000 | 0.0000 | 0.40 | 0.4952 |
| 0.1000 | 0.9856 | 0.0881 | 0.41 | 0.5038 |
| 0.1500 | 1.0410 | 0.0057 | 0.46 | 0.5042 |
| 0.2000 | 1.1906 | -0.0241 | 0.49 | 0.5066 |
| 0.3000 | 1.3486 | -0.0541 | 0.52 | 0.5346 |
| 0.4000 | 1.3645 | 0.0909 | 0.49 | 0.5955 |
| 0.4500 | 1.2761 | 0.2673 | 0.45 | 0.6107 |
| 0.5000 | 1.1443 | 0.4890 | 0.37 | 0.6160 |
| 0.6000 | 0.9902 | 0.7610 | 0.23 | 0.6081 |
| 0.7000 | 0.8538 | 0.8896 | 0.17 | 0.5935 |
| 0.7250 | 1.0255 | 0.8939 | 0.17 | 0.5857 |
| 0.7500 | 1.1035 | 0.8988 | 0.16 | 0.5932 |
| 0.8000 | 1.1172 | 0.9171 | 0.15 | 0.5844 |
| 0.8500 | 1.0110 | 0.9455 | 0.13 | 0.5778 |
| 0.8750 | 0.9051 | 0.9797 | 0.08 | 0.5692 |
| 0.9000 | 0.6725 | 0.9902 | 0.04 | 0.5609 |
| 0.9250 | 0.2365 | 0.9981 | 0.01 | 0.5548 |
| 0.9500 | 0.0056 | 0.9999 | 0.00 | 0.5526 |
| 1.0000 | 0.0058 | 1.0000 | 0.00 | 0.5522 |
