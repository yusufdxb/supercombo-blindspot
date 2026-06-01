# E5 causal intervention: is the collapse AT the summarizer bottleneck?

Inject a chosen `/summarizer/Div_output_0` (== hidden_state) trajectory into the
downstream subgraph (Div + host inputs -> outputs) and measure output-head
activity vs the real_baseline. 219 analysis frames (post-100 warmup).
Real Div = alpha=0 (Subaru source), CARLA Div = alpha=1, from the E5 sweep cache.

Activity ratio = sum of per-element temporal std (condition / real_baseline).
1.0 = real-like; near 0 = collapsed. A condition that RESTORES activity toward
1.0 localises the collapse to the bottleneck VALUE; one that stays near 0
localises it downstream of the bottleneck.

Div-isolating conditions hold the no-bottleneck vision path at CARLA, so
pose/meta (which take that path) are identical across them by design.
`real_div_only` ~= 1.0 is expected BY CONSTRUCTION (recurrent heads never
read the vision feature); it is a consistency check that the graph cut is
clean, not an independent finding. The load-bearing conditions are mu_swap,
scale_swap, and real_history.

| head | carla_baseline | real_div_only | mu_swap | scale_swap | real_history |
|---|---|---|---|---|---|
| accel_t0 | 0.0155 | 1.0000 | 0.0046 | 0.1230 | 1.1240 |
| desired_curv | 0.0015 | 1.0000 | 0.0001 | 0.0295 | 0.0164 |
| lead_prob | 0.3884 | 1.0000 | 0.0119 | 2.1151 | 8.0957 |
| plan | 0.0416 | 1.0000 | 0.0082 | 0.1789 | 1.7474 |
| lane_lines | 0.0266 | 1.0000 | 0.0065 | 0.1836 | 0.9338 |
| road_edges | 0.0493 | 1.0000 | 0.0133 | 0.1981 | 1.4750 |
| lead | 0.0196 | 1.0000 | 0.0039 | 0.0788 | 1.3533 |
| pose | 1.6159 | 1.6159 | 1.6159 | 1.6159 | 1.6159 |
| desire_state | 0.0153 | 1.0000 | 0.0169 | 0.1637 | 0.7759 |
| meta | 2.5472 | 2.5472 | 2.5472 | 2.5472 | 2.5472 |

**Sanity gate:** carla_baseline reproduces the teardown collapse (5/5 heads with activity ratio < 0.1); real_baseline is healthy by construction.

**Verdict:** The recurrent heads are a clean function of the bottleneck stream: a full real Div trajectory reproduces real activity exactly (ratio 1.00, by construction, since these heads never read the vision feature), so there is no independent sim-sensitivity downstream of Div. But the collapse is NOT a recoverable mean-shift: swapping only the per-dim mean of the CARLA bottleneck to real leaves the heads collapsed (median ratio 0.01, vs carla 0.027) -- the 'DC-offset saturates the recurrent state' hypothesis is falsified. Scale-only correction is erratic, not a clean fix. Feeding a real 99-frame history with a CARLA current token recovers most spatial heads (median 1.35) but not curvature, so the temporal buffer dominates and the per-frame bottleneck corruption compounds through it. Net: the recurrent collapse is the FULL distributional corruption of the summarizer bottleneck, not a simple mean or scale offset.
