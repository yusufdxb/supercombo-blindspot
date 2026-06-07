# Daytime Attractor Diagnostic: H1 + H2

## Regime statistics
- Frames total: 319
- Low-norm (< 0.1): 276 (86.5%)
- High-norm (>= 0.5): 43 (13.5%)
- Gap (0.1-0.5, should be 0): 0
- Norm range: [0.0039, 1.0004]

## H1: prev_desired_curv latch

Mean |desired_curv| at lag k for low-norm vs high-norm frames at time t.
H1 predicts: low-norm frames should show lower |curv| at some lag > 0.

| lag | low-norm mean|curv| | high-norm mean|curv| | ratio (low/high) |
|-----|---------------------|----------------------|-----------------|
| 0 | 0.197828 | 0.122577 | 1.6139 |
| 1 | 0.195615 | 0.135290 | 1.4459 |
| 2 | 0.194522 | 0.141084 | 1.3788 |
| 3 | 0.193383 | 0.147482 | 1.3112 |
| 4 | 0.191383 | 0.160306 | 1.1939 |
| 5 | 0.190577 | 0.165127 | 1.1541 |

**H1 verdict:** **NOT SUPPORTED**: |desired_curv| is similar for both regimes (min ratio=1.154 at lag=5). Curvature is not the trigger.

## H2: Shared CARLA basin

| metric | value |
|--------|-------|
| low-norm centroid norm | 0.0700 |
| high-norm centroid norm | 0.9694 |
| CARLA mean norm | 0.0184 |
| cosine(low-norm centroid, CARLA mean) | 0.3090 |
| cosine(high-norm centroid, CARLA mean) | 0.3909 |
| cosine(low-norm centroid, high-norm centroid) | 0.3796 |
| k=2 label purity (agreement with norm labels) | 0.925 |

**H2 verdict:** **DIFFERENT BASIN**: cosine(low-norm, CARLA) = 0.3090. The daytime attractor is geometrically distinct from the CARLA collapse. H2 not supported; a different mechanism drives the daytime collapse.

## Conclusion

See `report/figures/attractor_norm_trajectory.png` (norm per frame) and
`report/figures/attractor_cluster.png` (PCA of hidden states).
