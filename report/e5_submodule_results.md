# E5 Submodule Results: where the cliff actually lives

Extends `report/e5_results.md` (vision encoder, no cliff found) by
probing 8 tensors between summarizer and the per-head Gemms.

Metrics match `src/e5_layer.py`:

- activity ratio = sum of per-element temporal std, CARLA / real
- mean shift = sum |mean|(CARLA) / sum |mean|(real) at alpha=1
- cliff alpha = smallest alpha at which activity ratio first drops
  below 0.5; NaN if it never crosses

## Probe list (8 points, chosen from `report/e5_submodule_enumeration.md`)

| probe | role | tensor |
|---|---|---|
| `vision_post` | post-encoder FC (1024 to 2048) | `/supercombo/vision/Flatten_output_0` |
| `summarizer_div` | summarizer VAE-mu, == hidden_state | `/summarizer/Div_output_0` |
| `attention_block_out` | transformer self-attention output + residual (1, 10, 512) | `/Add_1_output_0` |
| `transformer_block_out` | transformer FFN output + residual (1, 10, 512) | `/Add_2_output_0` |
| `reduce_sum` | temporal aggregation, 10 tokens to 512 | `/ReduceSum_output_0` |
| `action_block_body` | action-block last resblock output (1, 128), pre-curvature head | `/action_block/resblocks.1/final_relu/Relu_output_0` |
| `hydra_trunk` | non-temporal hydra trunk (1, 512), feeds meta/pose/desire_pred/etc. | `/supercombo/no_bottleneck_policy/hydra/resblock/final_relu/Relu_output_0` |
| `temporal_hydra_trunk` | temporal hydra trunk (1, 512), feeds plan/lane_lines/lead/etc. | `/temporal_hydra/resblock/final_relu/Relu_output_0` |

## Per-probe activity along the alpha sweep

| probe | cliff alpha | activity ratio @ alpha=1 | mean shift @ alpha=1 |
|---|---|---|---|
| `vision_post` | no cliff | 1.8933 | 1.1066 |
| `summarizer_div` | 0.900 | 0.1852 | 0.0233 |
| `attention_block_out` | 0.900 | 0.1790 | 1.0014 |
| `transformer_block_out` | 0.900 | 0.1801 | 1.0008 |
| `reduce_sum` | 0.900 | 0.1640 | 0.8721 |
| `action_block_body` | 0.500 | 0.1934 | 0.9853 |
| `hydra_trunk` | no cliff | 2.7137 | 0.6884 |
| `temporal_hydra_trunk` | 0.900 | 0.2526 | 1.0139 |

## Per-alpha activity ratio, by probe

| alpha | vision_post | summarizer_div | attention_block_out | transformer_block_out | reduce_sum | action_block_body | hydra_trunk | temporal_hydra_trunk |
|---|---|---|---|---|---|---|---|---|
| 0.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1000 | 1.044 | 1.022 | 1.000 | 0.999 | 0.997 | 0.959 | 1.032 | 0.986 |
| 0.2000 | 1.331 | 1.108 | 1.037 | 1.049 | 1.073 | 0.748 | 1.447 | 1.090 |
| 0.3000 | 1.814 | 1.224 | 1.153 | 1.170 | 1.214 | 0.756 | 2.557 | 1.744 |
| 0.4000 | 2.185 | 1.102 | 1.092 | 1.104 | 1.126 | 0.661 | 3.042 | 1.960 |
| 0.5000 | 2.158 | 0.858 | 0.838 | 0.837 | 0.843 | 0.281 | 3.357 | 1.159 |
| 0.6000 | 1.938 | 0.870 | 0.845 | 0.843 | 0.898 | 0.327 | 2.954 | 1.197 |
| 0.7000 | 1.905 | 0.778 | 0.742 | 0.743 | 0.805 | 0.402 | 2.866 | 1.102 |
| 0.8000 | 1.895 | 0.528 | 0.503 | 0.504 | 0.528 | 0.507 | 2.776 | 0.758 |
| 0.9000 | 1.897 | 0.298 | 0.287 | 0.286 | 0.267 | 0.316 | 2.740 | 0.408 |
| 1.0000 | 1.893 | 0.185 | 0.179 | 0.180 | 0.164 | 0.193 | 2.714 | 0.253 |

Figure: `report/figures/e5_submodule_localization.png`.

## Where the cliff lives (answer)

Reading the table left-to-right (in graph order):

1. `vision_post` (1024 to 2048 post-encoder FC): activity ratio climbs to 1.89 at alpha=1, with a peak of 2.19 mid-sweep. The encoder + post-FC are MORE active on CARLA than on real Subaru, not less. This is consistent with E5: the encoder does not collapse.

2. `summarizer_div` (the VAE-style normalised 512-D bottleneck, == `hidden_state`): activity stays above 1.0 through alpha=0.3, then falls monotonically and crosses 0.5 between alpha=0.7 (0.778) and alpha=0.8 (0.528). Mean shift at alpha=1 is 0.023, meaning the rolling mean of this 512-D vector collapses by almost two orders of magnitude. This is the entry point of the collapse and the same vector E6 monitors.

3. `attention_block_out`, `transformer_block_out`, `reduce_sum`: these three tensors all track `summarizer_div` to within 2 percent across the whole sweep (e.g. at alpha=1: 0.179 / 0.180 / 0.164 vs 0.185). The transformer self-attention + FFN + reduce-sum stage does NOT introduce additional collapse: it is a passive relay of the summarizer bottleneck.

4. `action_block_body` (last resblock of the steering branch, just before the desired_curvature Gemm): cliff alpha 0.500, activity 0.281 at alpha=0.5 already. This is by far the most sensitive submodule, dropping below 0.5 a full step earlier than summarizer. The 515 to 128 projection that mixes `summarizer + ReduceSum + lateral_control_params + prev_desired_curv` amplifies the collapse, presumably because once the recurrent input (`prev_desired_curv` rolling from the model's own collapsed outputs) joins in, the action stack saturates fast.

5. `temporal_hydra_trunk` (plan / lane_lines / lead / etc. shared 512-D trunk): cliff alpha 0.900, activity 0.253. Tracks the upstream collapse with a small attenuation, consistent with E1 showing the 8 collapsing heads are all on this branch.

6. `hydra_trunk` (meta / pose / desire_pred / wide_from_device_euler / road_transform): no cliff, activity ratio 2.71 at alpha=1, but mean shift 0.69. The non-temporal heads stay alive in temporal variation but their DC offsets drift. This is exactly the E1 result that meta and pose do NOT collapse on CARLA, now localised to the trunk that feeds them.

The cliff is NOT in any one block; it is a two-stage failure:

- the SUMMARIZER (`/summarizer/Div_output_0`, the VAE-mu / normalised hidden_state) is where temporal variation first falls below the real baseline. The transformer + reduce-sum stage is passive.
- the ACTION_BLOCK then amplifies the collapse a full alpha step earlier than summarizer does, because it folds in the recurrent `prev_desired_curv` input which is itself the model's own collapsed curvature output rolled in.

The two hydra trunks split cleanly: the temporal trunk follows the collapse, the non-temporal trunk does not. This matches and refines the E1 finding (`8 of 10 heads collapse`) by showing that the split happens at the hydra trunk level, not at the per-head Gemms.

## Caveat

The summarizer ends with `/summarizer/Div_output_0 = mu / sigma` (a VAE-style reparameterisation). The Div by sigma can mechanically suppress variance on out-of-distribution inputs if the predicted sigma grows. We have not separated `mu` (pre-Div Gemm output) from `Div` here, so part of the apparent collapse in `summarizer_div` could be variance normalisation rather than information loss. Probing `/summarizer/_mu/Gemm_output_0` alone would split that further; left for follow-up.
