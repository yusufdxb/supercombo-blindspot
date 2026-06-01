# E5 Submodule Results: where the cliff actually lives

Extends `report/e5_results.md` (vision encoder, no cliff found) by
probing 8 tensors between summarizer and the per-head Gemms.

Metrics match `src/e5_layer.py`:

- activity ratio = sum of per-element temporal std, CARLA / real
- mean shift = sum |mean|(CARLA) / sum |mean|(real) at alpha=1
- cliff alpha = smallest alpha at which activity ratio first drops
  below 0.5; N/A if it never crosses

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
