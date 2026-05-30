# E5 Submodule Enumeration: tensors between vision-encoder head and outputs

Source: `models/supercombo.onnx` (openpilot v0.9.7, fp16 ONNX export).

Entry point (already-probed in E5): `/supercombo/vision/_en/head/global_pool/flatten/Flatten_output_0` (shape `[1, 1024]`).

Forward DFS from that tensor reaches 234 named tensors. No literal `GRU` / `LSTM` / `RNN` op appears in this region of the graph: temporal context is threaded by the host (see `src/state.py`) via the `features_buffer` input (`[1, 99, 512]`), and a transformer block over 10 tokens (1 current + 8 from buffer + 1 other) provides the in-graph aggregation. The 82 `Gemm` + 67 `Relu` ops are all feed-forward (resblocks in summarizer / action_block / hydras + linear heads).

## Structural roles (op counts downstream of the encoder head, in graph order)

| role | what it is | representative tensors |
|---|---|---|
| vision_post | post-encoder FC: 1024 to 2048 | `/supercombo/vision/_en/head/fc/Gemm_output_0` `[1, 2048]`, `/supercombo/vision/Flatten_output_0` `[1, 2048]` |
| summarizer | VAE-style residual block + mu/std normalisation, 2048 to 512 | `/summarizer/resblock/final_relu/Relu_output_0` `[1, 512]`, `/summarizer/_mu/Gemm_output_0` `[1, 512]`, `/summarizer/Div_output_0` `[1, 512]` (this IS `hidden_state`) |
| feature_buffer_concat | builds the 10-token sequence (1 current + 8 history + 1 extra) | `/Unsqueeze_output_0` `[1, 1, 512]`, `/Gather_output_0` `[1, 8, 512]`, `/Concat_1_output_0` `[1, 10, 512]` |
| temporal_encode | linear embed of the 10-token sequence | `/_encode/_encode.0/Add_output_0` `[1, 10, 512]`, `/_encode/_encode.1/Relu_output_0` `[1, 10, 512]` |
| temporal_attention | 8-head self-attention over the 10 tokens | `/_attention/c_attn/Add_output_0` `[1, 10, 1536]`, `/_attention/c_proj/Add_output_0` `[1, 10, 512]`, `/Add_1_output_0` `[1, 10, 512]` (post-residual) |
| temporal_mlp | transformer FFN (GELU-flavour via Elu) | `/_mlp/c_fc/Add_output_0` `[1, 10, 2048]`, `/_mlp/c_proj/Add_output_0` `[1, 10, 512]`, `/Add_2_output_0` `[1, 10, 512]` (post-FFN residual = transformer block output) |
| temporal_reduce | weighted sum across the 10 tokens (replaces GRU) | `/ReduceSum_output_0` `[1, 512]` |
| action_block_in | concat with `lateral_control_params` + gathered `prev_desired_curv`, then 515 to 128 | `/Concat_2_output_0` `[1, 515]`, `/action_block/action_block_in/action_block_in.1/Relu_output_0` `[1, 128]` |
| action_block_body | 2 resblocks at 128 channels | `/action_block/resblocks.0/final_relu/Relu_output_0` `[1, 128]`, `/action_block/resblocks.1/final_relu/Relu_output_0` `[1, 128]` |
| action_block_out | desired_curvature head, 128 to 2 with sigmoid-like Clip/Mul | `/action_block/action_block_out/Gemm_output_0` `[1, 2]`, `/action_block/Mul_output_0` `[1, 2]` |
| hydra_trunk | shared 512-channel resblock feeding the non-temporal hydra heads (meta, pose, desire_pred, wide_from_device_euler, road_transform) | `/supercombo/no_bottleneck_policy/hydra/resblock/final_relu/Relu_output_0` `[1, 512]` |
| hydra_heads | 5 small MLP heads off hydra_trunk to their output slots | `/supercombo/no_bottleneck_policy/hydra/meta_1/Gemm_output_0` `[1, 48]`, `.../pose_1/Gemm_output_0` `[1, 12]`, etc. |
| temporal_hydra_trunk | shared 512-channel resblock feeding the temporal hydra heads (plan, lane_lines, lane_lines_prob, road_edges, lead, lead_prob, desire_state, sim_pose) | `/temporal_hydra/resblock/final_relu/Relu_output_0` `[1, 512]` |
| temporal_hydra_heads | 8 small MLP heads off temporal_hydra_trunk | `/temporal_hydra/plan_1/Gemm_output_0` `[1, 4955]`, `/temporal_hydra/lane_lines_1/Gemm_output_0` `[1, 528]`, etc. |
| output_slicing | final concat of all 15 head outputs | `outputs` `[1, 6504]` |

## Per-tensor probe-candidacy table (key tensors only)

| tensor | shape | upstream op | downstream op | role | probe? | justification |
|---|---|---|---|---|---|---|
| `/supercombo/vision/_en/head/global_pool/flatten/Flatten_output_0` | `[1, 1024]` | GlobalAveragePool | Gemm | vision_post (entry) | no | already in E5 as `head` |
| `/supercombo/vision/Flatten_output_0` | `[1, 2048]` | Gemm | Gemm | vision_post (output) | YES | extends E5 into the dense post-encoder FC, cheap, isolates whether the 1024 to 2048 expansion shifts |
| `/summarizer/resblock/final_relu/Relu_output_0` | `[1, 512]` | Add | Gemm | summarizer (resblock out) | maybe | redundant with summarizer/Div; pick Div instead since it is what E6 monitors |
| `/summarizer/Div_output_0` | `[1, 512]` | Div | Unsqueeze, ReduceSum | summarizer (VAE mu/normalised, == hidden_state) | YES | this IS the E6 monitor target and `hidden_state` slot; pivot of the story |
| `/Concat_1_output_0` | `[1, 10, 512]` | Concat | MatMul | feature_buffer_concat | no | mostly a cat; activity dominated by summarizer/Div + history. Probe at temporal_encode output instead. |
| `/_encode/_encode.1/Relu_output_0` | `[1, 10, 512]` | Relu | Add | temporal_encode | no | one Gemm + Relu past Concat_1, not load-bearing as a localisation point |
| `/Add_1_output_0` | `[1, 10, 512]` | Add | Add | temporal_attention (post-residual) | YES | output of the self-attention block over 10 tokens; first place where temporal mixing happens |
| `/Add_2_output_0` | `[1, 10, 512]` | Add | ReduceSum | temporal_mlp (transformer block out) | YES | output of the transformer FFN residual; the last 10-token tensor before reduction |
| `/ReduceSum_output_0` | `[1, 512]` | ReduceSum | Concat, Gemm | temporal_reduce | YES | the single 512-D temporal-aggregated vector that drives BOTH the hydra trunks; if collapse starts here it implicates the transformer + reduction step, not the heads |
| `/action_block/action_block_in/action_block_in.1/Relu_output_0` | `[1, 128]` | Relu | Add, Gemm | action_block_in | maybe | redundant with action_block resblocks.1 output; pick the last |
| `/action_block/resblocks.1/final_relu/Relu_output_0` | `[1, 128]` | Add | Gemm | action_block_body (out) | YES | the final action-block representation before the curvature head; localises desired_curvature collapse cleanly |
| `/supercombo/no_bottleneck_policy/hydra/resblock/final_relu/Relu_output_0` | `[1, 512]` | Add | Gemm | hydra_trunk | YES | the shared trunk for non-temporal heads; if this collapses but temporal_hydra_trunk does not (or vice versa) we have a smoking gun |
| `/temporal_hydra/resblock/final_relu/Relu_output_0` | `[1, 512]` | Add | Gemm | temporal_hydra_trunk | YES | the shared trunk for temporal heads (incl. plan); E1/E4 showed plan collapses, this is the closest pre-head tensor |
| `/temporal_hydra/plan/Gemm_output_0` | `[1, 256]` | Gemm | Relu | temporal_hydra_heads (plan stem) | no | inside the plan head, one step before plan_1 output; redundant with temporal_hydra_trunk for localisation purposes |
| `outputs` | `[1, 6504]` | Concat | (graph out) | output_slicing | no | this is the existing E4 signal, no new info |

The remaining ~210 tensors are intermediate Gemm/Relu inside resblocks or per-head MLPs; probing them is redundant given a probe at the trunk output of each block.
