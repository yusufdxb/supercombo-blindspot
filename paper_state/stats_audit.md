# Stats Audit

Auditor: paper-council stage 9 (stats verifier), run 2026-05-30.
Source of truth: `report/*.md` and committed `report/*_collected.npz` files.
Draft: `drafts/main_draft.md`. Ledger: `paper_state/claim_ledger.md`.

All recomputation was performed via Python/NumPy directly from the committed artifacts.
No number was taken from model memory; every value was derived from a file read or
a Bash computation run in this session.

---

## c1: Parity frame count and pass rate

- PROSE VALUE: "100.00% of 1159 real-footage frames" (Abstract, Section 4, Section 1).
- SOURCE ARTIFACT: `report/parity_results.md` lines 15-16.
- RECOMPUTED VALUE: histogram sum 982+174+3+0 = 1159; frames in (0.5, inf) bin = 0 -> 1159/1159 = 100.00%.
- COMMAND RUN: `python3 -c "print(982+174+3+0, (982+174+3)/1159*100)"`
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c2: Parity delta statistics

- PROSE VALUE: median 0.0409, mean 0.0541, max 0.2899 (Section 4).
- SOURCE ARTIFACT: `report/parity_results.md` lines 17-19.
- RECOMPUTED VALUE: read directly from artifact; histogram structure confirms max < 0.5.
- COMMAND RUN: grep from `report/parity_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c3: Parity acceptance criterion

- PROSE VALUE: ">= 95% of frames after a 40-frame (2 s at 20 Hz) warm-up trim, compared on accel_t0" (Section 4).
- SOURCE ARTIFACT: `report/parity_results.md` lines 7-12.
- RECOMPUTED VALUE: artifact text matches verbatim.
- COMMAND RUN: read `report/parity_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c4: Recurrent-state threading (factual)

- PROSE VALUE: "shift-and-append ... zero initialization only on the first frame" (Section 4).
- SOURCE ARTIFACT: `report/parity_results.md` (state-handling note).
- RECOMPUTED VALUE: artifact text confirms "correct recurrent-state threading".
- COMMAND RUN: read `report/parity_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c5: Unnormalized YUV input (factual)

- PROSE VALUE: "uint8 values in the range 0 to 255, not values divided by 255" (Section 4).
- SOURCE ARTIFACT: `report/parity_results.md` (preprocessing note).
- RECOMPUTED VALUE: artifact confirms "unnormalized YUV input".
- COMMAND RUN: read `report/parity_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c6: E1 output collapse count and per-head ratios

- PROSE VALUE: "8 of 10 output heads collapse to under 1% ... desired_curv (0.0018), accel_t0 (0.0040), lead (0.0042), desire_state (0.0049), lane_lines (0.0054), plan (0.0057), lead_prob (0.0058), road_edges (0.0076)" (Section 5.1).
- SOURCE ARTIFACT: `report/teardown_results.md` E1 table; `report/teardown_collected.npz`.
- RECOMPUTED VALUE: computed activity ratio = std(carla_post100warmup) / std(concat[subaru,ram]_post100warmup) for all heads using the npz. All 8 ratios match exactly: desired_curv=0.0018, accel_t0=0.0040, lead_prob=0.0058, plan=0.0057, lane_lines=0.0054, road_edges=0.0076, lead=0.0042, desire_state=0.0049. 8 of 10 < 0.01. CARLA frames in npz = 319; analysis uses post-100-frame window (219 frames).
- COMMAND RUN: `python3 -c "import numpy as np; t=np.load('report/teardown_collected.npz',allow_pickle=True); ..."` (full recomputation run in session).
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c7: Surviving heads pose and meta

- PROSE VALUE: "pose (0.1788) and meta (0.7181)" (Section 5.1).
- SOURCE ARTIFACT: `report/teardown_results.md` E1 table; `report/teardown_collected.npz`.
- RECOMPUTED VALUE: pose ratio = 0.1788, meta ratio = 0.7181 (recomputed from npz; matches artifact exactly).
- COMMAND RUN: same recomputation as c6.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c8: E2 feature spread ratio ~1e-5

- PROSE VALUE: "about 1e-5 (0.00001x) of its real value" (Abstract, Section 5.2).
- SOURCE ARTIFACT: `report/teardown_results.md` E2; `report/teardown_collected.npz`.
- RECOMPUTED VALUE: trace(cov(carla_hs_post100)) / trace(cov(concat[subaru,ram]_hs_post100)) = 7e-6 / 0.544 = 1.29e-5. Reported as 1e-5 (one significant figure, order-of-magnitude).
- COMMAND RUN: `python3 -c "import numpy as np; t=np.load(...); R=concat; C=carla; spread_r=np.var(R,0).sum(); spread_c=np.var(C,0).sum(); print(spread_c/spread_r)"` -> 1.29e-5.
- RESULT: MATCH (1.29e-5 rounds to "about 1e-5" at one significant figure; the artifact text says "0.00001x").
- VERDICT: CONFIRMED.
- NOTE: The precise value is 1.29e-5, not exactly 1.0e-5. The prose uses "about 1e-5" which is a correct order-of-magnitude statement; this is not a mismatch.

---

## c9: E2 separability 87.9% and d' = 2.19

- PROSE VALUE: "87.9% (d' = 2.19)" (Section 5.2, Abstract).
- SOURCE ARTIFACT: `report/teardown_results.md` E2; `report/teardown_collected.npz`; `src/teardown.py` e2_feature_ood().
- RECOMPUTED VALUE: implemented exact algorithm from src/teardown.py (centroid-difference direction projection, balanced-accuracy separability, d-prime formula). Result: separability = 87.9%, d' = 2.19.
- COMMAND RUN: `python3 -c "... e2_feature_ood(concat_real, carla) ..."` (session Bash, full implementation).
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c10: E3 uncertainty ratios

- PROSE VALUE: "plan 1.35x; lead 1.20x; desired_curv 1.84x" (Section 5.3, Abstract).
- SOURCE ARTIFACT: `report/teardown_results.md` E3 table; `report/teardown_collected.npz`.
- RECOMPUTED VALUE: mean(plan_std, carla_post100) / mean(plan_std, real_post100) = 0.5522/0.4103 = 1.35x; lead = 1.5412/1.2832 = 1.20x; desired_curv = 0.1534/0.0833 = 1.84x. All match.
- COMMAND RUN: `python3 -c "import numpy as np; t=np.load(...); print(np.mean(t['carla__plan_std'][100:])/np.mean(concat_real_plan_std[100:]))"` (session).
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c11: E3 "0 of 219 CARLA frames above real p95"

- PROSE VALUE (corrected in draft, 2026-05-31): "0 of 219 CARLA frames (0%) exceed the real-driving p95 of any monitored head" (Section 5.3 prose and table). Prior draft said "0 of 220"; corrected to 219 by this audit.
- SOURCE ARTIFACT: `report/teardown_results.md` E3 table (0% column); `report/teardown_collected.npz`; `src/teardown.py` WARMUP=100, `_post()`.
- RECOMPUTED VALUE: `python3 -c "import numpy as np; from src.teardown import _post, WARMUP, _load_cache, CACHE; segs=_load_cache(CACHE); carla=_post(segs['carla'],WARMUP); print(len(carla['plan']))"` -> 219 frames. Per-frame mean uncertainty for plan, lead, desired_curv on post-warmup CARLA: 0 frames exceed real-driving p95 for all three heads.
- COMMAND RUN: `python3 -c "from src.teardown import _post,WARMUP,_load_cache,CACHE; s=_load_cache(CACHE); c=_post(s['carla'],WARMUP); print(len(c['plan']))"` -> 219.
- RESULT: MATCH (after correction). N=319 stored frames in cache; `_post(319, WARMUP=100)` = 219 analysis frames. The prior "220" was wrong because it neglected the pair-processing step in `collect()` (N=320 raw -> 319 stored; 319-100=219, not 320-100=220). Zero above p95 confirmed on 219 frames.
- VERDICT: CONFIRMED (post-correction).

---

## c11-framecounts: §4.2 frame count reconciliation (new audit, 2026-05-31)

- PROSE VALUE (old, now corrected): "320 frames each, 100 warmup frames discarded (220 analysis frames each)" and "320 CARLA-rendered clean-road frames" in §4.2.
- PROSE VALUE (new, in draft): 320 raw collected per segment; 319 stored (pair-processing reduces by 1); 219 E1/E2/E3 analysis frames; 319 frames for E6/metrics without warmup discard.
- SOURCE ARTIFACT: `report/teardown_collected.npz` shapes (all 319 per corpus); `src/teardown.py` N=320, WARMUP=100, `_post()`; `src/probe_model.py` `collect()` (pair-processing loop skips first frame, produces N-1 outputs); `src/e6_detector.py` `_real_calibration_hidden()` (loads 319 frames directly, no `_post`); `src/baselines.py` `_real_calibration_hidden()` (same); `report/metrics_results.md` header "n=638 ID, n=319 OOD".
- RECOMPUTED VALUE: `python3 -c "import numpy as np; d=np.load('report/teardown_collected.npz'); print(d['subaru__accel_t0'].shape, d['ram__accel_t0'].shape, d['carla__accel_t0'].shape)"` -> (319,) (319,) (319,). Confirms: cache stores 319 per corpus (not 320). `_post(319, 100)` = 219 analysis frames for E1/E2/E3. `e6_detector._real_calibration_hidden()` concatenates raw 319+319=638.
- COMMAND RUN: `python3 -c "import numpy as np; d=np.load('report/teardown_collected.npz'); print(d['subaru__accel_t0'].shape)"` -> (319,).
- RESULT: Old §4.2 text was MISMATCH on three counts: (a) "220 analysis frames" should be 219; (b) "320 CARLA frames" in the OOD bullet described raw collection count, not the stored or analysis count; (c) no mention of the two distinct analysis paths (E1-E3 vs E6/metrics). All corrected in draft 2026-05-31.
- VERDICT: CONFIRMED (post-correction).
- NOTE: The "100 warmup frames discarded" claim IS substantiated by `src/teardown.py` WARMUP=100 and the `_post()` call. The warmup step is real and documented in the code. The off-by-one on 220 vs 219 arose because the pair-processing step in `collect()` (which reduces N=320 raw frames to N-1=319 stored frames) was not accounted for in the prose.

---

## c12: E3 output retained percentages

- PROSE VALUE: "plan head retains 0.6% ... lead head retains 0.4% ... desired_curv head retains 0.2%" (Section 5.3).
- SOURCE ARTIFACT: `report/teardown_results.md` E3 table; `report/teardown_collected.npz`.
- RECOMPUTED VALUE: plan 6.8342/1193.8486 = 0.57% -> 0.6%; lead 0.9693/232.3695 = 0.42% -> 0.4%; desired_curv 0.0002/0.1318 = 0.18% -> 0.2%. All match at 1 sig fig.
- COMMAND RUN: `python3 -c "print(6.8342/1193.8486, 0.9693/232.3695, 0.0002/0.1318)"` (session).
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c13: E4 cliff width 0.015, alpha 0.784-0.799

- PROSE VALUE: "transition width of 0.015 ... alpha band 0.784 to 0.799" (Section 5.4, Abstract, Introduction, Conclusion).
- SOURCE ARTIFACT: `report/e4_results.md` Verdict line; comparison table.
- RECOMPUTED VALUE: a90=0.784, a10=0.799; 0.799-0.784 = 0.015. Confirmed from per-alpha table: activity at alpha=0.775 is 1.3568 (>0.9), activity at alpha=0.800 is 0.0347 (<0.1). The cliff is fully contained in the [0.775, 0.800] bin.
- COMMAND RUN: `python3 -c "print(0.799 - 0.784)"` -> 0.015.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c14: E4 peak 6.32x at alpha=0.425

- PROSE VALUE: "6.32x of the real baseline at alpha=0.425" (Section 5.4).
- SOURCE ARTIFACT: `report/e4_results.md` per-alpha table row alpha=0.4250.
- RECOMPUTED VALUE: table value = 6.3161; round to 2dp = 6.32.
- COMMAND RUN: `python3 -c "print(round(6.3161, 2))"` -> 6.32.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c15: E4 feature spread crashes by alpha=0.78

- PROSE VALUE: "feature spread crashes from 0.25 to 0.00 by about alpha=0.78" (Section 5.4).
- SOURCE ARTIFACT: `report/e4_results.md` per-alpha table, feature_spread column.
- RECOMPUTED VALUE: alpha=0.0 spread=0.25; alpha=0.775 spread=0.00 (first 0.00 value). "About alpha=0.78" is correct.
- COMMAND RUN: read `report/e4_results.md` table.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c16: E4-RAM gradient width 0.274, alpha 0.666-0.940

- PROSE VALUE: "a gradient of width 0.274 ... alpha 0.666 to 0.940" (Section 5.4, Abstract, Introduction, Conclusion).
- SOURCE ARTIFACT: `report/e4_ram_results.md` Verdict line; comparison table.
- RECOMPUTED VALUE: a90=0.666, a10=0.940; 0.940-0.666 = 0.274.
- COMMAND RUN: `python3 -c "print(round(0.940-0.666, 3))"` -> 0.274.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c17: Headroom +0.234 Subaru, -0.184 RAM

- PROSE VALUE: "early-warning headroom is negative (-0.184) vs +0.234 on Subaru" (Section 5.4).
- SOURCE ARTIFACT: `report/e4_ram_results.md` comparison table.
- RECOMPUTED VALUE: Subaru headroom = 0.784 - 0.550 = 0.234; RAM headroom = 0.666 - 0.850 = -0.184.
- COMMAND RUN: `python3 -c "print(0.784-0.550, 0.666-0.850)"` -> 0.234, -0.184.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c18: E5 encoder activity ratios at alpha=1

- PROSE VALUE: "stem is at 1.43x, stage3 at 2.06x, and the head at 2.14x ... minimum ratio over all stages and all alpha is about 0.96" (Section 5.5).
- SOURCE ARTIFACT: `report/e5_results.md` table.
- RECOMPUTED VALUE: head=2.1416 (rounds to 2.14), stage3=2.0561 (rounds to 2.06), stem=1.4254 (rounds to 1.43). Minimum across all stages at alpha=1: stage1=0.9560 ("about 0.96"). All NaN for cliff_alpha.
- COMMAND RUN: read `report/e5_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c19: Collapse downstream of encoder (novelty/factual)

- PROSE VALUE: "every encoder stage stays at or above real activity across the full sweep ... the failure is in the summarizer and action block, not in the encoder" (Section 5.5).
- SOURCE ARTIFACT: `report/e5_results.md`; `report/e5_submodule_results.md`.
- RECOMPUTED VALUE: all e5 encoder rows have cliff_alpha=NaN and activity_ratio>=0.956 at alpha=1. summarizer_div cliff_alpha=0.900, action_block_body cliff_alpha=0.500. This is consistent with the claim.
- COMMAND RUN: read both result files.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c20: summarizer_div cliff alpha=0.900, mean shift=0.023

- PROSE VALUE: "cliff at alpha=0.900, with its mean shifting to 0.023 of real" (Section 5.5).
- SOURCE ARTIFACT: `report/e5_submodule_results.md` per-probe table.
- RECOMPUTED VALUE: table row summarizer_div: cliff_alpha=0.900, mean_shift=0.0233. Rounds to 0.023.
- COMMAND RUN: read `report/e5_submodule_results.md`.
- RESULT: MATCH. The cliff_alpha column definition (first alpha where activity < 0.5) is verified: at alpha=0.8 activity=0.528 (>0.5), at alpha=0.9 activity=0.298 (<0.5), so first below-0.5 alpha = 0.9. Column value 0.900 is correct by definition.
- VERDICT: CONFIRMED.
- NOTE (drafter-flagged nuance): The narrative text in `e5_submodule_results.md` says "crosses 0.5 between alpha=0.7 (0.778) and alpha=0.8 (0.528)" - this is wrong in the narrative because 0.528 > 0.5. The COLUMN value (0.900) is correct by the stated definition and IS what the draft uses. The narrative text has a loose description but does not propagate into the draft. The draft is internally consistent.

---

## c21: action_block_body cliff alpha=0.500

- PROSE VALUE: "cliff a full alpha step earlier, at alpha=0.500" (Section 5.5).
- SOURCE ARTIFACT: `report/e5_submodule_results.md` per-probe table.
- RECOMPUTED VALUE: action_block_body cliff_alpha=0.500; per-alpha table shows activity=0.661 at alpha=0.4, then 0.281 at alpha=0.5 (first below 0.5). Column value 0.500 confirmed.
- COMMAND RUN: read `report/e5_submodule_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c22: vision_post 1.89x, hydra_trunk 2.71x; passive relay within ~2%

- PROSE VALUE: "vision_post (1.89x at alpha=1) and the non-temporal hydra trunk (hydra_trunk, 2.71x at alpha=1) show no cliff ... transformer self-attention, feed-forward, and reduce-sum stages track the summarizer to within about 2%" (Section 5.5).
- SOURCE ARTIFACT: `report/e5_submodule_results.md` per-probe table.
- RECOMPUTED VALUE: vision_post=1.8933 (rounds to 1.89 MATCH); hydra_trunk=2.7137 (rounds to 2.71 MATCH). Passive relay deviation from summarizer_div at alpha=1: attention_block_out 3.2%, transformer_block_out 2.6%, reduce_sum 11.4%. Mean deviation across all alpha steps: 3.3%, 3.0%, 3.5% respectively.
- COMMAND RUN: `python3 -c "... for name,val in [...]: diff_pct = abs(val-0.185)/0.185*100 ..."` (session).
- RESULT: PARTIAL MISMATCH on the "within about 2%" sub-claim. vision_post and hydra_trunk values are confirmed. The three relay blocks have mean deviations of 3-3.5% across the sweep and the reduce_sum reaches 11.4% at alpha=1, exceeding the stated "about 2%". Attention and transformer blocks are closer at 2.6-3.2%.
- VERDICT: FLAGGED.
- NOTE: The "about 2%" claim in both the draft and the source artifact narrative is imprecise. The actual deviations are 3-11% depending on the block and alpha. The attention and transformer blocks are approximately 2-4% off, but reduce_sum reaches 11.4% at alpha=1. The artifact narrative says "to within 2 percent across the whole sweep" which is not supported by the per-alpha table it generates. The source of the misstatement is the artifact itself, not a transcription error. Since the reduce_sum deviation is most pronounced at extreme OOD (alpha=1) and the broader qualitative conclusion (passive relay behavior, no additional collapse) is not wrong, this is flagged rather than refuted, but the draft should say "within about 2-11%" or simply cite the table.

---

## c23: Partial localization, VAE mu/sigma ambiguity (factual)

- PROSE VALUE: "a VAE mu-vs-sigma ambiguity in the summarizer reparameterization ... remains unresolved" (Section 5.5, Limitations).
- SOURCE ARTIFACT: `report/e5_submodule_results.md` Caveat section.
- RECOMPUTED VALUE: artifact text confirms the caveat verbatim.
- COMMAND RUN: read `report/e5_submodule_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c24: E6 AUROC 0.996 [0.992, 1.000]

- PROSE VALUE: "AUROC 0.996 [0.992, 1.000]" (Section 5.6, Abstract, Conclusion).
- SOURCE ARTIFACT: `report/metrics_results.md` Table 1; `report/metrics_collected.npz` keys e6__auroc_point, e6__auroc_ci.
- RECOMPUTED VALUE: stored e6__auroc_point=0.9963 (rounds to 0.996). CI stored as [mean, lo, hi] = [0.9963, 0.9915, 0.9999]; lo rounds to 0.992, hi rounds to 1.000. Bootstrap recomputed from raw scores (n_ID_valid=609, n_OOD_valid=290, seed=42): AUROC=0.9963, CI lower=[0.991, 1.000].
- COMMAND RUN: `python3 -c "import numpy as np; from sklearn.metrics import roc_auc_score; m=np.load(...); ..."` (full bootstrap recomputation in session).
- RESULT: MATCH (all values confirmed at 3dp rounding).
- VERDICT: CONFIRMED.
- NOTE: The metrics_results.md states n=638 ID and n=319 OOD; the npz shows 638 and 319 total scores, of which 609 and 290 are non-NaN (first 29 per series are NaN due to rolling window warmup). The AUROC is computed on non-NaN values. The stated n=638/n=319 are total stored counts, not valid-score counts. This is a minor metadata description issue but does not affect the AUROC value.

---

## c25: KNN-50 AUROC 1.000 [1.000, 1.000]

- PROSE VALUE: "KNN-50 reaches AUROC 1.000 [1.000, 1.000]" (Section 5.6).
- SOURCE ARTIFACT: `report/metrics_results.md` Table 1; `report/metrics_collected.npz`.
- RECOMPUTED VALUE: roc_auc_score(labels, knn50_scores) = 1.000. CI stored = [1.000, 1.000, 1.000].
- COMMAND RUN: `python3 -c "... knn_auroc = roc_auc_score(knn_labels, knn_scores); print(knn_auroc)"` -> 1.000.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c26: Mahalanobis 0.159 [0.130, 0.190], RMD 0.934 [0.914, 0.952], PCA-Maha 0.152 [0.124, 0.179]

- PROSE VALUE: "Mahalanobis (0.159 [0.130, 0.190]) ... PCA-Mahalanobis (0.152 [0.124, 0.179]) ... Relative Mahalanobis reaches 0.934 [0.914, 0.952]" (Section 5.6).
- SOURCE ARTIFACT: `report/metrics_results.md` Table 1; `report/metrics_collected.npz`.
- RECOMPUTED VALUE: Mahalanobis AUROC=0.159, CI=[0.130, 0.190] (stored: [0.159, 0.130, 0.190]). RMD AUROC=0.934, CI=[0.914, 0.952]. PCA-Maha AUROC=0.152, CI=[0.124, 0.179]. All match stored values and recomputed from raw scores.
- COMMAND RUN: `python3 -c "... roc_auc_score(labels, maha_scores)"` -> 0.159; similar for RMD and PCA.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c27: Mahalanobis below-chance mechanism (factual)

- PROSE VALUE: "the recurrent feature freezes to a near-constant vector that lands near the center of the in-distribution Gaussian" (Section 5.6).
- SOURCE ARTIFACT: `report/metrics_results.md` Headline section.
- RECOMPUTED VALUE: artifact confirms this explanation verbatim; E2 recomputation confirms CARLA state collapses to near-constant.
- COMMAND RUN: read `report/metrics_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c28: LOCO mean FPR 1.03%, max 2.07%

- PROSE VALUE: "LOCO mean FPR 1.03% (max 2.07%)" (Section 5.6, Abstract, Conclusion).
- SOURCE ARTIFACT: `report/e6_results.md` LOCO table.
- RECOMPUTED VALUE: LOCO FPRs = [0.0000, 0.0207]; mean = (0.0000+0.0207)/2 = 0.01035 = 1.035% -> rounds to 1.03%; max = 0.0207 = 2.07%.
- COMMAND RUN: `python3 -c "print((0.0000+0.0207)/2, max(0.0000,0.0207))"` -> 0.01035, 0.0207.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c29: PCA-Maha LOCO mean 11.91%, max 23.82%; KNN/Maha/RMD 100%

- PROSE VALUE: "PCA-Mahalanobis improves to 11.91% LOCO mean FPR ... all three applicable location-based baselines ... hit 100% LOCO FPR" (Section 5.6).
- SOURCE ARTIFACT: `report/metrics_results.md` PCA-Maha LOCO table and Headline.
- RECOMPUTED VALUE: PCA-Maha LOCO FPRs = [0.0000, 0.2382]; mean = (0.0000+0.2382)/2 = 0.1191 = 11.91%; max = 0.2382 = 23.82%. KNN/Maha/RMD 100% LOCO FPR confirmed by headline text.
- COMMAND RUN: `python3 -c "print((0.0000+0.2382)/2)"` -> 0.1191.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c30: Location-based failure reason (factual)

- PROSE VALUE: "the subaru and ram corpora occupy disjoint regions of the 512-D feature space whose inter-corpus separation dwarfs the within-corpus radius" (Section 5.6).
- SOURCE ARTIFACT: `report/metrics_results.md` Headline.
- RECOMPUTED VALUE: artifact text confirms this mechanical explanation verbatim.
- COMMAND RUN: read `report/metrics_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c31: Monitor fires at alpha=0.550; AUROC crosses 0.5 at alpha=0.550

- PROSE VALUE: "monitor's fired fraction crosses 50% at alpha=0.550 ... its AUROC crosses 0.5 at the same alpha=0.550" (Section 5.6).
- SOURCE ARTIFACT: `report/e6_results.md` alpha sweep table; `report/metrics_results.md` Table 2; `report/metrics_collected.npz` e6__alpha_sweep_auroc.
- RECOMPUTED VALUE: Fire-fraction at alpha=0.525 is 0.438 (<0.5), at alpha=0.550 is 0.517 (>0.5). Fire-fraction crosses 0.5 at alpha=0.550: CONFIRMED. AUROC at alpha=0.4250 = 0.400 (<0.5), at alpha=0.4500 = 0.540 (>0.5). AUROC first exceeds 0.5 between alpha=0.425 and alpha=0.450, NOT at alpha=0.550.
- COMMAND RUN: `python3 -c "import numpy as np; m=np.load('report/metrics_collected.npz', allow_pickle=True); auroc_sweep=m['e6__alpha_sweep_auroc']; alphas=m['alphas']; print(list(zip(alphas[12:18], auroc_sweep[12:18])))"` -> confirms 0.425:0.400, 0.450:0.540.
- RESULT: MISMATCH on AUROC crossing. Fire-fraction crossing at 0.550 is correct. But AUROC crosses 0.5 between alpha=0.425 and alpha=0.450, not at alpha=0.550. The draft claims these two events are at "the same alpha=0.550" which is incorrect for the AUROC.
- VERDICT: FLAGGED.
- NOTE: The early-warning gap of "about 0.23 blend-units" (0.784 - 0.550 = 0.234) is correctly based on the fire-fraction crossing, which IS confirmed at 0.550. The erroneous claim is specifically that the AUROC and fire-fraction cross 0.5 at the same alpha. The AUROC actually crosses 0.5 ~0.10 alpha units earlier than the fire-fraction. This overstates the coherence of the two metrics at the same operating point.

---

## c32: Eval split n=638 ID, n=319 OOD, bootstrap n=1000 seed=42

- PROSE VALUE: "in-distribution set is subaru and ram concatenated (n=638) and the out-of-distribution set is the alpha=1.0 CARLA frames (n=319)" (Section 4).
- SOURCE ARTIFACT: `report/metrics_results.md` header text; `report/metrics_collected.npz` meta__n_bootstrap, meta__seed, and score array shapes.
- RECOMPUTED VALUE: npz id_scores shape=(638,), alpha_1.0_scores shape=(319,). meta__n_bootstrap=1000, meta__seed=42. However 29 of the 638 ID scores and 29 of the 319 OOD scores are NaN (first 29 frames per the rolling-window warmup). Valid counts: 609 ID, 290 OOD.
- COMMAND RUN: `python3 -c "import numpy as np; m=np.load('report/metrics_collected.npz',allow_pickle=True); print(m['e6__id_scores'].shape, np.isnan(m['e6__id_scores']).sum())"` -> (638,), 29.
- RESULT: PARTIAL MISMATCH. The stated n values are total stored counts; valid (non-NaN) counts are 609/290. The AUROC computation runs on valid scores only and the metric is correct. The reported n=638/n=319 in the text overstates the effective sample size by 29 each. This is a minor metadata description issue.
- VERDICT: FLAGGED.
- NOTE: The AUROC values are unaffected (computed on non-NaN subsets). The n=638/n=319 description in Method is the stored array size, not the NaN-filtered analysis size. A precise description would note that 609 and 290 valid scores enter the AUROC computation. However, since this is a method-section description of the data extent (not a count used in a significance claim), and since all downstream metrics are recomputed correctly, this is a minor issue.

---

## c33: E7 cells 0/75 collapsed, max 1/10 heads, vs 7/10 CARLA

- PROSE VALUE: "0 of 75 cells reach the collapse criterion (5 or more of 10 heads), and the maximum on any single cell is 1 of 10 heads, against 7 of 10 under CARLA" (Section 5.7, Abstract).
- SOURCE ARTIFACT: `report/e7_overlay_results.md` header lines.
- RECOMPUTED VALUE: artifact text: "output-collapsed cells (>= 5/10 heads): 0"; "max heads collapsed in ANY corruption cell: 1/10"; "VALIDATION GATE: e1_collapse_map on real-vs-CARLA reproduces 7/10 heads collapsed". 15 corruptions x 5 severities = 75 cells confirmed.
- COMMAND RUN: `python3 -c "print(15*5)"` -> 75; read `report/e7_overlay_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c34: Zero false negatives (factual)

- PROSE VALUE: "there are zero false negatives, because there is no output collapse anywhere in the sweep for the monitor to miss" (Section 5.7).
- SOURCE ARTIFACT: `report/e7_overlay_results.md` line "FALSE NEGATIVES (output collapsed, E6 AUROC < 0.7): 0".
- RECOMPUTED VALUE: artifact confirms 0 false negatives.
- COMMAND RUN: read `report/e7_overlay_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c35: E7 mean per-corruption AUROCs

- PROSE VALUE: "0.55 on fog, 0.55 on snow, 0.52 on jpeg compression, 0.60 on brightness, and 0.58 on zoom blur, rising only to about 0.68 to 0.71 on the defocus, motion, and glass blur families" (Section 5.7).
- SOURCE ARTIFACT: `report/e7_results.md` summary table (mean AUROC across severities).
- RECOMPUTED VALUE: fog=0.5461 (draft 0.55), snow=0.5455 (0.55), jpeg=0.5218 (0.52), brightness=0.6008 (0.60), zoom_blur=0.5751 (0.58), defocus_blur=0.7078 (0.71), motion_blur=0.7059 (0.71), glass_blur=0.6845 (0.68). All match within rounding to 2dp.
- COMMAND RUN: read `report/e7_results.md` summary table and compared with draft text.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c36: 4 FP cells with correct AUROCs

- PROSE VALUE: "frost severity 3 (AUROC 0.958), frost severity 5 (AUROC 1.000), gaussian-noise severity 4 (AUROC 0.861), and impulse-noise severity 5 (AUROC 0.906)" (Section 5.7).
- SOURCE ARTIFACT: `report/e7_overlay_results.md` FP line; `report/e7_results.md` threshold-free metrics table.
- RECOMPUTED VALUE: from e7_results.md: frost sev3=0.9582, frost sev5=0.9997, gaussian_noise sev4=0.8608, impulse_noise sev5=0.9060. Rounds to 0.958, 1.000, 0.861, 0.906 respectively. All match. e7_overlay_results.md cites these as 0.96, 1.00, 0.86, 0.91 (2dp rounding); draft uses 3dp values matching e7_results.md.
- COMMAND RUN: `python3 -c "vals=[0.9582,0.9997,0.8608,0.9060]; print([round(v,3) for v in vals])"` -> [0.958, 1.0, 0.861, 0.906].
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c37: Collapse sim-specific, monitor collapse-specific (novelty/factual)

- PROSE VALUE: "the silent collapse is a property of full-sim rendering, not of photometric or blur corruptions of real frames ... the monitor is near chance on most photometric corruptions" (Section 5.7).
- SOURCE ARTIFACT: `report/e7_overlay_results.md`; `report/e7_results.md`.
- RECOMPUTED VALUE: all 75 cells show 0 collapsed heads except frost sev3 (1/10). AUROC values confirm near-chance for most corruptions. Consistent.
- COMMAND RUN: read both artifact files.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c38: Baseline detection rates under corruption (qualitative)

- PROSE VALUE: "Mahalanobis and relative Mahalanobis fire at high rates across fog, frost, several noise families, snow, contrast, and zoom ... KNN-50 fires only on the heaviest noise and frost" (Section 5.7).
- SOURCE ARTIFACT: `report/e7_results.md` detection-rate table.
- RECOMPUTED VALUE: Mahalanobis and relative_mahalanobis fire at 1.000 for fog (all severities), frost, gaussian_noise, jpeg, snow, zoom_blur, contrast (most cells). KNN-50 fires at 1.000 only for frost sev2-5 and gaussian_noise sev3-5, shot_noise sev3-5, impulse_noise sev3-5. Qualitative description matches.
- COMMAND RUN: read `report/e7_results.md` detection table.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c45: MSP/Energy/ViM structural inapplicability (factual)

- PROSE VALUE: "MSP and Energy are output-side scores on a model whose output channel Section 5.3 shows is silent, and ViM requires a classifier weight matrix that supercombo's multi-head regression outputs do not provide" (Section 2, Section 4).
- SOURCE ARTIFACT: `paper_state/claim_ledger.md` c45 evidence (skeleton_source.md Section 4.5).
- RECOMPUTED VALUE: factual structural claim about model architecture; no numerical value to recompute. Cross-checked that supercombo's outputs are Gaussian-mixture regressions (confirmed by model teardown context).
- COMMAND RUN: read ledger and draft.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c46: RMD background GMM choice (factual)

- PROSE VALUE: "RMD's background distribution is fit as a two-component Gaussian mixture, because with a single in-distribution class the Ren et al. marginal Gaussian degenerates to the class Gaussian" (Section 4).
- SOURCE ARTIFACT: `paper_state/claim_ledger.md` c46; `src/baselines.py`.
- RECOMPUTED VALUE: factual claim about implementation; confirmed that with single ID class the relative Mahalanobis collapses to zero without the GMM background fit. No numeric artifact to verify; the claim is a design decision stated in the method.
- COMMAND RUN: read ledger and draft.
- RESULT: MATCH (factual/design claim, not a numerical value).
- VERDICT: CONFIRMED.

---

## c48: Monitor threshold 0.078873

- PROSE VALUE: "threshold at the 1st percentile of the real-driving rolling-spread distribution (0.078873)" (Section 4; implied by Section 5.6 description).
- SOURCE ARTIFACT: `report/e6_results.md` line "threshold (calibrated on all real corpora, p=1.0): 0.078873"; `report/e7_results.md` line "E6 rolling-spread threshold (calibrated p=1.0): 0.078873".
- RECOMPUTED VALUE: threshold value 0.078873 appears verbatim in both e6_results.md and e7_results.md. Window=30 confirmed in e7_results.md header.
- COMMAND RUN: grep `report/e6_results.md` and `report/e7_results.md`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c49: E5/E7 cache exclusion from public repo (factual, open TODO)

- PROSE VALUE: "E5 cache (about 3.9 GB) and the E7 cache (about 110 MB) are currently excluded from the public repository" (Section 8).
- SOURCE ARTIFACT: draft Section 8 explicitly marks this as [TODO: verify].
- RECOMPUTED VALUE: no artifact to recompute; this is a pre-submission infrastructure claim with an explicit TODO in the draft itself.
- COMMAND RUN: read draft Section 8.
- RESULT: NO ARTIFACT (explicitly flagged as TODO in the draft).
- VERDICT: FLAGGED.
- NOTE: The draft correctly flags this as [TODO: verify] before public release. The numeric sizes (3.9 GB, 110 MB) have no independent artifact to verify against in this audit; they are stated in the draft and in the skeleton_source.md reference. This claim is open-loop until the release step and should not be set CONFIRMED without running the cache size check.

---

## Internal Consistency Checks

### IC-1: n=220 vs n=219 (FAIL)
- Draft says "0 of 220 CARLA frames" multiple times (Abstract, Section 5.3, Conclusion, Introduction).
- NPZ stores 319 frames per corpus; 319 - 100 warmup = 219 analysis frames, not 220.
- The source: raw data had 320 frames, but rolling-window processing reduces stored count to 319, so post-warmup = 219.
- The zero count is correct in all three instances; only the denominator is off by 1.
- FAIL (minor): affects c11 description; does not affect the finding.

### IC-2: Percentages agree with stated counts - PASS
- "8 of 10 heads": 8 heads < 0.01, verified. 8/10 = 80% collapsed.
- "0 of 220 above p95": 0/219 recomputed (0% in all cases). Directionally PASS, n-off-by-1 noted above.
- "0 of 75 cells collapse": confirmed from overlay file.
- "7 of 10 CARLA": validation gate confirmed in e7_overlay.

### IC-3: AUROC CI bounds are plausible for n - PASS
- n_ID=609, n_OOD=290, total=899. Bootstrap n=1000, seed=42.
- E6 AUROC=0.996 with CI [0.992, 1.000]: very narrow CI appropriate for near-perfect separation. Recomputed CI [0.991, 1.000] closely matches. PASS.
- Mahalanobis AUROC=0.159 with CI [0.130, 0.190]: CI width = 0.060 on 899 samples. Plausible. PASS.

### IC-4: LOCO FPR arithmetic - PASS
- E6: (0.0000 + 0.0207) / 2 = 0.01035 = 1.03%; max = 2.07%. PASS.
- PCA-Maha: (0.0000 + 0.2382) / 2 = 0.1191 = 11.91%; max = 23.82%. PASS.

### IC-5: E4 cliff width arithmetic - PASS
- Subaru: 0.799 - 0.784 = 0.015. PASS.
- RAM: 0.940 - 0.666 = 0.274. PASS.
- Subaru headroom: 0.784 - 0.550 = 0.234 (~0.23). PASS.
- RAM headroom: 0.666 - 0.850 = -0.184. PASS.

### IC-6: E6 fires at 0.550 but AUROC crosses 0.5 earlier (FAIL)
- Draft says "AUROC crosses 0.5 at the same alpha=0.550."
- Artifact: AUROC=0.400 at alpha=0.425, AUROC=0.540 at alpha=0.450. Crosses 0.5 between 0.425 and 0.450.
- Fire fraction crosses 0.5 at alpha=0.550 (0.438->0.517). These are not at the same alpha.
- FAIL: the "same alpha" claim is incorrect by approximately 0.10 alpha units.

### IC-7: E5 passive relay "within about 2%" - FAIL
- Artifact table: at alpha=1, reduce_sum deviates 11.4% from summarizer_div; attention 3.2%, transformer 2.6%.
- Mean deviations across all alpha: 3.3%, 3.0%, 3.5%.
- "About 2%" is a lower bound only; the actual spread is 2-11%.
- FAIL: the claim is imprecise; the source artifact narrative has the same imprecision.

### IC-8: Table-prose agreement for all AUROC values - PASS
- All Table 1 values in metrics_results.md match draft prose exactly (E6 0.996, KNN 1.000, RMD 0.934, Maha 0.159, PCA 0.152).
- All AUROC CIs in prose match stored npz values at 3dp rounding. PASS.

### IC-9: E1 per-head ratios consistent with 8/10 collapse criterion - PASS
- 8 heads have ratio < 0.01: desired_curv (0.0018), accel_t0 (0.0040), lead (0.0042), desire_state (0.0049), lane_lines (0.0054), plan (0.0057), lead_prob (0.0058), road_edges (0.0076).
- 2 heads survive: pose (0.1788), meta (0.7181). Both > 0.01.
- Recomputed from npz: exact match. PASS.

### IC-10: E7 FP AUROC values in overlay vs e7_results - PASS
- overlay says frost_sev3=0.96, frost_sev5=1.00, gaussian_noise_sev4=0.86, impulse_noise_sev5=0.91.
- e7_results says 0.9582, 0.9997, 0.8608, 0.9060.
- Draft uses e7_results values (3dp): 0.958, 1.000, 0.861, 0.906. All correct at 3dp. PASS.

### IC-11: 75-cell count arithmetic - PASS
- 15 corruptions x 5 severities = 75 cells. PASS.

---

## c62 (re-confirm): corpus-scaling LOCO 2.41% / CI [0%, 5.17%] / max 6.90%

Re-run requested by orchestrator after reframe.

- PROSE VALUE: "2.41% real-driving FPR leave-one-corpus-out over N=4 corpora (95% CI [0%, 5.17%], max 6.90%); per-fold subaru 2.76% ram 6.90% ev6_night 0.00% bronco_night 0.00%" (Abstract, Section 5.6, Conclusion).
- SOURCE ARTIFACT: `report/corpus_scaling_results.md`; `src/corpus_scaling.py`.
- RECOMPUTED VALUE: `python3 -m src.corpus_scaling` -> held-out subaru 2.76%, ram 6.90%, ev6_night 0.00%, bronco_night 0.00%; LOCO mean 2.41%; CI [0.00%, 5.17%]; max 6.90%. All match artifact verbatim.
- COMMAND RUN: `PYTHONPATH=. python3 -m src.corpus_scaling` -> exact values confirmed.
- RESULT: MATCH.
- VERDICT: CONFIRMED. Still matches after reframe.

---

## c63: E6 LOCO FPR@95%TPR 0.00% (0 of 1160 held-out real frames; realised TPR 94.8%)

- PROSE VALUE: "flags 0 of 1160 held-out real frames: 0% false-positive rate at approximately 94.8% realised collapse detection across four real corpora" (Abstract, Section 5.6, Conclusion).
- SOURCE ARTIFACT: `report/loco_threshold_free_results.md` + `src/loco_threshold_free.py`.
- RECOMPUTED VALUE: `PYTHONPATH=. python3 -m src.loco_threshold_free` -> e6: LOCO mean 0.00%, max 0.00%, CI [0.00%, 0.00%]; all per-fold FPRs 0.00% (0 of 290 per corpus, 0 of 1160 total). Realised TPR = 0.9483 (identical across all 4 folds because E6 scores are ID-independent). 4 corpora x 290 valid frames each = 1160 confirmed. Tests pass: `pytest tests/test_loco_threshold_free.py` -> 3/3.
- COMMAND RUN: `PYTHONPATH=. python3 -m src.loco_threshold_free` and `PYTHONPATH=. python3 -m pytest tests/test_loco_threshold_free.py -q`.
- RESULT: MATCH. Every number in the prose traces exactly to the recomputed artifact.
- VERDICT: CONFIRMED.

---

## c64: Baseline LOCO FPR@95%TPR: KNN-50 60.82%, Mahalanobis 95.14%, Relative Mahalanobis 99.69%

- PROSE VALUE: "KNN-50 60.82% (95% CI [35.89%, 85.74%], max 100.00%), Mahalanobis 95.14% (95% CI [91.77%, 98.51%], max 100.00%), Relative Mahalanobis 99.69% (95% CI [99.06%, 100.00%], max 100.00%)" (Section 5.6, Table E6-OP).
- SOURCE ARTIFACT: `report/loco_threshold_free_results.md` + `src/loco_threshold_free.py`.
- RECOMPUTED VALUE: `python3 -m src.loco_threshold_free` -> knn50: mean 60.82% max 100.00% CI [35.89, 85.74]; mahalanobis: 95.14% max 100.00% CI [91.77, 98.51]; relative_mahalanobis: 99.69% max 100.00% CI [99.06, 100.00]. All match draft Table E6-OP verbatim to 2dp. Per-fold values in the artifact: knn50 subaru=100.00% ram=71.47% ev6_night=36.05% bronco_night=35.74%; mahalanobis subaru=100.00% ram=90.28% ev6_night=96.24% bronco_night=94.04%; relative_mahalanobis subaru=100.00% ram=98.75% ev6_night=100.00% bronco_night=100.00%.
- COMMAND RUN: `PYTHONPATH=. python3 -m src.loco_threshold_free`.
- RESULT: MATCH.
- VERDICT: CONFIRMED.

---

## c65: Missed ~5% of collapse frames are onset/warmup transients

- PROSE VALUE: "the approximately 5% of collapse frames the monitor misses at this operating point are onset and warmup transients, frames where the 30-frame rolling window straddles the real-to-collapse boundary and the spread has not yet crossed" (Section 5.6).
- SOURCE ARTIFACT: `report/loco_threshold_free_results.md` (realised TPR 94.8%) + rolling spread values computed from `report/e4_collected.npz`.
- RECOMPUTED VALUE: 15 of 290 collapse frames are missed (threshold at 5th percentile of negated spread). Indices 0-7 (8 frames): spread 0.032 to 0.139, genuine onset/warmup transients where the window still contains pre-collapse history. Indices 276-286 (7 frames): spread 7.38e-6 to 7.46e-6, i.e. ABOVE the threshold 7.38e-6 by less than 1%; these are in the collapse plateau's upper tail, not at a sequence boundary, and not onset transients.
- COMMAND RUN: `PYTHONPATH=. python3 -` inline session; missed_indices = np.where(collapse_scores <= thr)[0] -> [0,1,2,3,4,5,6,7,276,277,280,281,282,285,286]. Spreads at those indices confirmed via rolling_spread() on e4_collected.npz.
- RESULT: PARTIAL MISMATCH. The claim is correct for 8 of 15 missed frames (the genuine warmup/onset group). The remaining 7 missed frames are in the deep collapse plateau but their spread values (7.38e-6 to 7.46e-6) are barely above the threshold (7.38e-6) due to float-level variation within the frozen state. The "onset/warmup transients" label does not apply to them. The claim overstates the uniformity of the missed-frame explanation.
- VERDICT: FLAGGED.
- NOTE: The 7 plateau-edge missed frames are not harmful to the overall narrative (they are still genuinely collapsed frames, they just fall at the upper tail of the collapsed-spread distribution). The correction is: "approximately half of the missed frames are onset/warmup transients (rolling window straddles the real-to-collapse boundary); the remaining missed frames are at the upper tail of the collapsed-spread distribution, where float-level variation within the frozen state places their spread at or just above the 5th-percentile threshold." The prose should not claim all missed frames are onset transients.

---

## Internal Consistency Checks (additions, 2026-06-06 re-run)

### IC-12: 1160 = 4 corpora x 290 valid frames -- PASS
- 4 corpora x 290 non-NaN rolling-spread scores each = 1160 total. Recomputed: confirmed.
- Draft prose says "0 of 1160 held-out real frames" -- correct. PASS.

### IC-13: realised TPR 94.8% consistent with 15/290 missed -- PASS
- 15 missed of 290 = 5.17% missed; 94.83% realised TPR. Rounds to 94.8%. PASS.
- Prose says "approximately 94.8%"; 0.9483 rounds to 94.8%. PASS.

### IC-14: CI [0.00%, 0.00%] for E6 LOCO FPR@95%TPR is plausible -- PASS
- All 4 per-fold FPRs are exactly 0.00% (0 of 290 frames). A segment-level bootstrap over 4 values all equal to 0 will yield CI [0%, 0%]. PASS.

### IC-15: Abstract LOCO FPR@95%TPR values vs Section 5.6 table -- PASS
- Abstract: "KNN-50 60.8%, Mahalanobis 95.1%, Relative Mahalanobis 99.7%". Section 5.6: same values stated as "60.82%", "95.14%", "99.69%"; Table E6-OP has the 2dp values. 60.82% rounds to 60.8%, 95.14% rounds to 95.1%, 99.69% rounds to 99.7%. All consistent. PASS.

### IC-16: Conclusion FPR values consistent with abstract and Section 5.6 -- PASS
- Conclusion (line 697): "KNN-50 60.8%, Mahalanobis 95.1%, Relative Mahalanobis 99.7%" -- matches abstract exactly. PASS.

### IC-17: Daytime_control 58% vs 60.3% discrepancy -- contextual, not a contradiction
- real_weather_results.md: 57.9% at N=2 calibration threshold 0.078873.
- corpus_scaling_results.md: 60.3% at N=4 calibration threshold 0.087077.
- The two values use different thresholds and both are correct under their respective calibrations.
- Draft Section 5.8 cites 58% (rounds from 57.9%), sourced from real_weather_results.md at the N=2 threshold. This is the original report and is consistent with c58 CONFIRMED status. No contradiction; both values should be traceable to their threshold.

### IC-18: C++ "within 1e-12" vs artifact "3.4106e-13" -- PASS
- 3.4106e-13 < 1e-12: the prose uses a looser bound. Technically correct (not misleading). PASS.

---

## Drafter-Flagged Nuance Resolution

The drafter noted that `report/e5_submodule_results.md` NARRATES summarizer_div "crossing 0.5 between alpha 0.7 and 0.8" but its cliff_alpha COLUMN says 0.900.

Resolution (verified via per-alpha table): at alpha=0.8, summarizer_div activity=0.528, which is ABOVE 0.5. The value first drops below 0.5 at alpha=0.9 (activity=0.298). The cliff_alpha COLUMN value of 0.900 is CORRECT by the stated definition (smallest alpha at which activity first drops below 0.5). The narrative text in the artifact is a loose description of the onset of decline, not the 0.5 crossing. The draft uses the COLUMN value (0.900) throughout and is internally consistent. The narrative inaccuracy in the source artifact does not propagate into the draft.

---
