# E7 overlay: E1 output-collapse vs E6 detection (cell-for-cell)

Resolves the E7 `[AUTHOR TODO]`: is E6's quiet response on a corruption correct (no output collapse) or a miss (collapse undetected)? Metric: the canonical `teardown.e1_collapse_map` (head collapsed at activity ratio < 0.1); a cell is output-collapsed at >= 5/10 heads collapsed. E6 AUROC read from the verified `e7_results.md`.

- VALIDATION GATE: e1_collapse_map on real-vs-CARLA reproduces **7/10** heads collapsed (published collapse confirmed).
- corruption cells evaluated: **75**
- output-collapsed cells (>= 5/10 heads): **0**
- max heads collapsed in ANY corruption cell: **1/10**
- **FALSE NEGATIVES (output collapsed, E6 AUROC < 0.7): 0**
- E6 fires with NO output collapse (FP): **4**: frost sev3 (AUROC 0.96), frost sev5 (AUROC 1.00), gaussian_noise sev4 (AUROC 0.86), impulse_noise sev5 (AUROC 0.91)

> Result (read carefully): **no ImageNet-C corruption cell collapses the output** (max 1/10 heads, vs 7/10 under CARLA). Two consequences: (1) the false-negative question is resolved trivially, there is no collapse to miss; (2) the **silent-collapse failure mode is CARLA / full-sim specific and does NOT reproduce under ImageNet-C corruptions of real frames.** The FP cells above are E6 firing on a recurrent-feature spread shift that is NOT an output collapse, so on this corpus E6's firings are decoupled from the collapse mode. **Draft implication:** the current E7 wording (E6 fires on corruptions that 'induce the same recurrent freeze' as CARLA) overstates the link, because no output collapse occurs here; the defensible E7 claim is narrower, the collapse is sim-specific and E6 is collapse-specific (and mostly quiet on real-frame corruptions).

| corruption | sev | heads collapsed (/10) | E6 AUROC | verdict |
|---|---|---|---|---|
| brightness | 1 | 0 | 0.661 | TN (correctly quiet) |
| brightness | 2 | 0 | 0.628 | TN (correctly quiet) |
| brightness | 3 | 0 | 0.602 | TN (correctly quiet) |
| brightness | 4 | 0 | 0.578 | TN (correctly quiet) |
| brightness | 5 | 0 | 0.534 | TN (correctly quiet) |
| contrast | 1 | 0 | 0.704 | marginal |
| contrast | 2 | 0 | 0.804 | marginal |
| contrast | 3 | 0 | 0.823 | marginal |
| contrast | 4 | 0 | 0.769 | marginal |
| contrast | 5 | 0 | 0.621 | TN (correctly quiet) |
| defocus_blur | 1 | 0 | 0.713 | marginal |
| defocus_blur | 2 | 0 | 0.714 | marginal |
| defocus_blur | 3 | 0 | 0.717 | marginal |
| defocus_blur | 4 | 0 | 0.708 | marginal |
| defocus_blur | 5 | 0 | 0.688 | TN (correctly quiet) |
| elastic_transform | 1 | 0 | 0.688 | TN (correctly quiet) |
| elastic_transform | 2 | 0 | 0.679 | TN (correctly quiet) |
| elastic_transform | 3 | 0 | 0.671 | TN (correctly quiet) |
| elastic_transform | 4 | 0 | 0.661 | TN (correctly quiet) |
| elastic_transform | 5 | 0 | 0.645 | TN (correctly quiet) |
| fog | 1 | 0 | 0.564 | TN (correctly quiet) |
| fog | 2 | 0 | 0.536 | TN (correctly quiet) |
| fog | 3 | 0 | 0.547 | TN (correctly quiet) |
| fog | 4 | 0 | 0.540 | TN (correctly quiet) |
| fog | 5 | 0 | 0.543 | TN (correctly quiet) |
| frost | 1 | 0 | 0.538 | TN (correctly quiet) |
| frost | 2 | 0 | 0.590 | TN (correctly quiet) |
| frost | 3 | 1 | 0.958 | FP (E6 fires, no collapse) |
| frost | 4 | 0 | 0.471 | TN (correctly quiet) |
| frost | 5 | 0 | 1.000 | FP (E6 fires, no collapse) |
| gaussian_noise | 1 | 0 | 0.529 | TN (correctly quiet) |
| gaussian_noise | 2 | 0 | 0.579 | TN (correctly quiet) |
| gaussian_noise | 3 | 0 | 0.596 | TN (correctly quiet) |
| gaussian_noise | 4 | 0 | 0.861 | FP (E6 fires, no collapse) |
| gaussian_noise | 5 | 0 | 0.751 | marginal |
| glass_blur | 1 | 0 | 0.709 | marginal |
| glass_blur | 2 | 0 | 0.687 | TN (correctly quiet) |
| glass_blur | 3 | 0 | 0.685 | TN (correctly quiet) |
| glass_blur | 4 | 0 | 0.671 | TN (correctly quiet) |
| glass_blur | 5 | 0 | 0.671 | TN (correctly quiet) |
| impulse_noise | 1 | 0 | 0.568 | TN (correctly quiet) |
| impulse_noise | 2 | 0 | 0.552 | TN (correctly quiet) |
| impulse_noise | 3 | 0 | 0.531 | TN (correctly quiet) |
| impulse_noise | 4 | 0 | 0.797 | marginal |
| impulse_noise | 5 | 0 | 0.906 | FP (E6 fires, no collapse) |
| jpeg_compression | 1 | 0 | 0.585 | TN (correctly quiet) |
| jpeg_compression | 2 | 0 | 0.565 | TN (correctly quiet) |
| jpeg_compression | 3 | 0 | 0.590 | TN (correctly quiet) |
| jpeg_compression | 4 | 0 | 0.452 | TN (correctly quiet) |
| jpeg_compression | 5 | 0 | 0.416 | TN (correctly quiet) |
| motion_blur | 1 | 0 | 0.714 | marginal |
| motion_blur | 2 | 0 | 0.717 | marginal |
| motion_blur | 3 | 0 | 0.711 | marginal |
| motion_blur | 4 | 0 | 0.702 | marginal |
| motion_blur | 5 | 0 | 0.686 | TN (correctly quiet) |
| pixelate | 1 | 0 | 0.700 | marginal |
| pixelate | 2 | 0 | 0.683 | TN (correctly quiet) |
| pixelate | 3 | 0 | 0.685 | TN (correctly quiet) |
| pixelate | 4 | 0 | 0.683 | TN (correctly quiet) |
| pixelate | 5 | 0 | 0.675 | TN (correctly quiet) |
| shot_noise | 1 | 0 | 0.462 | TN (correctly quiet) |
| shot_noise | 2 | 0 | 0.686 | TN (correctly quiet) |
| shot_noise | 3 | 0 | 0.756 | marginal |
| shot_noise | 4 | 0 | 0.803 | marginal |
| shot_noise | 5 | 0 | 0.387 | TN (correctly quiet) |
| snow | 1 | 0 | 0.590 | TN (correctly quiet) |
| snow | 2 | 0 | 0.507 | TN (correctly quiet) |
| snow | 3 | 0 | 0.527 | TN (correctly quiet) |
| snow | 4 | 0 | 0.536 | TN (correctly quiet) |
| snow | 5 | 0 | 0.567 | TN (correctly quiet) |
| zoom_blur | 1 | 0 | 0.609 | TN (correctly quiet) |
| zoom_blur | 2 | 0 | 0.569 | TN (correctly quiet) |
| zoom_blur | 3 | 0 | 0.513 | TN (correctly quiet) |
| zoom_blur | 4 | 0 | 0.546 | TN (correctly quiet) |
| zoom_blur | 5 | 0 | 0.639 | TN (correctly quiet) |
