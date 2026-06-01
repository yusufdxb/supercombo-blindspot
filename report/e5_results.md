# E5 Results: Layer-Localized Collapse

Activity ratio = sum of per-element temporal std, CARLA / real (captures temporal variation).
Mean shift = sum |mean|(CARLA) / sum |mean|(real) at alpha=1 (captures DC offset).

| layer | cliff alpha | activity ratio @ alpha=1 | mean shift @ alpha=1 |
|---|---|---|---|
| head | N/A | 2.1416 | 1.3267 |
| stage0 | N/A | 0.9785 | 0.9386 |
| stage1 | N/A | 0.9560 | 0.9818 |
| stage2 | N/A | 1.2408 | 0.8440 |
| stage3 | N/A | 2.0561 | 0.9607 |
| stem | N/A | 1.4254 | 1.2352 |
