# Reproducibility Guide

The artifact has three reproduction levels. The distinction is enforced by
`paper/artifact_manifest.json` and checked by `scripts/verify_paper.py`.

| Level | Requirements | What it proves |
|---|---|---|
| Verify | CPU and Python | Committed evidence hashes match and manuscript claims still resolve to those artifacts |
| Core recomputation | CPU, pinned environment, tracked caches | Recomputes E1-E4, v0.9.6, real-weather, baselines, calibration, operating-point diagnostics, and ablations |
| Extended recomputation | Core requirements plus local large caches | Also recomputes E4-RAM, E5-submodule, and E7 |
| Recollection | Model, source frames, inference dependencies, and accelerator | Recreates activation caches from inference |

## Public verification

```bash
env -u PYTHONPATH .venv/bin/python -m scripts.verify_paper
env -u PYTHONPATH .venv/bin/python -m scripts.build_pdf --check
```

The first command verifies 16 claim-to-evidence links and 37 artifact hashes. The second checks the
paper source, bibliography parse, figure inventory, and anonymous-review scrub.

## Core recomputation

```bash
bash scripts/repro_from_caches.sh
```

The core path uses only these tracked activation caches:

- `report/teardown_collected.npz`
- `report/teardown_v096_collected.npz`
- `report/e4_collected.npz`
- `report/e4_v096_collected.npz`
- `report/real_weather_collected.npz`

It also regenerates the derived baseline, metric, and ablation caches. The script exits on a missing
cache or failed analysis and finishes by running the paper verifier.

## Extended recomputation

```bash
bash scripts/repro_from_caches.sh --extended
```

This additionally requires:

- `report/e4_ram_collected.npz`
- `report/e5_submodule_collected.npz`
- `report/e7_collected.npz`

These files are not distributed because of their size. Their result tables and figures are committed
and hash-verified, but a fresh clone cannot independently recompute them. The E5 layer cache is also
unavailable, so the encoder-stage localization is inspect only. These are artifact limitations, not
successful reproduction claims.

## Raw recollection

Raw collection requires separately obtained source footage and released model files. The main entry
points are:

```bash
env -u PYTHONPATH .venv/bin/python -m src.teardown --collect
env -u PYTHONPATH .venv/bin/python -m src.teardown_v096 --collect
env -u PYTHONPATH .venv/bin/python -m src.e4_interp --collect
env -u PYTHONPATH .venv/bin/python -m src.e4_interp_v096 --collect
env -u PYTHONPATH .venv/bin/python -m src.e4_ram --collect
env -u PYTHONPATH .venv/bin/python -m src.e5_submodule --collect --alphas 11 --frames 320
env -u PYTHONPATH .venv/bin/python -m src.e7_corruption --collect
env -u PYTHONPATH .venv/bin/python -m src.real_weather --collect
env -u PYTHONPATH .venv/bin/python -m src.run_parity
```

The source segments are public comma driving records fetched by `scripts/fetch_upgrade_data.py`. The
repository does not redistribute video, logs, or model binaries. The parity result requires comma's
logged reference output and therefore is not part of the tracked-cache recomputation path.

## Paper build

```bash
env -u PYTHONPATH .venv/bin/python -m scripts.build_pdf
```

Pandoc and WeasyPrint produce author and anonymous PDFs under `paper/build/`. The generated build
directory is ignored; `paper/manuscript.md`, `paper/references.bib`, the claim matrix, and build script
are the versioned sources.

## Provenance updates

After intentionally regenerating evidence files, update and review the artifact manifest:

```bash
env -u PYTHONPATH .venv/bin/python -m scripts.update_artifact_manifest
git diff -- paper/artifact_manifest.json report/
env -u PYTHONPATH .venv/bin/python -m scripts.verify_paper
```

Do not update hashes merely to silence a mismatch. The report diff must first show that the regenerated
result is expected.
