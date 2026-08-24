# Reproducing supercombo-blindspot

The repository has three evidence levels. Do not treat them as interchangeable.

## 1. Verify the public package

This checks 16 manuscript claims against their evidence files, validates artifact hashes, verifies every
paper figure exists, and checks the anonymous manuscript scrub:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-ci.txt matplotlib scikit-learn
env -u PYTHONPATH .venv/bin/python -m scripts.verify_paper
env -u PYTHONPATH .venv/bin/python -m scripts.build_pdf --check
```

The claim boundary is in `paper/claim_matrix.md`; artifact hashes and reproduction levels are in
`paper/artifact_manifest.json`.

Containerized test environment:

```bash
docker build -t supercombo-blindspot .
docker run --rm supercombo-blindspot
```

The container covers the CPU test and artifact-verification environment. It does not provide CARLA,
the released ONNX model, source driving footage, or GPU inference recollection.

## 2. Recompute results from tracked caches

The core path recomputes E1-E4, the v0.9.6 teardown, real-weather controls, detector baselines,
threshold-free metrics, four-corpus calibration, the collapse-aware operating-point diagnostic,
conformal results, lead-time analysis, and ablations:

```bash
bash scripts/repro_from_caches.sh
```

This path needs no CARLA instance, model inference, raw video, or GPU. It does not rerun the parity
inference or the large E4-RAM, E5, and E7 activation collections.

## 3. Recompute extended local-cache results

If the undistributed E4-RAM, E5-submodule, and E7 activation caches are present:

```bash
bash scripts/repro_from_caches.sh --extended
```

The script fails rather than silently skipping a requested extended cache. On a fresh clone, the
corresponding committed result tables and figures remain inspectable and hash-verified, but they are not
independently recomputed. The larger E5 layer cache is unavailable; that localization result is inspect
only until a replacement public summary cache is produced.

## 4. Build the paper

With Pandoc and WeasyPrint installed:

```bash
env -u PYTHONPATH .venv/bin/python -m scripts.build_pdf
```

Outputs:

- `paper/build/manuscript.pdf`
- `paper/build/manuscript_anonymous.pdf`
- corresponding assembled Markdown files

## 5. Recollect inference artifacts

Recollection requires the released model, source frames, and the dependencies in `requirements.txt`.
Parity additionally requires comma's logged reference output. The collection entry points are:

```bash
python -m src.teardown --collect
python -m src.e4_interp --collect
python -m src.e4_ram --collect
python -m src.e5_submodule --collect
python -m src.e7_corruption --collect
python -m src.real_weather --collect
python -m src.run_parity
```

Model files and source driving segments are fetched separately and are not redistributed here.
