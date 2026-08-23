# Data

Two case studies, two very different availability situations.

## COVID-19 (public)

Anonymised patient, laboratory and outcome records from Hospital Sírio-Libanês,
published through the **FAPESP COVID-19 Data Sharing/BR** repository:

<https://repositoriodatasharingfapesp.uspdigital.usp.br/>

Download the Hospital Sírio-Libanês set (registration required) and unzip it.
You need three pipe-separated files:

```
HSL_Pacientes_4.csv     ~15k patients
HSL_Exames_4.csv        ~3.0M laboratory results
HSL_Desfechos_4.csv     ~90k encounter outcomes
```

Then:

```bash
personareact data build --domain covid --source /path/to/unzipped
personareact data verify --update      # pins the checksum in configs/domain/covid.yaml
```

The build takes a few minutes and produces `data/covid.sqlite` (~500 MB) with
three tables and indexes on the join keys.

## Football (restricted)

Physical and physiological tracking for every Fortaleza E.C. player across the
2023 season, captured with Polar Team Pro wearables at 1 Hz. Released to the
author under an agreement with the club and **cannot be redistributed** —
section 5.1 of the thesis states this.

What *is* in this repository: the complete schema, the builder, the corrected
views, and the full question corpus. So the pipeline is auditable, and it runs
against any database with the same schema.

If you have the processed files, the builder expects two CSVs:

```
game_trajectories_processed.csv   ~2.5M trajectory points, 30-point segments
stats_processed.csv               ~165k per-segment statistics
```

```bash
personareact data build --domain soccer --source /path/to/csvs
personareact data verify --update
```

### A missing step

The pipeline that turns the raw Polar GPX exports into those two CSVs — the
30-point segmentation and the recomputation of speed, acceleration, direction
and distance described in section 5.1 — was **not** in the original repository
and is not here. `create_database.py` consumed the processed CSVs directly. If
the segmentation code still exists somewhere, it belongs in
`src/personareact/datasets/` and this note should be replaced.

## Importing from the original tree

If you have the twelve-repository tree, you can skip the rebuild:

```bash
personareact data import-legacy --legacy-root ~/doutorado/implementacoes
personareact data verify --update
```

This copies the prebuilt databases **and repairs the two broken football views**
(`docs/ERRATA.md`, SV-15). Views are metadata, so no data is touched.

## Why the checksums matter

The original tree held nine copies of the football database across nine project
folders, which drifted into three different builds — and the treatment and the
baseline of the main experiment ended up querying different ones. See SV-14.

Every domain config pins a SHA-256. It is verified before every run and a
mismatch aborts, rather than producing results that silently are not comparable
with the previous ones.
