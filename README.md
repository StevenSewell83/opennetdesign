# netdesign

Optimal **microseismic network design** from a set of candidate station sites: which
stations, and how many, give the best *absolute* hypocentre locations over a target volume,
accounting for velocity-model uncertainty, detectability and station failure.

Built on [NonLinLoc](https://github.com/alomax/NonLinLoc) for travel times and validation.

## How it works

1. **Velocity ensemble** — a starting model (1D layered, or 3D nodes) perturbed in 1D
   (Vp, Vp/Vs, interfaces, Q) and, optionally, with 3D spatially correlated, geologically
   structured heterogeneity (anisotropic von Kármán fields per layer, interface relief).
2. **Travel times** — NonLinLoc `Vel2Grid`/`Grid2Time` for every candidate, phase and model
   (2D grids for layered models; 3D grids streamed and deleted as they are used).
3. **Sensitivities** — travel times and their spatial derivatives at target points.
4. **Error model** — linearised location error with velocity-model error correlated between
   stations, for a NonLinLoc-like locator (validated against NonLinLoc itself), optionally
   weighted by relative detectability (distance, Q, site κ₀/amplification/noise).
5. **Design** — greedy + exchange search with restarts for every network size, random-network
   baseline, one-station-failure robustness and a recommended size.
6. **QC** — checks and figures for every stage, collected in a self-contained HTML report.

## Install

Requires Linux or macOS and a NonLinLoc build (`Vel2Grid`, `Grid2Time`, `NLLoc`; v7
tested; see https://github.com/alomax/NonLinLoc). The installer sets up everything else —
[uv](https://docs.astral.sh/uv/) if missing, Python 3.11/3.12 and a `.venv` with all
dependencies:

```bash
./install.sh --nll-dir /path/to/NonLinLoc/bin --test   # --global: `netdesign` everywhere
```

(Equivalent by hand: `uv sync --all-extras`.)

Point the package at NonLinLoc with `nll.bin_dir` in the config (edit it in
`examples/config.yaml` for your machine; the tests use it too), or put the programs on `PATH`.

## Usage

The full user guide is in [docs/user-guide.md](docs/user-guide.md).

Everything is configured in one YAML file. Start a project from `templates/config.yaml`
(the commonly used parameters; fill in the lines marked `EDIT`); `examples/config.yaml`
lists every option with its default (synthetic example sites and model).

```bash
source /path/to/netdesign/.venv/bin/activate   # or prefix: uv run --project /path/to/netdesign
netdesign -c config.yaml            # everything: sensitivity, validate, design (+ QC report)
```

(the same as `netdesign run config.yaml`), or stage by stage (`netdesign -c config.yaml
sensitivity` also works):

```bash
netdesign sensitivity config.yaml   # ensemble, grids, travel times, sensitivities (+ QC)
netdesign validate    config.yaml   # NonLinLoc check of the predictions (validation.enabled)
netdesign design      config.yaml   # optimal networks and recommended size
netdesign report      config.yaml   # rebuild <output_dir>/qc/report.html
netdesign export      config.yaml   # shapefiles in <output_dir>/gis (candidates, targets, networks)
```

Intermediate commands `ensemble`, `grids` and `traveltimes` build and keep those stages only.

### Inputs

- **Candidate sites**: CSV `station,long,lat,z` (geographic CRS) or
  `station,easting,northing,z` (projected CRS); `z` is elevation in metres above mean sea
  level (e.g. from a DEM). The CRS of all coordinate inputs is set by EPSG code (`crs.epsg`).
  Or generate them: `netdesign sites config.yaml` lays out a grid (`sites`: centre, spacing,
  extent, rectangle/ellipse, rotation) and takes `z` from a DEM, either Copernicus GLO-30/90
  (downloaded for the area only and cached) or a local GeoTIFF. It writes
  `stations.candidates` (`--force` to overwrite) and a map in `qc/sites_map.png`. Sites with
  no DEM data (sea) or below `min_elevation_m` are dropped.
- **Velocity model**: NonLinLoc `LAYER` lines or CSV (layered), or a CSV of 3D nodes.
- **Target volume**: a box and depth range where events are expected.

### Outputs (`<output_dir>/`)

- `design/` — best network per size, recommended network.
- `gis/` (`netdesign export`) — shapefiles in the input CRS: input candidates, target boxes,
  candidates with their design outcome, the final network (by rank), every network size.
- `qc/report.html` — all checks and figures; `qc/candidates_outcome.csv`/`.geojson` —
  every candidate with its status, rank and contribution.
- `sensitivity/`, `validation/`, `models/` — intermediate results.

## Development

```bash
scripts/check.sh   # ruff format/lint, mypy, pytest
```
