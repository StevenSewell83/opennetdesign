# netdesign user guide

netdesign chooses which stations to install, and how many, from a set of candidate sites so
that a microseismic network locates events in a target volume as accurately as possible. It
designs for **absolute** hypocentre accuracy and accounts for:

- velocity-model uncertainty,
- whether each station can actually pick a small event (detectability),
- one station failing.

Travel times come from [NonLinLoc](https://github.com/alomax/NonLinLoc), and NonLinLoc is
also used to check the predicted location errors.

This guide covers installation, setting up a project, every input, the commands, how to
read the results, and how to choose the main settings. `examples/config.yaml` lists every
config option with its default, and `templates/config.yaml` is the starting point for a new
project.

---

## Contents

1. [How it works](#1-how-it-works)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Setting up a project](#4-setting-up-a-project)
5. [Inputs](#5-inputs)
6. [Configuration reference](#6-configuration-reference)
7. [Commands](#7-commands)
8. [Recommended workflow](#8-recommended-workflow)
9. [Reading the results](#9-reading-the-results)
10. [Choosing the main settings](#10-choosing-the-main-settings)
11. [Run time and disk space](#11-run-time-and-disk-space)
12. [Troubleshooting](#12-troubleshooting)
13. [Worked example: Mak-Ban](#13-worked-example-mak-ban)

---

## 1. How it works

The pipeline has three main stages: **sensitivity**, **validate** (optional) and
**design**. Each stage writes QC checks and figures to an HTML report.

1. **Velocity ensemble.** The starting model (1D layered, or 3D nodes) is perturbed many
   times to represent what you don't know about it: layer Vp, Vp/Vs, interface depths and
   Q. Optionally, each realisation also gets 3D heterogeneity: spatially correlated random
   fields per layer, stretched along a structural strike, plus relief on the interfaces.
2. **Travel times.** NonLinLoc `Vel2Grid` and `Grid2Time` compute P and S travel-time
   grids for every candidate site in every realisation. Layered models use fast 2D grids;
   3D models use 3D grids, which are made and deleted in batches to stay within disk space.
3. **Sensitivities.** Travel times and their spatial derivatives are taken at the target
   points (a grid or random sample of where events are expected).
4. **Error model.** For any set of stations, netdesign predicts the location error of a
   NonLinLoc-like locator that uses the starting model while the true earth is one of the
   realisations. Model errors are correlated between stations, so adding nearby stations
   helps less than independent picks would suggest. With detection on, each station's
   picks are weighted by how likely it is to see the event.
5. **Design.** For every network size, a greedy search followed by station exchanges (from
   several random restarts) finds the best network. Random networks of the same size give a
   baseline, and each best network is re-scored with each station removed (robustness). A
   recommended size is chosen from your accuracy goals, or else from the point where an
   extra station stops helping much.
6. **Validation** (optional). NonLinLoc locates synthetic events whose "true" travel times
   come from random realisations, with pick noise. Its mislocations are compared with the
   predictions to show that the error model can be trusted.

---

## 2. Installation

**Requirements:** Linux or macOS, and a NonLinLoc build containing `Vel2Grid`, `Grid2Time`
and `NLLoc` (v7 is tested). Build NonLinLoc from https://github.com/alomax/NonLinLoc.

```bash
./install.sh --nll-dir /path/to/NonLinLoc/src/bin --test
```

The installer:

- installs [uv](https://docs.astral.sh/uv/) if it is missing (asks first; `--yes` skips
  the question),
- creates `.venv` with Python 3.11/3.12 and all dependencies (`uv sync --all-extras`),
- checks that the NonLinLoc programs are in `--nll-dir` (or on `PATH`),
- with `--global`, makes the `netdesign` command available everywhere (`uv tool`),
- with `--test`, runs the test suite.

Then use netdesign in one of these ways:

```bash
source /path/to/netdesign/.venv/bin/activate     # then: netdesign ...
uv run --project /path/to/netdesign netdesign ...  # without activating
netdesign ...                                      # after ./install.sh --global
```

Tell netdesign where NonLinLoc is with `nll.bin_dir` in each config (or put the programs
on `PATH`).

---

## 3. Quick start

```bash
mkdir myproject && cd myproject
cp /path/to/netdesign/templates/config.yaml .
# edit every line marked EDIT; add candidates.csv and a velocity model
netdesign -c config.yaml        # or: netdesign run config.yaml
```

When it finishes:

- the terminal shows the best network for every size and the recommended one,
- `out/design/design.json` and `out/design/networks.csv` hold the results,
- `out/qc/report.html` has every check and figure,
- `netdesign export config.yaml` writes shapefiles to `out/gis/`.

---

## 4. Setting up a project

A project is a folder with one YAML config and its input files:

```
myproject/
  config.yaml        all settings
  candidates.csv     candidate sites (or generate them: `netdesign sites`)
  velocity.csv       starting velocity model (or .layer, or a 3D node CSV)
  out/               results (output_dir)
```

- **Relative paths** in the config are relative to the config file, not to where you run
  the command. This applies to `output_dir`, `velocity.base`, `velocity.model_3d`,
  `stations.candidates`, `nll.bin_dir` and the DEM paths.
- **All settings live in the config.** The only command-line option is `--out`/`-o`, which
  overrides `output_dir`.
- **Unknown keys are errors.** A typo in a key name stops the run with a message naming
  the field, so a misspelt setting is never silently ignored.
- Keep one config per run (for example `config.yaml` for a design, `config_validation.yaml`
  for checking one network) with different `output_dir`s, so runs don't overwrite each
  other.

---

## 5. Inputs

### 5.1 Coordinate system

`crs.epsg` sets the coordinate system of **all** horizontal coordinates in the inputs:
candidates, target, grid origin, sites centre and 3D model.

- Geographic (e.g. `4326` WGS-84): columns and keys are `long` and `lat`.
- Projected (e.g. `2193` NZTM2000): columns and keys are `easting` and `northing`, in
  metres.

Elevations (`z`) are metres above mean sea level. Depths (`depth_km`) are km below sea
level, so negative depths are above sea level.

### 5.2 Candidate sites

A CSV with a header, one row per possible station site:

```
station,long,lat,z
C0171,121.15520,14.09793,158
C0172,121.15520,14.09612,151
```

(`station,easting,northing,z` for a projected CRS.) Lines starting with `#` are comments.

- `station`: unique name, used everywhere else in the config (`design.fixed`,
  `detection.sites`, `validation.stations`).
- `z`: sensor elevation in m above sea level. For a borehole sensor, give the sensor's
  elevation, not the wellhead's.

**Generating candidates on a grid.** `netdesign sites config.yaml` lays out a regular
grid of sites from the `sites` section and takes each site's elevation from a DEM:

```yaml
sites:
  centre: {lat: -38.605, long: 176.13}
  spacing_m: 1000.0
  extent_m: [6000.0, 6000.0]     # full width across and along the grid axes
  shape: rectangle               # or ellipse
  rotation_deg: 0.0
  min_elevation_m: 1.0           # drop sea and shore sites
  dem:
    source: copernicus30         # copernicus30 | copernicus90 (downloaded) | file
```

It writes the file named in `stations.candidates` (use `--force` to overwrite an existing
one) and a map in `<output_dir>/qc/sites_map.png`. The Copernicus DEMs are downloaded for
the area only and cached in `dem.cache_dir`. Sites with no DEM data (sea) or below
`min_elevation_m` are dropped. With `source: file`, give a local GeoTIFF in `dem.path`
(any CRS, metres above sea level).

### 5.3 Velocity model

Give exactly one of `velocity.base` (layered) or `velocity.model_3d` (3D nodes).

**Layered model, CSV** (`.csv`): one row per layer, values at the layer top. `top_km`,
`vp` and `vs` are required; `rho`, `vp_grad`, `vs_grad`, `rho_grad`, `qp`, `qs`,
`qp_grad` and `qs_grad` are optional.

```
# comments allowed
top_km,vp,vs
-1.10,2.43,1.406
0.50,3.70,2.141
1.00,3.80,2.199
```

The first layer's top must be above the highest station. Otherwise that station sits
outside the model.

**Layered model, NonLinLoc** (any other extension): NonLinLoc `LAYER` lines:

```
# LAYER  depth  VpTop VpGrad  VsTop VsGrad  rhoTop rhoGrad
LAYER  -0.50  2.00  0.80  1.10  0.45  2.20  0.00
```

**3D model** (`velocity.model_3d`, needs `grid.mode: 3d`): a CSV of nodes on a full
regular grid, such as a tomography export:

```
long,lat,depth_km,vp,vs        (or vpvs instead of vs; easting,northing for projected)
```

The model is interpolated (trilinear) onto the design grid, and outside the model's extent
the nearest edge value is used. With a 3D starting model, `perturb` uses only global
`vp_pct` and `vpvs`, and `heterogeneity` needs scalar amplitudes and no interface relief.

### 5.4 Target volume

Where events are expected. Either one box:

```yaml
target:
  long: [176.00, 176.10]
  lat: [-38.70, -38.60]
  depth_km: [0.5, 4.0]
```

or several boxes, each optionally rotated and with its own depth range and share of the
events:

```yaml
target:
  depth_km: [1.0, 4.0]                 # default for every box
  boxes:
    - name: main
      centre: {lat: 14.103698, long: 121.227257}
      size_km: [1.0, 1.0]              # [across, along]
      rotation_deg: 45.0               # 'along' axis, clockwise from north
      weight: 8.0                      # relative share of events
    - name: secondary
      centre: {lat: 14.0933069, long: 121.1928637}
      size_km: [0.5, 0.5]
      weight: 2.0
```

If no box has a `weight`, events are spread evenly through the total volume. If one box has
a weight, every box needs one.

The target is sampled on a grid (`spacing_km`, `depth_spacing_km`) or at random points
(`sampling: random`, `n_points`). `depth_weight: {model: normal, mean_km: 2.5, sd_km:
0.75}` concentrates events around a depth; the default is uniform.

---

## 6. Configuration reference

This section explains what each section controls and which settings matter most. For every
option and its default, see `examples/config.yaml`.

### `output_dir`, `crs`, `phases`

| Key | Default | Meaning |
|---|---|---|
| `output_dir` | `out` | Results folder, relative to the config file. `--out` overrides it. |
| `crs.epsg` | `4326` | EPSG code of all horizontal inputs (see 5.1). |
| `phases` | `[P, S]` | Phases to model. |

### `stations`

| Key | Meaning |
|---|---|
| `candidates` | Candidate-site CSV (see 5.2). |

### `velocity`

| Key | Meaning |
|---|---|
| `base` / `model_3d` | Starting model (see 5.3). Give exactly one. |
| `attenuation` | Q, used only by detection. By default `Qs = qs_per_vs × Vs[km/s]` and `Qp = qp_qs_ratio × Qs`. Leave it out for no Q. `model: layered` takes `qp`/`qs` from a CSV model. |
| `perturb` | 1D model uncertainty: the velocity ensemble. |
| `heterogeneity` | 3D spatially correlated variation on top of each perturbed model (needs `grid.mode: 3d`). |

**`perturb`** (1D uncertainty):

| Key | Example | Meaning |
|---|---|---|
| `enabled` | `true` | Default `false`: starting model only, so no model error. |
| `n_realizations` | `100` | Ensemble size. More gives smoother statistics but takes longer. 10 is enough for a quick pass; 30 to 100 for a final design. |
| `include_base` | `true` | `model_000` is the unperturbed starting model. Validation needs it. |
| `vp_pct` | `{sd: 5.0, mode: per_layer}` | Vp uncertainty, % (1 sd). `global` moves all layers together; `per_layer` draws each layer separately. |
| `vpvs` | `{sd: 0.03, mode: global}` | Additive Vp/Vs uncertainty. |
| `interface_sd_km` | `0.05` | Interface depth uncertainty (layered models only). |
| `layer_corr` | `0.5` | Correlation between adjacent layers' `per_layer` draws. |
| `q_log10` | `{sd: 0.3}` | Q uncertainty in log10 (0.3 is about a factor of 2). |
| `seed` | | Random seed. The same seed gives the same ensemble. |

**`heterogeneity`** (3D variation):

| Key | Example | Meaning |
|---|---|---|
| `enabled` | `true` | Default `false`. |
| `structure.correlation` | `von_karman` | Or `exponential` / `gaussian`. |
| `structure.hurst` | `0.3` | von Kármán roughness (0 to 1; smaller is rougher). |
| `structure.corr_len_km` | `[3.0, 1.0, 0.4]` | Correlation lengths along strike, across strike and vertically. |
| `structure.strike_deg` | `45.0` | Structural trend, degrees clockwise from north. |
| `per_layer` | `true` | An independent field in each layer, so properties change abruptly at interfaces. |
| `vp_pct` | `5.0` or `[8, 6, 5, 3, 2]` | Vp sd in %: one value, or one per layer. |
| `vpvs_sd` | `0.03` | Vp/Vs sd. |
| `interface_sd_km` | `0.05` | Interface relief sd: one value, or one per interface. |

`model_000` (the base model) never gets heterogeneity.

### `grid`

| Key | Example | Meaning |
|---|---|---|
| `mode` | `2d` / `3d` | `2d` uses NonLinLoc 2D grids and is fast, but only for a layered model without heterogeneity. `3d` is needed for `heterogeneity` or `model_3d`. |
| `dx_km` | `0.05` to `0.1` | Cell size. Disk use and run time grow as `1/dx³` in 3D. |
| `depth_max_km` | `6.0` | Grid bottom, km below sea level. Put it below the deepest target, with margin. |
| `padding_km` | `2.0` | Extra grid around the candidates. |
| `origin` | | Local-frame origin. The default is the candidates' centroid. |

### `nll`

| Key | Example | Meaning |
|---|---|---|
| `bin_dir` | path | Folder with the NonLinLoc programs. Default: `PATH`. |
| `jobs` | `4` | NonLinLoc runs in parallel. Default: number of CPUs. |
| `keep_grids` | `base` | Grids kept after `sensitivity`: `none`, `base` (`model_000`, which `validate` needs) or `all`. |
| `max_disk_gb` | `null` | Disk space travel-time grids may use at once. Grids not kept are made and deleted in station batches that fit. Default: half the free space. |

### `picks`

| Key | Example | Meaning |
|---|---|---|
| `sigma_s` | `{P: 0.010, S: 0.020}` | Pick uncertainty (1 sd), in seconds. |
| `model_error` | `correlated` | How model error enters: `none`, `independent` or `correlated` (the ensemble covariance between picks). Leave this as `correlated`. |

### `detection`

Detection decides how likely each station is to pick each phase of the design event. With
`enabled: false`, every station picks every phase.

| Key | Example | Meaning |
|---|---|---|
| `enabled` | `true` | Default `false`. |
| `mode` | `magnitude` | `magnitude`: absolute SNR of an Mw `magnitude` event against site noise. `relative`: pick probability relative to the best sites, set by `margin_db`. |
| `magnitude` | `0.5` | Smallest event the network must locate (Mw). |
| `magnitude_sensitivity` | `[-0.5, 0.5, 1.0]` | Extra magnitudes the design is re-run at, to show how the recommendation changes. `[]` turns this off. |
| `snr_threshold_db` | `10.0` | SNR at which the pick probability is 50 %. |
| `width_db` | `5.0` | How sharply the probability rises around the threshold (1 sd). |
| `margin_db` | `20.0` | Relative mode: smaller values design for smaller events. |
| `frequency_band_hz` | `[5, 30]` | Picking band. |
| `stress_drop_mpa` | `1.0` | Brune stress drop, which sets the source corner frequency. |
| `default_site` | `surface` | Site type of stations not listed in `sites`. |
| `sites` | `{BH01: borehole}` | Station-to-site-type mapping. |
| `site_types` | | Per type: `kappa_s` (near-surface attenuation), `amplification`, `noise_psd_db` (magnitude mode: acceleration PSD in dB rel. 1 (m/s²)²/Hz) and `noise_db` (relative mode). |

The default site values are literature-style placeholders. Replace `noise_psd_db` with
noise measured at the sites if you can: it has the largest effect on which events are
detected. For reference, Peterson's NLNM is about −168 dB and NHNM about −91 dB at 10 Hz.

### `design`

| Key | Default | Meaning |
|---|---|---|
| `measure` | `achievable` | `achievable`: error of a NonLinLoc-like location (the measure validated against NonLinLoc). `gls`: a lower bound from an ideal locator. |
| `criterion` | `rms` | `rms` (RMS 3D error), `percentile` (e.g. the worst 10 % of the target) or `logdet` (error-ellipsoid volume). |
| `vertical_weight` | `1.0` | Weight of depth error relative to horizontal error. |
| `fixed` | `[]` | Existing stations that are always included. |
| `exclude` | `[]` | Candidates that are never used. |
| `n_min` / `n_max` | 3 / all | Range of network sizes searched. |
| `n_restarts` | `5` | Extra exchange searches from random networks. More makes it likelier the search finds the true best network. |
| `n_random` | `200` | Random networks per size, for the baseline. |
| `target_m` | `null` | Recommend the smallest network whose criterion is at most this (m). |
| `target_horizontal_m`, `target_depth_m` | `null` | Recommend the smallest network meeting both RMS goals (m, over located events). |
| `min_gain_pct` | `5.0` | Without goals: stop at the size where one more station improves the criterion by less than this. |
| `locatability` | `true` | With detection, an event counts only if it gets at least `locate_min_picks` picks including `locate_min_p_picks` P picks. |
| `min_located_fraction` | `0.9` | Share of design events a network must locate to be recommended. |
| `min_separation_km` | `0.0` | No two chosen stations closer than this (horizontally). |
| `jobs` | CPUs | Threads for the search. |

### `validation`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Whether `netdesign run` includes validation. Needs `nll.keep_grids: base` or `all`, and `perturb.include_base: true`. |
| `n_events` | `200` | Synthetic events. |
| `stations` | all | The network to test, e.g. the recommended one. |
| `pick_noise` | `true` | Add Gaussian pick noise (`picks.sigma_s`). |
| `method` | `GAU_ANALYTIC` | NonLinLoc `LOCMETH`. |

### `qc`

QC is on by default. After every stage, netdesign writes checks and figures to
`<output_dir>/qc/` and rebuilds `report.html`. The thresholds (eikonal accuracy, validation
ratios, detection coverage and so on) are listed in `examples/config.yaml`.

---

## 7. Commands

Give the config file with `-c` before the command, or after the command:

```bash
netdesign -c config.yaml                # no command: the whole pipeline (same as `run`)
netdesign -c config.yaml design         # one stage
netdesign design config.yaml            # the same, config after the command
netdesign -c config.yaml -o other_out   # --out/-o overrides output_dir
```

| Command | What it does |
|---|---|
| `netdesign run config.yaml` | Runs the whole pipeline: `sensitivity`, then `validate` (if `validation.enabled`), then `design`. It checks first that the later stages can run, so a setting they can't use fails before the long sensitivity stage. It stops at the first stage that fails. |
| `netdesign sensitivity config.yaml` | Ensemble, grids, travel times and sensitivities at the target points. This is the long stage. |
| `netdesign validate config.yaml` | Locates synthetic events with NonLinLoc and compares them with the predictions. Run `sensitivity` first. |
| `netdesign design config.yaml` | Searches for the best networks and the recommended size. Run `sensitivity` first. Re-running `design` alone is quick, so you can change `design` and `detection` settings without repeating `sensitivity`. |
| `netdesign export config.yaml` | Writes shapefiles to `<output_dir>/gis/`. |
| `netdesign report config.yaml` | Rebuilds `<output_dir>/qc/report.html` from the existing QC files. |
| `netdesign sites config.yaml [--force]` | Generates the candidate grid with DEM elevations (see 5.2). |
| `netdesign ensemble` / `grids` / `traveltimes` | Build and keep only the intermediate stages, for inspection. |

**Which settings need `sensitivity` re-run?** Anything that changes travel times or target
points: `velocity`, `grid`, `target`, `stations`, `phases`. Changes to `design`,
`detection` and `picks` only need `design` re-run.

---

## 8. Recommended workflow

1. **Check the inputs.** Run `netdesign sensitivity` with a small ensemble
   (`n_realizations: 10`) and `grid.mode: 2d`, and look at the first report: the input map
   (candidates and target), the velocity profiles, and the eikonal check.
2. **Fast 2D design.** Run `netdesign run` with a layered model, `grid.mode: 2d` and
   validation on. Use the validation result to confirm that the predicted errors match
   NonLinLoc (see 9.3). Try the accuracy goals and detection settings by re-running
   `netdesign design` only.
3. **Final 3D design.** Switch to `grid.mode: 3d`, add `heterogeneity`, and use a larger
   ensemble (30 to 100). With many candidates, 3D grids for every station don't fit on
   disk, so set `nll.keep_grids: none` and `validation.enabled: false` for the design run.
4. **Validate the recommended network in 3D.** Make a second config that uses only the
   recommended stations (a candidates file with just those rows), with
   `keep_grids: base`, `validation.enabled: true` and a new `output_dir`. Then run
   `netdesign sensitivity` and `netdesign validate` on it.
5. **Evaluate a chosen network.** To score one specific network (for example the
   recommended stations plus a site you want for practical reasons), set
   `design.fixed` to those stations and `n_min` and `n_max` to their number, with
   `n_restarts: 0`.
6. **Export** with `netdesign export` for GIS.

---

## 9. Reading the results

### 9.1 Terminal summary

`design` prints one row per network size:

```
  n  rms (achievable, m)  horiz  depth  located  stations
  5      347.3    333    324      90%  C1524 C1655 C1943 C1988 C2237
  8*     ...
recommended: 8 stations: ...
```

- `*` marks the recommended size.
- `rms` is the design criterion (with `vertical_weight` applied).
- `horiz` and `depth` are the RMS horizontal and depth errors in metres, over located
  events.
- `located` is the share of design events that get enough picks to be located.

Warnings (in yellow) mean the recommendation is not reliable. The most common is that no
network locates `min_located_fraction` of the design events, because the design event is
too small for these candidates.

### 9.2 Files in `<output_dir>/`

| Path | Contents |
|---|---|
| `design/design.json` | Everything about the design: the recommended network, then for each size the stations, criterion, horizontal and depth errors, located fraction, greedy and random-baseline scores, the share of random networks that beat it, and the criterion if each station is lost. Also notes, warnings, and `recommendation_reliable`. |
| `design/networks.csv` | One row per size: `n,criterion_m,horizontal_m,depth_m,located_fraction,stations`. |
| `qc/report.html` | Self-contained report with every check and figure. Open this first. |
| `qc/candidates_outcome.csv` / `.geojson` | Every candidate with its `status` (`fixed`, `selected`, `candidate` or `excluded`), its rank in the recommended network (`selection_order`, by how much losing it would hurt), `greedy_order`, `gain_if_added_pct` for unused candidates, and its horizontal and depth contributions. |
| `gis/*.shp` | From `export`, in the input CRS: `candidates_input`, `target_boxes`, `candidates_outcome`, `network_final` (by rank) and `networks_all` (every size). |
| `validation/summary.json`, `events.csv` | Validation statistics and per-event mislocations. |
| `sensitivity/`, `models/`, `grids/`, `traveltimes/` | Intermediate results. |
| `stations_local.csv` | Candidates in the local km frame. |

### 9.3 The QC report

The report has a section for each stage, each with pass/warn checks:

- **Inputs:** a map of the candidates and target boxes.
- **Ensemble:** velocity profiles of sample realisations, and heterogeneity slices with the
  achieved versus requested sd and correlation lengths.
- **Sensitivity:** the eikonal check of the travel-time grids, uncertainty maps with all
  candidates at several depths, and each station's contribution.
- **Validation:** NonLinLoc versus predicted mislocations (scatter plot and vectors).
- **Design:** criterion versus network size against the random baseline, robustness to
  losing a station, the design map, uncertainty maps for the recommended network, and the
  detection probability by station and as a map of expected picks.

**Is the error model trustworthy?** Validation prints a table like:

```
axis  NLLoc rms  linear rms  predicted (achievable)  design   corr(NLL, lin)
   x     70.6 m      71.4 m            85.7 m     49.8 m       0.99
   z    188.2 m     178.1 m           181.1 m     95.5 m       0.98
within predicted 68% / 95% ellipsoids: 67% / 94%
```

It is good when:

- `NLLoc rms` is close to `predicted (achievable)` (the QC accepts a ratio of 0.7 to 1.4),
- `corr(NLL, lin)` is above 0.9, which means the linearised model reproduces NonLinLoc
  event by event,
- about 68 % and 95 % of events fall inside the predicted 68 % and 95 % ellipsoids.

The `design` column is the ideal-locator bound. It is normally smaller than the others,
because NonLinLoc doesn't know the model-error covariance.

**How sure is the search?** In `design.json`, `fraction_random_better` near 0 and a high
`restarts_reaching_best` mean the best network was found reliably.

---

## 10. Choosing the main settings

**Accuracy goals.** Set `target_horizontal_m` and `target_depth_m` to what the project
needs. The recommendation is then the smallest network meeting both. Set
`vertical_weight = (horizontal goal / depth goal)²` so the search balances the two goals
(for example 150 m and 500 m give 0.09). Without goals, the recommendation is the "knee",
where one more station improves the criterion by less than `min_gain_pct`.

**Design event.** In `magnitude` mode, `detection.magnitude` is the smallest event the
network must locate. If the result warns that the located fraction is too low, the event is
below what these sites can record. Raise the magnitude, use quieter (borehole) sites, or
add candidates closer to the target. `magnitude_sensitivity` shows how the recommended
network changes with the event size.

**Model uncertainty.** `vp_pct`, `vpvs` and `interface_sd_km` should reflect how well the
model is actually known. Overstating them makes networks look worse than they are, and
understating them makes networks look better. Correlated model error is what usually
limits absolute accuracy, so these settings matter as much as the station geometry.

**Pick accuracy.** `picks.sigma_s` is the pick standard deviation. Typical values are
5–10 ms for P and 10–20 ms for S on good microseismic data.

**Existing stations.** List them in `design.fixed`. They are always in the network, and the
search adds stations around them.

**Practical constraints.** Use `design.exclude` for sites you can't use and
`min_separation_km` to keep stations apart.

---

## 11. Run time and disk space

- `sensitivity` takes most of the time. It grows with the number of candidates,
  realisations and grid cells. 2D grids are cheap. 3D grids grow as `1/dx_km³`, so halving
  `dx_km` makes them about 8 times bigger and slower.
- Grids not kept are deleted as soon as they are used. When all grids don't fit,
  netdesign makes and deletes them in station batches that fit within `nll.max_disk_gb`
  (default: half the free space).
- `keep_grids: base` keeps the starting model's grids for every candidate. In 3D with
  hundreds of candidates, this can fill the disk. That's why the recommended 3D workflow
  validates only the recommended network (see section 8, step 4).
- `design` is fast once `sensitivity` is done. Each `magnitude_sensitivity` entry adds
  another design run, and so do `n_max`, `n_restarts` and `n_random`.
- Progress goes to stderr. Each stage's timings are written to the QC folder.

---

## 12. Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `error: config.yaml: <field>: ...` | A config value or key is invalid; the message names the field. Unknown keys are errors, so check the spelling against `examples/config.yaml`. |
| `velocity: give exactly one of 'base' (layered) or 'model_3d'` | Both or neither model is given. |
| `validation.enabled needs the starting model's grids` | Set `nll.keep_grids: base` (or `all`), or `validation.enabled: false`. |
| `validation.enabled needs the unperturbed base model` | Set `velocity.perturb.include_base: true`. |
| `config is missing section(s) [...]` | `run` needs `stations`, `grid` and `target`. |
| `detection.sites: unknown station(s)` | A station in `detection.sites` is not in the candidates file. |
| NonLinLoc programs not found | Set `nll.bin_dir`, or put `Vel2Grid`, `Grid2Time` and `NLLoc` on `PATH`. |
| WARNING: the recommended network locates only X% | The design event is too small for these sites (see section 10, Design event). |
| A station outside the model or grid | Raise the first layer's top above the highest station, and check `grid.padding_km` and `depth_max_km`. |
| Input files not found | Relative paths are relative to the config file, not the current directory. |
| Disk fills during 3D `sensitivity` | Use `keep_grids: none`, lower `nll.max_disk_gb`, or increase `grid.dx_km`. |
| Validation ratio outside 0.7–1.4, or low correlation | The linearisation is weak for this geometry (often with very few stations or poorly constrained depths). Trust the NonLinLoc numbers, and check `validation.locgrid_*` and `oct_*` resolution. |

---

## 13. Worked example: Mak-Ban

The Mak-Ban MEQ network design is in the repo:

- `config/config.yaml`: the final 3D design config.
- `mak-ban/candidates.csv`: candidate sites from the MEQ grid (200 m cells, noisy locations
  removed). The station name is `C` + the cell id, and `z` is the cell's elevation.
- `mak-ban/velocity.csv`: 1D model from a VELEST Monte Carlo inversion (Vp/Vs 1.728).
- `mak-ban/Mak-Ban MEQ Grid WGS84.csv`: the original MEQ grid.

The main settings:

- **Target:** two boxes from 1 to 4 km below sea level. The main box is 1 × 1 km, rotated
  45°, and gets weight 8. The secondary box is 0.5 × 0.5 km and gets weight 2.
- **Velocity:** 30 perturbed realisations (Vp 5 % per layer, Vp/Vs 0.03, interfaces
  50 m), plus 3D von Kármán heterogeneity (5 % Vp, correlation lengths 3 / 1 / 0.4 km,
  strike 45°).
- **Grid:** 3D, 100 m cells, down to 5 km.
- **Picks:** P 10 ms, S 20 ms.
- **Detection:** Mw 0.5 design event. Mw 0 is below what these sites can locate.
- **Goals:** 150 m horizontal and 500 m depth RMS, so `vertical_weight: 0.09`. Networks
  of up to 40 stations are searched, with 20 restarts.
- **Grids:** `keep_grids: none` and validation off for the design run. The recommended
  network is validated in a separate run (section 8, step 4).

Run it from the repo:

```bash
netdesign run config/config.yaml      # results in config/out_3d/
netdesign export config/config.yaml
```

Then validate the recommended network: make a candidates file with only the recommended
stations from `config/out_3d/design/design.json`, and a copy of the config that points to
it, with `output_dir: out_3d_validation`, `nll.keep_grids: base` and
`validation.enabled: true`. Run `netdesign sensitivity` and `netdesign validate` on that
copy.

Results are written under `config/`. Keep them out of git by adding `out_*/` to
`.gitignore`, or by pointing `output_dir` outside the repo.
