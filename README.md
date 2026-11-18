# URAGEO open code

Reference scripts and **synthetic** data accompanying four manuscripts on uranium favourability modelling and the Tarat Formation
(Tim Mersoï Basin, Niger):

| Folder | Article (working titles) | What is here |
|---|---|---|
| `A1_URAGEO_favourability` | A1 · URAGÉO weighted-overlay favourability model | AHP weights and consistency, Monte-Carlo stability of the judgements, reclassification into scores 0–4, weighted overlay, top 10 %, AUC, ablation, variants V14/V15, share of pits in the top 10 %/20 % against a rigid-shift null model |
| `A2_structural_null_model` | A2 · Fault directions and nodes as controls | Rigid-shift null model for the association between fault families / nodes and mining units, Holm correction |
| `A3_tarat_geostatistics` | A3 · Geostatistics of 731 Tarat boreholes | Statistics (S1–S5, S6 reduced to a WoE/ROC worked example) and geostatistics (G1–G7): descriptive statistics, correlations, PCA, bootstrap, AHP, Moran, Gi*, variograms, kriging, leave-one-out and block cross-validation, unit thicknesses (RMSE/MAE), extra analyses |
| `A4_transfer_six_sites` | A4 · Transferability to six sites | AUC per site, site bootstrap, placement null model, Holm correction; **`regional_grids/`: the three real regional index grids of A4 (derived from published maps) and the script that rebuilds them** |

## Model version

The scripts implement **version v37** of the URAGÉO model, the one described in the articles:

* seven layers scored from 0 to 4 (Table 1 of A1); the host layer F4 is a **binary condition** (4 where the host or a younger unit is mapped, 0 below the host);
* AHP weights (geometric mean): F4 20.7 %, N110–150 32.2 %, F3 (fault nodes) 20.7 %, N70 11.2 %, F1 (granites) 6.1 %, N–S 6.1 %, N30 2.9 %; λmax = 7.115, CI = 0.019, CR = 0.015;
* index I = Σ wᵢ·Rcᵢ / Σ wᵢ (0 to 4), five classes of width 0.8 and a top 10 % defined by the 90th percentile.

The previous weighting (28.5, 28.5, 18.3, 10.4, 5.8, 5.8, 2.8 %) is kept only as variant V14. The Monte-Carlo perturbation of the judgements
(5,000 draws, seed 7) reproduces the 95 % intervals reported in A1 (F4 and F3: 16–27 %; N110–150: 26–37 %); the share of draws in which N110–150 keeps
the largest weight may differ by a few tenths of a percent from the published value because of the random generator.

## Read this first: what is and is not reproducible

* **The real data are not included.** The borehole logs, pit outlines and site positions belong to TSUMCO (formerly SOMAÏR) and are
  confidential. Everything in `data_synthetic/` is randomly generated (fixed seeds) and carries **no real value**.
* **Results on the synthetic data are not the results of the articles.** They show that the code runs end to end and what each
  step computes. A planted spatial structure makes some tests detect something and others not; do not interpret them geologically.
* **To reproduce the articles with the real data** you need the original files and the written agreement of the data owner. Point the scripts to
  a table with the same columns (see `data_synthetic/tarat_boreholes_synthetic.csv`) with the environment variable `URAGEO_DATA_CSV`
  (and `URAGEO_STRATI_CSV` for the unit intervals, `URAGEO_GIS_DIR` for the rasters of A1, A2, A4).
* **Exception: the regional grids of A4** (`A4_transfer_six_sites/regional_grids/`) are real derived data. They come only from published maps (SIGU fault base, geological maps of the Aïr / Tim Mersoï area), contain no TSUMCO data and are the grids on which Tables 3 and 4 of A4 were computed (added in version 1.1.0).
* **Construction of the real factor rasters** (distance to fault families, nodes, host continuity, granite proximity and their classes) was done in
  ArcMap/QGIS and is described in the thesis and in the ArcMap guides; it is not scripted here. In A1, A2 and A4 the factors are simple reclassifications of synthetic
  distance rasters, so that only the statistical machinery is reproduced.
* The original S6 script also re-plotted site-specific results (AUC per named site, success rates, sensitivity scenarios) read from project rasters. Those values are not published here; the corresponding tests for A1, A2 and A4 are re-implemented on synthetic layers.
* Figure labels and comments in the A3 scripts are in French (as in the thesis).

## Quick start

```bash
pip install -r requirements.txt
python run_all.py --quick      # A1, A2, A4 and S1-S6 (under a minute)
python run_all.py              # everything, including kriging and cross-validation (about 2 minutes)
```

`run_all.py` creates the synthetic data on first use (`common/synthetic_boreholes.py`, `common/synthetic_gis.py`). Outputs (JSON, CSV, PNG, GeoTIFF) are written to `outputs/`.
Each script has a header explaining its purpose, inputs and outputs, and can be run alone, for example
`python A2_structural_null_model/rigid_shift_null_model.py`.

## Data format

`data_synthetic/tarat_boreholes_synthetic.csv`: one row per borehole. Columns used by the scripts: `hole_id, x, y` (local metric coordinates), `puiss_m` (thickness, m),
`toit_m` (depth to top, m), `acc_oxy, acc_sulf, acc_mo` (semi-quantitative accumulation indices), `acc_carb` (only 31 boreholes), `PC1_calc, PC2_calc, PC3_calc`.
`data_synthetic/strati_units_synthetic.csv`: `hole_id, code, from_m, to_m` (codes 310 = U1, 320 = U2, 330 = U3, 340/344 = U4).
`data_synthetic/gis_synthetic.npz`: distance rasters (m) on a 100 m local grid (260 × 200 cells). `sites_synthetic.csv`: pits, workings and external sites as discs (x, y, r).

## Tests performed

All scripts were run successfully on the synthetic data with Python 3.13 (numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11, rasterio 1.5, geopandas 1.2, shapely 2.2, scikit-learn 1.9).
They have **not** been validated on other operating systems.

## Licences

Code: MIT (`LICENSE`). Synthetic data: CC0 (`data_synthetic/LICENSE-DATA.txt`). Regional grids of A4: CC BY 4.0.

## How to cite

See `CITATION.cff`. All versions: DOI 10.5281/zenodo.24323618 (concept DOI); version 1.0.0: 10.5281/zenodo.24323619.
