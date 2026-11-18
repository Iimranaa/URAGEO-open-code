# Regional index grids of article A4 (real data, derived from published maps)

Three GeoTIFFs, EPSG:30732 (WGS 84 / UTM 32N), 100 m cells, window 240–370 km E × 1 920–2 090 km N (1 300 × 1 700 cells), float32, no-data −9999:

| File | Content (A4, Table 2) |
|---|---|
| `A4_full_index_binary_host_revised_weights_100m.tif` | Full index: seven layers, binary host rule on the regional geological map, revised AHP weights (0–4) |
| `A4_previous_full_index_100m.tif` | Previous full index: host by distance to the Carboniferous outcrop, previous weights |
| `A4_structural_subindex_100m.tif` | Structural sub-index: five fault-based layers, revised weights renormalised |

These are the grids on which Tables 3 and 4 of A4 were computed (AUC per site, placement null). They are derived only from published
sources: the regional fault base (SIGU) and geological maps of the Aïr / Tim Mersoï area. They contain **no TSUMCO data**; the site footprints
used in A4 (TSUMCO pits, COMINAK workings) are not distributed.

`build_regional_grids.py` rebuilds the three grids from those maps (paths given by environment variables; see the header). Expected checks:
2 028 faults ≥ 500 m in the window + 20 km, 856 merged intersection nodes, host class 0 on 5.1 % of the window; granite layer constant (0) in the window.

Licence of the grids: CC BY 4.0. Cite the article A4 and this repository.
