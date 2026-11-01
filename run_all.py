# -*- coding: utf-8 -*-
"""Generate the synthetic data (if missing) and run every script of the four articles. Outputs go to outputs/.
Usage:  python run_all.py            (about 2 minutes)
        python run_all.py --quick    (A1, A2, A4 and the statistical scripts S1-S6 only)"""
import subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
quick = '--quick' in sys.argv
if not (R / 'data_synthetic' / 'tarat_boreholes_synthetic.csv').exists():
    subprocess.check_call([sys.executable, str(R / 'common' / 'synthetic_boreholes.py')])
if not (R / 'data_synthetic' / 'gis_synthetic.npz').exists():
    subprocess.check_call([sys.executable, str(R / 'common' / 'synthetic_gis.py')])
jobs = [('A1', R / 'A1_URAGEO_favourability' / 'ahp_overlay_ablation.py'),
        ('A2', R / 'A2_structural_null_model' / 'rigid_shift_null_model.py'),
        ('A4', R / 'A4_transfer_six_sites' / 'transfer_auc_and_placement_null.py')]
sc = R / 'A3_tarat_geostatistics' / 'scripts'
names = ['S1_descriptif', 'S2_correlations', 'S3_acp', 'S4_bootstrap', 'S5_ahp', 'S6_woe_roc_example']
if not quick: names += ['G1_echantillonnage', 'G2_moran', 'G3_gi_lisa', 'G4_variogrammes', 'G5_validation_croisee', 'G6_krigeage', 'G7_unites_tarat_cycles', 'A3_extra_blockcv_spacing_detrend']
jobs += [('A3', sc / (n + '.py')) for n in names]
failed = []
for tag, p in jobs:
    t = time.time(); r = subprocess.run([sys.executable, str(p)], cwd=str(p.parent), capture_output=True, text=True)
    print(f'[{tag}] {p.name:45s} {"OK" if r.returncode == 0 else "FAILED"}  ({time.time()-t:.0f} s)')
    if r.returncode: failed.append(p.name); print(r.stderr[-800:])
sys.exit(1 if failed else 0)
