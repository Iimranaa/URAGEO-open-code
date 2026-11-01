# -*- coding: utf-8 -*-
"""Shared module of the A3 calculation scripts (statistics and geostatistics).
Reads the borehole table (synthetic by default; set URAGEO_DATA_CSV to use another file with the same columns)
and writes results and figures to outputs/A3 (or URAGEO_OUTPUT).
Figure labels and comments are in French, as in the thesis.
Python 3: numpy, pandas, scipy, matplotlib, geopandas, rasterio, shapely, scikit-learn."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
# Input table: synthetic by default. To run on your own data (same columns), set URAGEO_DATA_CSV.
CSV_DONNEES = Path(os.environ.get('URAGEO_DATA_CSV', REPO / 'data_synthetic' / 'tarat_boreholes_synthetic.csv'))
STRATI_CSV = Path(os.environ.get('URAGEO_STRATI_CSV', REPO / 'data_synthetic' / 'strati_units_synthetic.csv'))
DOSSIER = Path(os.environ.get('URAGEO_OUTPUT', REPO / 'outputs' / 'A3'))
OUT = str(DOSSIER) + os.sep
FIG = OUT + 'FIG' + os.sep
os.makedirs(OUT + 'res', exist_ok=True)
NAVY='#1F3864';CU='#B8742A';TEAL='#2A7F8F';GREY='#6B6B6B';RED='#B03A2E';GRN='#3E7C4F';SAND='#F6F4EF'
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
 'figure.dpi':100,'savefig.dpi':200,'axes.edgecolor':'#444','axes.grid':False,'font.family':'DejaVu Sans'})
def save(fig, folder, name):
    os.makedirs(FIG + folder, exist_ok=True)
    p = FIG + folder + '/' + name + '.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white'); plt.close(fig); return p
def load():
    """Borehole table with accumulation indices, thickness, roof depth and PCA scores."""
    return pd.read_csv(CSV_DONNEES)
LAB={'acc_oxy':'Oxy-hydroxydes','acc_sulf':'Sulfures','acc_mo':'Matière organique','acc_carb':'Carbonates','puiss_m':'Épaisseur Tarat (m)','toit_m':'Profondeur du toit (m)','x':'X (m)','y':'Y (m)'}
NOTE="Source : jeu de sondages (synthétique par défaut)."
RES={}
def savejson(name):
    json.dump(RES, open(OUT + 'res/' + name, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
