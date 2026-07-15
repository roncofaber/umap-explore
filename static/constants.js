// ── Slider step values ────────────────────────────────────────────────────────
export const N_NEIGHBORS_STEPS = [5, 10, 15, 20, 30, 50, 100];
export const MIN_DIST_STEPS    = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0];
export const MCS_STEPS         = [5, 10, 15, 20, 30, 50];
export const MS_STEPS          = [1, 3, 5, 10, 15, 20];
export const CSE_STEPS         = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]; // cluster_selection_epsilon
export const PERPLEXITY_STEPS  = [5, 15, 30, 50, 100];                  // t-SNE perplexity

// ── Plotly layout constants ───────────────────────────────────────────────────
// l+r = t+b = 120 → equal margins keep the axes box square
export const MARGIN = { t: 30, r: 70, b: 110, l: 70 }; // l+r = t+b = 140 → square axes box

// ── Colorscale: CET_L20 (perceptually uniform, dark → bright yellow) ─────────
export const COLORSCALE = [
  [0.0000, '#303030'],
  [0.0667, '#3c365b'],
  [0.1333, '#423d84'],
  [0.2000, '#4247a7'],
  [0.2667, '#3d54c1'],
  [0.3333, '#3466c7'],
  [0.4000, '#227ea9'],
  [0.4667, '#2f9287'],
  [0.5333, '#55a066'],
  [0.6000, '#84a946'],
  [0.6667, '#aeb127'],
  [0.7333, '#dab512'],
  [0.8000, '#f5be14'],
  [0.8667, '#fdcf12'],
  [0.9333, '#fee30f'],
  [1.0000, '#f8f809'],
];

// ── Scatter animation ─────────────────────────────────────────────────────────
export const ANIM_DURATION = 600; // ms

// ── Horizontal legend — cap entries so it never exceeds ~4 rows ───────────────
const LEGEND_MAX_ROWS        = 4;
const LEGEND_ENTRIES_PER_ROW = 7;   // approximate for a 700–900 px plot width
export const LEGEND_MAX_ENTRIES = LEGEND_MAX_ROWS * LEGEND_ENTRIES_PER_ROW; // 28

export const AXIS_LABEL_FONT = {
  family: "'JetBrains Mono', monospace",
  size: 14,
  color: '#515978',
};

export const TICK_FONT = {
  family: "'JetBrains Mono', monospace",
  size: 13,
  color: '#8a94b2',
};

export const AXIS_BOX = {
  showline: true, linecolor: '#000', linewidth: 1.5, mirror: true,
  showgrid: false, zeroline: false,
  ticks: 'outside', ticklen: 4, tickwidth: 1, tickcolor: '#c0c8d8',
  tickfont: TICK_FONT, nticks: 5, tickformat: '.1f',
};

// ── Code-generation templates per dataset ────────────────────────────────────
export const DATASET_CODE = {
  iris: {
    imports: 'from sklearn.datasets import load_iris',
    load: 'data = load_iris()\nX = data.data',
  },
  digits: {
    imports: 'from sklearn.datasets import load_digits',
    load: 'data = load_digits()\nX = data.data',
  },
  swiss_roll: {
    imports: 'from sklearn.datasets import make_swiss_roll',
    load: 'X, _ = make_swiss_roll(n_samples=2000, random_state=42)',
  },
  breast_cancer: {
    imports: 'from sklearn.datasets import load_breast_cancer',
    load: 'data = load_breast_cancer()\nX = data.data',
  },
  penguins: {
    imports: "import pandas as pd",
    load: "df = pd.read_csv('penguins.csv').dropna()\nfeatures = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']\nX = df[features].values",
  },
  olivetti_faces: {
    imports: 'from sklearn.datasets import fetch_olivetti_faces',
    load: 'data = fetch_olivetti_faces(shuffle=True, random_state=42)\nX = data.data',
  },
  pbmc3k: {
    imports: 'import scanpy as sc',
    load: "adata = sc.datasets.pbmc3k_processed()\nX = adata.obsm['X_pca']  # 50 PCA components",
  },
};
