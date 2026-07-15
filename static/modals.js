import { state, datasetInfo, cachedData } from './state.js';
import { els } from './elements.js';
import { DATASET_CODE } from './constants.js';

// ── Code modal ────────────────────────────────────────────────────────────────
function generateCode() {
  const ds = DATASET_CODE[state.dataset];
  if (!ds) return '';
  const lines = [];
  const method = state.method;
  const isHdbscan = state.tab === 'hdbscan';

  // Imports
  if (method === 'umap') lines.push('import umap');
  if (method === 'tsne') lines.push('from sklearn.manifold import TSNE');
  if (method === 'pca')  lines.push('from sklearn.decomposition import PCA');
  if (isHdbscan)         lines.push('import hdbscan');
  lines.push(ds.imports);
  if (state.scale === 'scaled') lines.push('from sklearn.preprocessing import StandardScaler');
  lines.push('');

  // Load
  lines.push('# Load data');
  lines.push(ds.load);

  if (state.scale === 'scaled') {
    lines.push('');
    lines.push('# Normalize features');
    lines.push('X = StandardScaler().fit_transform(X)');
  }

  // Embedding
  lines.push('');
  if (method === 'umap') {
    lines.push('# Reduce dimensions with UMAP');
    lines.push('reducer = umap.UMAP(');
    lines.push(`    n_neighbors=${state.nNeighbors},`);
    lines.push(`    min_dist=${state.minDist},`);
    lines.push(`    n_components=2,`);
    lines.push(`    metric='${state.metric}',`);
    lines.push(')');
    lines.push('embedding = reducer.fit_transform(X)');
  } else if (method === 'tsne') {
    lines.push('# Reduce dimensions with t-SNE');
    lines.push('reducer = TSNE(');
    lines.push(`    perplexity=${state.perplexity},`);
    lines.push(`    metric='${state.metric}',`);
    lines.push(`    n_components=2,`);
    lines.push(`    learning_rate='auto',`);
    lines.push(')');
    lines.push('embedding = reducer.fit_transform(X)');
  } else {
    const PC = ['PC1', 'PC2', 'PC3'];
    lines.push('# Reduce dimensions with PCA');
    lines.push('reducer = PCA(n_components=3)');
    lines.push('coords = reducer.fit_transform(X)');
    lines.push(`embedding = coords[:, [${state.pcX}, ${state.pcY}]]  # ${PC[state.pcX]} vs ${PC[state.pcY]}`);
  }

  // HDBSCAN clustering
  if (isHdbscan) {
    lines.push('');
    lines.push('# Cluster the embedding with HDBSCAN');
    lines.push('clusterer = hdbscan.HDBSCAN(');
    lines.push(`    min_cluster_size=${state.minClusterSize},`);
    lines.push(`    min_samples=${state.minSamples},`);
    lines.push(`    cluster_selection_method='${state.clusterSelectionMethod}',`);
    if (state.clusterSelectionEpsilon > 0)
      lines.push(`    cluster_selection_epsilon=${state.clusterSelectionEpsilon},`);
    if (state.allowSingleCluster)
      lines.push(`    allow_single_cluster=True,`);
    lines.push(')');
    lines.push('labels = clusterer.fit_predict(embedding)');
    lines.push('# labels == -1 → noise; labels >= 0 → cluster id');
  }

  return lines.join('\n');
}

export function initCodeModal() {
  const modal      = document.getElementById('code-modal');
  const codeBlock  = document.getElementById('code-block');
  const copyBtn    = document.getElementById('copy-code-btn');
  const closeBtn   = document.getElementById('close-code-btn');

  els.showCodeBtn.addEventListener('click', () => {
    codeBlock.textContent = generateCode();
    if (window.Prism) Prism.highlightElement(codeBlock);
    modal.hidden = false;
  });

  closeBtn.addEventListener('click', () => { modal.hidden = true; });
  modal.addEventListener('click', e => { if (e.target === modal) modal.hidden = true; });

  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(codeBlock.textContent).then(() => {
      const orig = copyBtn.textContent;
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = orig; }, 1500);
    }).catch(() => {});
  });

  return { modal };
}

// ── Data modal ────────────────────────────────────────────────────────────────
const MAX_ROWS = 200;

function fmt(v) {
  if (typeof v !== 'number') return v;
  return Math.abs(v) < 1000 ? v.toFixed(3) : v.toExponential(2);
}

function renderDataTable(data, tableEl, footerEl) {
  const { X, feature_names, labels, label_names } = data;
  const n    = X.length;
  const cols = feature_names || X[0].map((_, i) => `f${i}`);
  const slice = X.slice(0, MAX_ROWS);

  const headerRow = `<tr>
    <th>#</th>
    <th>${label_names ? 'class' : 'value'}</th>
    ${cols.map(c => `<th>${c}</th>`).join('')}
  </tr>`;

  const rows = slice.map((row, i) => {
    const label = label_names ? label_names[labels[i]] : fmt(labels[i]);
    const cells = row.map(v => `<td>${fmt(v)}</td>`).join('');
    return `<tr><td>${i + 1}</td><td>${label}</td>${cells}</tr>`;
  }).join('');

  tableEl.innerHTML  = `<thead>${headerRow}</thead><tbody>${rows}</tbody>`;
  footerEl.textContent = n > MAX_ROWS ? `showing ${MAX_ROWS} of ${n} rows` : '';
}

export function initDataModal() {
  const modal       = document.getElementById('data-modal');
  const titleEl     = document.getElementById('data-modal-title');
  const tableEl     = document.getElementById('data-table');
  const footerEl    = document.getElementById('data-footer');
  const closeBtn    = document.getElementById('close-data-btn');
  const rawBtn      = document.getElementById('data-scale-raw');
  const scaledBtn   = document.getElementById('data-scale-scaled');

  let dataScale = 'raw';

  function closeModal() {
    modal.hidden        = true;
    tableEl.innerHTML   = '';
    footerEl.textContent = '';
  }

  async function loadAndShow(scale) {
    const ds = state.dataset;
    if (!ds) return;
    titleEl.textContent = datasetInfo[ds]?.label || ds;

    const key = `${ds}_${scale}`;
    if (cachedData[key]) { renderDataTable(cachedData[key], tableEl, footerEl); return; }

    tableEl.innerHTML = '<caption style="color:rgba(255,255,255,0.4);padding:2rem">Loading…</caption>';
    const resp = await fetch(`/api/data/${ds}?scale=${scale}`);
    if (!resp.ok) {
      tableEl.innerHTML = `<caption style="color:#f87171;padding:2rem">${await resp.text()}</caption>`;
      return;
    }
    cachedData[key] = await resp.json();
    renderDataTable(cachedData[key], tableEl, footerEl);
  }

  els.viewDataBtn.addEventListener('click', () => {
    modal.hidden = false;
    loadAndShow(dataScale);
  });

  closeBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });

  rawBtn.addEventListener('click', () => {
    dataScale = 'raw';
    rawBtn.classList.add('active'); scaledBtn.classList.remove('active');
    loadAndShow('raw');
  });

  scaledBtn.addEventListener('click', () => {
    dataScale = 'scaled';
    scaledBtn.classList.add('active'); rawBtn.classList.remove('active');
    loadAndShow('scaled');
  });

  return { closeModal };
}
