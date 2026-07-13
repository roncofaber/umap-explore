const N_NEIGHBORS_STEPS = [5, 10, 15, 20, 30, 50, 100];
const MIN_DIST_STEPS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0];

const state = {
  dataset: null,
  method: 'umap',
  nNeighbors: 15,
  minDist: 0.1,
  metric: 'euclidean',
  scale: 'scaled',
  isFirstRender: true,
};

const els = {
  datasetSelect: document.getElementById('dataset-select'),
  nnSlider:      document.getElementById('n-neighbors-slider'),
  nnValue:       document.getElementById('n-neighbors-value'),
  mdSlider:      document.getElementById('min-dist-slider'),
  mdValue:       document.getElementById('min-dist-value'),
  metricSelect:  document.getElementById('metric-select'),
  methodUmap:    document.getElementById('method-umap'),
  methodPca:     document.getElementById('method-pca'),
  umapParams:    document.getElementById('umap-params'),
  paramStatus:     document.getElementById('param-status'),
  sidebarToggle:   document.getElementById('sidebar-toggle'),
  sidebar:         document.getElementById('controls'),
  datasetInfo:   document.getElementById('dataset-info'),
  datasetStats:  document.getElementById('dataset-stats'),
  datasetDesc:   document.getElementById('dataset-desc'),
  scaleOn:       document.getElementById('scale-on'),
  scaleOff:      document.getElementById('scale-off'),
  plot:          document.getElementById('plot'),
  legend:        document.getElementById('legend'),
  loading:       document.getElementById('loading'),
};

// ── Axis range ────────────────────────────────────────────────────────────────

function axisRange(arr) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < mn) mn = arr[i];
    if (arr[i] > mx) mx = arr[i];
  }
  const pad = (mx - mn) * 0.06;
  return [mn - pad, mx + pad];
}

// ── Fetch ─────────────────────────────────────────────────────────────────────

async function fetchEmbedding() {
  const params = new URLSearchParams({
    method:       state.method,
    n_neighbors:  state.nNeighbors,
    min_dist:     state.minDist,
    n_components: 2,
    metric:       state.metric,
    scale:        state.scale,
  });
  const resp = await fetch(`/api/embeddings/${state.dataset}?${params}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

// ── Dataset metadata (label colors keyed by dataset name) ────────────────────

const datasetInfo = {};

// ── Sidebar collapse ──────────────────────────────────────────────────────────

function updateToggleLabel() {
  els.sidebarToggle.textContent = els.sidebar.classList.contains('collapsed') ? '›' : '‹';
}

els.sidebarToggle.addEventListener('click', () => {
  const collapsed = els.sidebar.classList.toggle('collapsed');
  els.sidebarToggle.classList.toggle('collapsed', collapsed);
  updateToggleLabel();
  requestAnimationFrame(() => { Plotly.relayout(els.plot, {}); repositionLegend(); });
});

updateToggleLabel();

// ── Slider tick positioning ───────────────────────────────────────────────────
// CSS flex/grid can't perfectly center variable-width labels at thumb positions,
// so we measure the rendered slider width and place each span absolutely.

function positionTicks(slider) {
  const container = slider.nextElementSibling;
  if (!container || !container.classList.contains('slider-ticks')) return;
  const w = slider.getBoundingClientRect().width;
  if (!w) return;
  const steps = parseInt(slider.max) - parseInt(slider.min);
  const halfThumb = 6.5;
  container.querySelectorAll('span').forEach((span, i) => {
    span.style.left = (halfThumb + i * (w - 2 * halfThumb) / steps) + 'px';
  });
}

function positionAllTicks() {
  positionTicks(els.nnSlider);
  positionTicks(els.mdSlider);
}

// ── Param status bar ─────────────────────────────────────────────────────────

function updateParamStatus() {
  if (!els.paramStatus) return;
  const scaleLabel = state.scale === 'scaled' ? 'scaled' : 'raw';
  if (state.method === 'pca') {
    els.paramStatus.textContent = `PCA  ·  ${scaleLabel}`;
  } else {
    els.paramStatus.textContent =
      `n_neighbors=${state.nNeighbors}  ·  min_dist=${state.minDist}  ·  metric=${state.metric}  ·  ${scaleLabel}`;
  }
}

// ── Dataset info card ─────────────────────────────────────────────────────────

function updateDatasetInfo() {
  const ds = datasetInfo[state.dataset];
  if (!ds) { els.datasetInfo.hidden = true; return; }

  const nClasses = ds.label_colors ? ds.label_colors.length : '—';
  const classLabel = ds.has_labels ? `${nClasses} class${nClasses !== 1 ? 'es' : ''}` : 'continuous';
  els.datasetStats.innerHTML =
    `${ds.n_points} points<span class="stat-sep">·</span>${ds.n_features} features<span class="stat-sep">·</span>${classLabel}`;
  els.datasetDesc.textContent = ds.description || '';
  els.datasetInfo.hidden = false;
}

// ── Legend ────────────────────────────────────────────────────────────────────

function repositionLegend() {
  if (!els.legend || !els.legend.children.length) return;
  // .nsewdrag is Plotly's drag rect that exactly covers the axes box
  const drag = els.plot.querySelector('.nsewdrag');
  if (!drag) return;
  const wrapper = els.plot.parentElement;          // #plot-wrapper, position:relative
  const wRect   = wrapper.getBoundingClientRect();
  const dRect   = drag.getBoundingClientRect();
  els.legend.style.top   = Math.round(dRect.top   - wRect.top   + 8) + 'px';
  els.legend.style.right = Math.round(wRect.right  - dRect.right + 8) + 'px';
}

function updateLegend(emb) {
  if (emb.label_names === null) { els.legend.innerHTML = ''; return; }
  const palette = datasetInfo[state.dataset]?.label_colors || [];
  els.legend.innerHTML = emb.label_names.map((name, i) =>
    `<div class="legend-item">
       <span class="legend-dot" style="background:${palette[i] || '#888'}"></span>
       <span class="legend-label">${name}</span>
     </div>`
  ).join('');
}

// ── Plotly trace / layout ─────────────────────────────────────────────────────

function makeTrace(emb) {
  const isContinuous = emb.label_names === null;
  const hoverText = isContinuous
    ? emb.labels.map(v => `value: ${v.toFixed(2)}`)
    : emb.labels.map(l => emb.label_names[l]);

  const marker = { size: 5, opacity: 0.8 };

  if (isContinuous) {
    marker.color = emb.labels;
    marker.colorscale = 'Viridis';
    marker.showscale = true;
  } else {
    const palette = datasetInfo[state.dataset]?.label_colors;
    marker.color = palette
      ? emb.labels.map(l => palette[l])
      : emb.labels.map(l => l / Math.max(emb.label_names.length - 1, 1));
    if (!palette) { marker.colorscale = 'Turbo'; marker.showscale = false; }
  }

  return {
    type: 'scattergl', mode: 'markers',
    x: emb.x, y: emb.y,
    text: hoverText,
    hovertemplate: '%{text}<extra></extra>',
    marker,
  };
}

const AXIS_LABEL_FONT = { family: "'JetBrains Mono', monospace", size: 14, color: '#515978' };
const TICK_FONT      = { family: "'JetBrains Mono', monospace", size: 10, color: '#8a94b2' };
const AXIS_BOX = {
  showline: true, linecolor: '#000', linewidth: 1.5, mirror: true,
  showgrid: false, zeroline: false,
  ticks: 'outside', ticklen: 4, tickwidth: 1, tickcolor: '#c0c8d8',
  tickfont: TICK_FONT, nticks: 5, tickformat: '.1f',
};

function makeLayout(emb) {
  return {
    margin: { t: 30, r: 40, b: 65, l: 70 },
    paper_bgcolor: '#eef0f5',
    plot_bgcolor: '#eef0f5',
    showlegend: false,
    xaxis: {
      ...AXIS_BOX,
      range: axisRange(emb.x),
      title: { text: 'coord 1', font: AXIS_LABEL_FONT, standoff: 6 },
    },
    yaxis: {
      ...AXIS_BOX,
      range: axisRange(emb.y),
      scaleanchor: 'x',
      scaleratio: 1,
      title: { text: 'coord 2', font: AXIS_LABEL_FONT, standoff: 6 },
    },
  };
}

// ── Custom animation ──────────────────────────────────────────────────────────

let currentEmb = null;
let animFrame  = null;

function cubicInOut(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function interpolateEmb(from, to, e) {
  const n = to.x.length;
  const x = new Array(n);
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = from.x[i] + (to.x[i] - from.x[i]) * e;
    y[i] = from.y[i] + (to.y[i] - from.y[i]) * e;
  }
  return { ...to, x, y };
}

function renderPlot(emb) {
  if (state.isFirstRender || !currentEmb || currentEmb.x.length !== emb.x.length) {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    Plotly.react(els.plot, [makeTrace(emb)], makeLayout(emb), { responsive: true });
    state.isFirstRender = false;
    currentEmb = emb;
    return;
  }

  const from  = currentEmb;
  const start = performance.now();
  const DURATION = 600;

  if (animFrame) cancelAnimationFrame(animFrame);

  (function tick() {
    const t      = Math.min((performance.now() - start) / DURATION, 1);
    const interp = interpolateEmb(from, emb, cubicInOut(t));
    currentEmb   = interp;
    Plotly.react(els.plot, [makeTrace(interp)], makeLayout(interp));
    animFrame = (t < 1) ? requestAnimationFrame(tick) : null;
    if (t >= 1) currentEmb = emb;
  })();
}

// ── Fetch + render ────────────────────────────────────────────────────────────

async function fetchAndRender() {
  els.loading.style.display = 'block';
  updateParamStatus();
  try {
    const emb = await fetchEmbedding();
    renderPlot(emb);
    updateLegend(emb);
    requestAnimationFrame(repositionLegend);
  } catch (e) {
    console.error('Failed to load embedding:', e);
  } finally {
    els.loading.style.display = 'none';
  }
}

let renderTimer = null;
function scheduleRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(fetchAndRender, 300);
}

// ── Event listeners ───────────────────────────────────────────────────────────

els.nnSlider.addEventListener('input', () => {
  state.nNeighbors = N_NEIGHBORS_STEPS[parseInt(els.nnSlider.value)];
  els.nnValue.textContent = state.nNeighbors;
  scheduleRender();
});

els.mdSlider.addEventListener('input', () => {
  state.minDist = MIN_DIST_STEPS[parseInt(els.mdSlider.value)];
  els.mdValue.textContent = state.minDist;
  scheduleRender();
});

els.metricSelect.addEventListener('change', () => {
  state.metric = els.metricSelect.value;
  fetchAndRender();
});

els.methodUmap.addEventListener('click', () => {
  if (state.method === 'umap') return;
  state.method = 'umap';
  els.methodUmap.classList.add('active');
  els.methodPca.classList.remove('active');
  els.umapParams.classList.remove('params-disabled');
  fetchAndRender();
});

els.methodPca.addEventListener('click', () => {
  if (state.method === 'pca') return;
  state.method = 'pca';
  els.methodPca.classList.add('active');
  els.methodUmap.classList.remove('active');
  els.umapParams.classList.add('params-disabled');
  fetchAndRender();
});

els.scaleOn.addEventListener('click', () => {
  if (state.scale === 'scaled') return;
  state.scale = 'scaled';
  els.scaleOn.classList.add('active');
  els.scaleOff.classList.remove('active');
  fetchAndRender();
});

els.scaleOff.addEventListener('click', () => {
  if (state.scale === 'raw') return;
  state.scale = 'raw';
  els.scaleOff.classList.add('active');
  els.scaleOn.classList.remove('active');
  fetchAndRender();
});

els.datasetSelect.addEventListener('change', () => {
  state.dataset = els.datasetSelect.value;
  state.isFirstRender = true;
  updateDatasetInfo();
  fetchAndRender();
});

// ── Show code modal ───────────────────────────────────────────────────────────

const DATASET_CODE = {
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
};

function generateCode() {
  const ds = DATASET_CODE[state.dataset];
  if (!ds) return '';
  const lines = [];

  if (state.method === 'umap') lines.push('import umap');
  if (state.method === 'pca')  lines.push('from sklearn.decomposition import PCA');
  lines.push(ds.imports);
  if (state.scale === 'scaled') lines.push('from sklearn.preprocessing import StandardScaler');
  lines.push('');

  lines.push('# Load data');
  lines.push(ds.load);

  if (state.scale === 'scaled') {
    lines.push('');
    lines.push('# Normalize features');
    lines.push('X = StandardScaler().fit_transform(X)');
  }

  lines.push('');
  if (state.method === 'umap') {
    lines.push('# Fit UMAP');
    lines.push('reducer = umap.UMAP(');
    lines.push(`    n_neighbors=${state.nNeighbors},`);
    lines.push(`    min_dist=${state.minDist},`);
    lines.push(`    n_components=2,`);
    lines.push(`    metric='${state.metric}',`);
    lines.push(')');
  } else {
    lines.push('# Fit PCA');
    lines.push('reducer = PCA(n_components=2)');
  }
  lines.push('embedding = reducer.fit_transform(X)');

  return lines.join('\n');
}

// ── Data table modal ──────────────────────────────────────────────────────────

const dataModal      = document.getElementById('data-modal');
const dataTable      = document.getElementById('data-table');
const dataModalTitle = document.getElementById('data-modal-title');
const dataBtnRaw     = document.getElementById('data-scale-raw');
const dataBtnScaled  = document.getElementById('data-scale-scaled');
const closeDataBtn   = document.getElementById('close-data-btn');
const viewDataBtn    = document.getElementById('view-data-btn');

let dataScale = 'raw';
let cachedData = {};   // { raw: {...}, scaled: {...} } per dataset

function fmt(v) {
  if (typeof v !== 'number') return v;
  return Math.abs(v) < 1000 ? v.toFixed(3) : v.toExponential(2);
}

const MAX_ROWS = 200;

function renderDataTable(data) {
  const { X, feature_names, labels, label_names } = data;
  const n = X.length;
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

  const footer = n > MAX_ROWS
    ? `<tfoot><tr><td colspan="${cols.length + 2}" style="text-align:center;color:rgba(255,255,255,0.3);padding:0.5rem">
        showing ${MAX_ROWS} of ${n} rows
       </td></tr></tfoot>`
    : '';

  dataTable.innerHTML = `<thead>${headerRow}</thead><tbody>${rows}</tbody>${footer}`;
}

async function loadAndShowData(scale) {
  const ds = state.dataset;
  if (!ds) return;
  dataModalTitle.textContent = datasetInfo[ds]?.label || ds;

  if (cachedData[ds]?.[scale]) {
    renderDataTable(cachedData[ds][scale]);
    return;
  }

  dataTable.innerHTML = '<caption style="color:rgba(255,255,255,0.4);padding:2rem">Loading…</caption>';
  const resp = await fetch(`/api/data/${ds}?scale=${scale}`);
  if (!resp.ok) {
    dataTable.innerHTML = `<caption style="color:#f87171;padding:2rem">${await resp.text()}</caption>`;
    return;
  }
  const data = await resp.json();
  cachedData[ds] = cachedData[ds] || {};
  cachedData[ds][scale] = data;
  renderDataTable(data);
}

viewDataBtn.addEventListener('click', () => {
  dataModal.hidden = false;
  loadAndShowData(dataScale);
});

function closeDataModal() {
  dataModal.hidden = true;
  dataTable.innerHTML = '';   // free the DOM nodes immediately
}

closeDataBtn.addEventListener('click', closeDataModal);
dataModal.addEventListener('click', e => { if (e.target === dataModal) closeDataModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeDataModal(); } });

dataBtnRaw.addEventListener('click', () => {
  dataScale = 'raw';
  dataBtnRaw.classList.add('active');
  dataBtnScaled.classList.remove('active');
  loadAndShowData('raw');
});

dataBtnScaled.addEventListener('click', () => {
  dataScale = 'scaled';
  dataBtnScaled.classList.add('active');
  dataBtnRaw.classList.remove('active');
  loadAndShowData('scaled');
});

const codeModal    = document.getElementById('code-modal');
const codeBlock    = document.getElementById('code-block');
const copyCodeBtn  = document.getElementById('copy-code-btn');
const closeCodeBtn = document.getElementById('close-code-btn');
const showCodeBtn  = document.getElementById('show-code-btn');

showCodeBtn.addEventListener('click', () => {
  codeBlock.textContent = generateCode();
  codeModal.hidden = false;
});

closeCodeBtn.addEventListener('click', () => { codeModal.hidden = true; });

codeModal.addEventListener('click', e => {
  if (e.target === codeModal) codeModal.hidden = true;
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') codeModal.hidden = true;
});

copyCodeBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(codeBlock.textContent).then(() => {
    const orig = copyCodeBtn.textContent;
    copyCodeBtn.textContent = 'Copied!';
    setTimeout(() => { copyCodeBtn.textContent = orig; }, 1500);
  });
});

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  try {
    const datasets = await fetch('/api/datasets').then(r => r.json());
    datasets.forEach(ds => {
      datasetInfo[ds.name] = ds;
      const opt = document.createElement('option');
      opt.value = ds.name;
      opt.textContent = ds.label;
      els.datasetSelect.appendChild(opt);
    });

    els.nnSlider.value = N_NEIGHBORS_STEPS.indexOf(state.nNeighbors);
    els.nnValue.textContent = state.nNeighbors;
    els.mdSlider.value = MIN_DIST_STEPS.indexOf(state.minDist);
    els.mdValue.textContent = state.minDist;

    if (datasets.length > 0) {
      state.dataset = datasets[0].name;
      updateDatasetInfo();
      fetchAndRender();
    }
    requestAnimationFrame(positionAllTicks);
  } catch (e) {
    console.error('Failed to load datasets:', e);
    els.loading.textContent = 'Failed to connect to server.';
    els.loading.style.display = 'block';
  }
}

init();
