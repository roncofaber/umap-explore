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

const AXIS_LABEL_FONT = { family: "'JetBrains Mono', monospace", size: 12, color: '#515978' };
const AXIS_BOX = {
  showline: true, linecolor: '#000', linewidth: 1.5, mirror: true,
  showgrid: false, zeroline: false, showticklabels: false, ticks: '',
};

function makeLayout(emb) {
  return {
    margin: { t: 30, r: 30, b: 50, l: 55 },
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
  const DURATION = 400;

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
  try {
    const emb = await fetchEmbedding();
    renderPlot(emb);
    updateLegend(emb);
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
  state.isFirstRender = true;
  fetchAndRender();
});

els.methodPca.addEventListener('click', () => {
  if (state.method === 'pca') return;
  state.method = 'pca';
  els.methodPca.classList.add('active');
  els.methodUmap.classList.remove('active');
  els.umapParams.classList.add('params-disabled');
  state.isFirstRender = true;
  fetchAndRender();
});

els.scaleOn.addEventListener('click', () => {
  if (state.scale === 'scaled') return;
  state.scale = 'scaled';
  els.scaleOn.classList.add('active');
  els.scaleOff.classList.remove('active');
  state.isFirstRender = true;
  fetchAndRender();
});

els.scaleOff.addEventListener('click', () => {
  if (state.scale === 'raw') return;
  state.scale = 'raw';
  els.scaleOff.classList.add('active');
  els.scaleOn.classList.remove('active');
  state.isFirstRender = true;
  fetchAndRender();
});

els.datasetSelect.addEventListener('change', () => {
  state.dataset = els.datasetSelect.value;
  state.isFirstRender = true;
  updateDatasetInfo();
  fetchAndRender();
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
  } catch (e) {
    console.error('Failed to load datasets:', e);
    els.loading.textContent = 'Failed to connect to server.';
    els.loading.style.display = 'block';
  }
}

init();
