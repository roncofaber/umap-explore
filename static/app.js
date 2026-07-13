const N_NEIGHBORS_STEPS = [5, 10, 15, 20, 30, 50];
const MIN_DIST_STEPS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0];

const state = {
  dataset: null,
  nNeighbors: 15,
  minDist: 0.1,
  nComponents: 2,
  metric: 'euclidean',
  isFirstRender: true,
  prevNComponents: null,
};

const els = {
  datasetSelect: document.getElementById('dataset-select'),
  nnSlider:      document.getElementById('n-neighbors-slider'),
  nnValue:       document.getElementById('n-neighbors-value'),
  mdSlider:      document.getElementById('min-dist-slider'),
  mdValue:       document.getElementById('min-dist-value'),
  btn2d:         document.getElementById('n-components-2d'),
  btn3d:         document.getElementById('n-components-3d'),
  metricSelect:  document.getElementById('metric-select'),
  plot:          document.getElementById('plot'),
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
    n_neighbors:  state.nNeighbors,
    min_dist:     state.minDist,
    n_components: state.nComponents,
    metric:       state.metric,
  });
  const resp = await fetch(`/api/embeddings/${state.dataset}?${params}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

// ── Plotly trace / layout ─────────────────────────────────────────────────────

function makeTrace(emb) {
  const isContinuous = emb.label_names === null;
  const markerColor = isContinuous
    ? emb.labels
    : emb.labels.map(l => l / Math.max(emb.label_names.length - 1, 1));
  const colorscale = isContinuous ? 'Viridis' : 'Turbo';
  const hoverText = isContinuous
    ? emb.labels.map(v => `value: ${v.toFixed(2)}`)
    : emb.labels.map(l => emb.label_names[l]);

  if (state.nComponents === 3) {
    return {
      type: 'scatter3d', mode: 'markers',
      x: emb.x, y: emb.y, z: emb.z,
      text: hoverText,
      hovertemplate: '%{text}<extra></extra>',
      marker: { size: 3, color: markerColor, colorscale, showscale: isContinuous, opacity: 0.8 },
    };
  }

  return {
    type: 'scattergl', mode: 'markers',
    x: emb.x, y: emb.y,
    text: hoverText,
    hovertemplate: '%{text}<extra></extra>',
    marker: { size: 5, color: markerColor, colorscale, showscale: isContinuous, opacity: 0.8 },
  };
}

const AXIS_LABEL_FONT = { family: "'JetBrains Mono', monospace", size: 10, color: '#515978' };
const AXIS_BOX = {
  showline: true, linecolor: '#000', linewidth: 1.5, mirror: true,
  showgrid: false, zeroline: false, showticklabels: false, ticks: '',
};

function makeLayout(emb) {
  const base = {
    margin: { t: 30, r: 30, b: 50, l: 55 },
    paper_bgcolor: '#eef0f5',
    plot_bgcolor: '#eef0f5',
    showlegend: false,
  };

  if (state.nComponents === 3) {
    return {
      ...base,
      uirevision: 'camera-3d',
      scene: {
        xaxis: { visible: false },
        yaxis: { visible: false },
        zaxis: { visible: false },
      },
    };
  }

  return {
    ...base,
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
// Drive both marker positions and axis ranges with one rAF loop so the
// bounds always contain the data and everything moves together.

let currentEmb = null;  // the embedding currently visible on screen
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
  const out = { ...to, x, y };
  if (to.z && from.z) {
    const z = new Array(n);
    for (let i = 0; i < n; i++) z[i] = from.z[i] + (to.z[i] - from.z[i]) * e;
    out.z = z;
  }
  return out;
}

function renderPlot(emb) {
  if (state.nComponents === 3 && !emb.z) {
    console.error('3D requested but embedding has no z data');
    return;
  }

  const dimensionChanged = state.prevNComponents !== null
    && state.prevNComponents !== state.nComponents;
  state.prevNComponents = state.nComponents;

  // Full re-render: first load, dimension switch, or dataset change (different n)
  if (state.isFirstRender || dimensionChanged
      || !currentEmb || currentEmb.x.length !== emb.x.length) {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    Plotly.react(els.plot, [makeTrace(emb)], makeLayout(emb), { responsive: true });
    state.isFirstRender = false;
    currentEmb = emb;
    return;
  }

  // Animated transition: start from wherever the markers currently are so
  // interrupted animations don't jump.
  const from  = currentEmb;
  const start = performance.now();
  const DURATION = 400;

  if (animFrame) cancelAnimationFrame(animFrame);

  (function tick() {
    const t     = Math.min((performance.now() - start) / DURATION, 1);
    const interp = interpolateEmb(from, emb, cubicInOut(t));
    currentEmb  = interp;
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
  } catch (e) {
    console.error('Failed to load embedding:', e);
  } finally {
    els.loading.style.display = 'none';
  }
}

// Sliders update the label immediately; fetch fires after 300 ms of inactivity
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

els.btn2d.addEventListener('click', () => {
  if (state.nComponents === 2) return;
  state.nComponents = 2;
  els.btn2d.classList.add('active');
  els.btn3d.classList.remove('active');
  fetchAndRender();
});

els.btn3d.addEventListener('click', () => {
  if (state.nComponents === 3) return;
  state.nComponents = 3;
  els.btn3d.classList.add('active');
  els.btn2d.classList.remove('active');
  fetchAndRender();
});

els.metricSelect.addEventListener('change', () => {
  state.metric = els.metricSelect.value;
  fetchAndRender();
});

els.datasetSelect.addEventListener('change', () => {
  state.dataset = els.datasetSelect.value;
  state.isFirstRender = true;
  fetchAndRender();
});

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  try {
    const datasets = await fetch('/api/datasets').then(r => r.json());
    datasets.forEach(ds => {
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
      fetchAndRender();
    }
  } catch (e) {
    console.error('Failed to load datasets:', e);
    els.loading.textContent = 'Failed to connect to server.';
    els.loading.style.display = 'block';
  }
}

init();
