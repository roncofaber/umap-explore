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
  highlightedLabel: null,
  colorBy: 'class',
  tab: 'umap',              // 'umap' | 'hdbscan'
  minClusterSize: 15,
  minSamples: 5,
  clusterSelectionMethod: 'eom',
  clusterResult: null,      // latest HDBSCAN result from server
  highlightedCluster: null, // null = no highlight; integer = highlighted cluster label
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
  scaleOn:         document.getElementById('scale-on'),
  scaleOff:        document.getElementById('scale-off'),
  colorBySelect:   document.getElementById('color-by-select'),
  colorByGroup:    document.getElementById('color-by-group'),
  resetBtn:        document.getElementById('reset-btn'),
  // tabs
  tabUmap:         document.getElementById('tab-umap'),
  tabHdbscan:      document.getElementById('tab-hdbscan'),
  contentUmap:     document.getElementById('tab-content-umap'),
  contentHdbscan:  document.getElementById('tab-content-hdbscan'),
  // hdbscan controls
  mcsSlider:       document.getElementById('mcs-slider'),
  mcsValue:        document.getElementById('mcs-value'),
  msSlider:        document.getElementById('ms-slider'),
  msValue:         document.getElementById('ms-value'),
  csmEom:          document.getElementById('csm-eom'),
  csmLeaf:         document.getElementById('csm-leaf'),
  clusterStat:     document.getElementById('cluster-stat'),
  plot:            document.getElementById('plot'),
  legend:          document.getElementById('legend'),
  loading:         document.getElementById('loading'),
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

const isMobile = () => window.innerWidth <= 640;

function updateToggleLabel() {
  const hidden = isMobile()
    ? !els.sidebar.classList.contains('mobile-open')
    : els.sidebar.classList.contains('collapsed');
  els.sidebarToggle.textContent = hidden ? '›' : '‹';
}

els.sidebarToggle.addEventListener('click', () => {
  if (isMobile()) {
    const open = els.sidebar.classList.toggle('mobile-open');
    els.sidebarToggle.classList.toggle('mobile-open', open);
    if (open) requestAnimationFrame(positionAllTicks);
  } else {
    const collapsed = els.sidebar.classList.toggle('collapsed');
    els.sidebarToggle.classList.toggle('collapsed', collapsed);
  }
  updateToggleLabel();
  requestAnimationFrame(() => Plotly.relayout(els.plot, {}));
});

updateToggleLabel();

// ── Instant color/highlight re-render (no position animation) ────────────────

function rerenderColors() {
  if (!currentEmb) return;
  Plotly.react(els.plot, [makeTrace(currentEmb)], makeLayout(currentEmb));
  updateLegend(currentEmb);
}

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
  positionTicks(els.mcsSlider);
  positionTicks(els.msSlider);
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


function updateLegend(emb) {
  if (state.tab === 'hdbscan' && state.clusterResult) {
    const { cluster_names, cluster_colors, n_noise, labels } = state.clusterResult;
    const hl = state.highlightedCluster;
    const uniqueClusters = [...new Set(labels.filter(l => l >= 0))].sort((a, b) => a - b);
    const noiseItem = n_noise > 0
      ? `<div class="legend-item${hl !== null && hl !== -1 ? ' dimmed' : ''}" data-cluster="-1">
           <span class="legend-dot" style="background:#c0c8d8"></span>
           <span class="legend-label">noise</span>
         </div>`
      : '';
    els.legend.innerHTML = uniqueClusters.map((cl, i) => {
      const dimmed = hl !== null && cl !== hl ? ' dimmed' : '';
      return `<div class="legend-item${dimmed}" data-cluster="${cl}">
        <span class="legend-dot" style="background:${cluster_colors[i]}"></span>
        <span class="legend-label">cluster ${cl}</span>
      </div>`;
    }).join('') + noiseItem;
    els.legend.querySelectorAll('.legend-item').forEach(item => {
      item.addEventListener('click', () => toggleClusterHighlight(parseInt(item.dataset.cluster)));
    });
    return;
  }

  if (emb.label_names === null || state.colorBy !== 'class') {
    els.legend.innerHTML = '';
    return;
  }
  const palette = datasetInfo[state.dataset]?.label_colors || [];
  const hl = state.highlightedLabel;
  els.legend.innerHTML = emb.label_names.map((name, i) => {
    const dimmed = hl !== null && i !== hl ? ' dimmed' : '';
    return `<div class="legend-item${dimmed}" data-label="${i}">
      <span class="legend-dot" style="background:${palette[i] || '#888'}"></span>
      <span class="legend-label">${name}</span>
    </div>`;
  }).join('');

  els.legend.querySelectorAll('.legend-item').forEach(item => {
    item.addEventListener('click', () => toggleHighlight(parseInt(item.dataset.label)));
  });
}

// ── Plotly trace / layout ─────────────────────────────────────────────────────

function makeTrace(emb) {
  const isContinuous = emb.label_names === null;
  const hoverText = isContinuous
    ? emb.labels.map(v => `value: ${v.toFixed(2)}`)
    : emb.labels.map(l => emb.label_names[l]);

  const marker = { size: 5, opacity: 0.8 };

  if (state.tab === 'hdbscan' && state.clusterResult) {
    const { colors, labels } = state.clusterResult;
    const hl = state.highlightedCluster;
    marker.color = hl !== null
      ? labels.map((l, i) => l === hl ? colors[i] : '#d0d5e8')
      : colors;
    hoverText.splice(0, hoverText.length,
      ...labels.map(l => l >= 0 ? `cluster ${l}` : 'noise'));
  } else if (state.colorBy !== 'class') {
    // Color by a specific feature value
    const fd = cachedData[state.dataset]?.[state.scale];
    const vals = fd ? fd.X.map(row => row[state.colorBy]) : emb.labels;
    const fname = datasetInfo[state.dataset]?.feature_names?.[state.colorBy] || `feature ${state.colorBy}`;
    marker.color = vals;
    marker.colorscale = 'Viridis';
    marker.showscale = true;
    // Horizontal colorbar sits in the bottom margin — box never shifts
    const W = els.plot.offsetWidth  || 700;
    const H = els.plot.offsetHeight || 700;
    marker.colorbar = {
      orientation: 'h',
      x: 0.5, xanchor: 'center',
      y: MARGIN.b / H - 0.02, yanchor: 'top',
      thickness: 10, len: 0.6,
      tickfont: TICK_FONT,
      title: { text: fname, font: TICK_FONT, side: 'bottom' },
    };
  } else if (!isContinuous && state.highlightedLabel !== null) {
    // Highlight one class, mute the rest
    const palette = datasetInfo[state.dataset]?.label_colors;
    marker.color = emb.labels.map(l =>
      l === state.highlightedLabel ? (palette ? palette[l] : '#5469d4') : '#d0d5e8'
    );
  } else if (isContinuous) {
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
const TICK_FONT      = { family: "'JetBrains Mono', monospace", size: 13, color: '#8a94b2' };
const AXIS_BOX = {
  showline: true, linecolor: '#000', linewidth: 1.5, mirror: true,
  showgrid: false, zeroline: false,
  ticks: 'outside', ticklen: 4, tickwidth: 1, tickcolor: '#c0c8d8',
  tickfont: TICK_FONT, nticks: 5, tickformat: '.1f',
};

const MARGIN = { t: 30, r: 60, b: 90, l: 60 };  // l+r = t+b = 120, symmetric left/right

function makeLayout(emb) {
  // Lock axes domain in paper coords so a colorbar can never shift the box.
  const W = els.plot.offsetWidth  || 700;
  const H = els.plot.offsetHeight || 700;
  return {
    margin: MARGIN,
    paper_bgcolor: '#eef0f5',
    plot_bgcolor: '#eef0f5',
    showlegend: false,
    xaxis: {
      ...AXIS_BOX,
      domain: [MARGIN.l / W, 1 - MARGIN.r / W],
      range: axisRange(emb.x),
      title: { text: 'coord 1', font: AXIS_LABEL_FONT, standoff: 6 },
    },
    yaxis: {
      ...AXIS_BOX,
      domain: [MARGIN.b / H, 1 - MARGIN.t / H],
      range: axisRange(emb.y),
      title: { text: 'coord 2', font: AXIS_LABEL_FONT, standoff: 6 },
    },
  };
}

// ── Custom animation ──────────────────────────────────────────────────────────

let currentEmb = null;
let animFrame  = null;
let plotListenersAttached = false;

function attachPlotListeners() {
  if (plotListenersAttached) return;
  plotListenersAttached = true;
  els.plot.on('plotly_click', data => {
    if (!data.points.length) return;
    const idx = data.points[0].pointIndex;
    if (state.tab === 'hdbscan' && state.clusterResult) {
      toggleClusterHighlight(state.clusterResult.labels[idx]);
      return;
    }
    if (!currentEmb || currentEmb.label_names === null || state.colorBy !== 'class') return;
    toggleHighlight(currentEmb.labels[idx]);
  });
  els.plot.on('plotly_doubleclick', () => {
    if (state.highlightedLabel !== null)  { state.highlightedLabel  = null; rerenderColors(); }
    if (state.highlightedCluster !== null){ state.highlightedCluster = null; rerenderColors(); }
  });
}

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
    attachPlotListeners();
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
    if (state.tab === 'hdbscan') {
      await fetchAndCluster();
    } else {
      updateLegend(emb);
      }
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

// ── HDBSCAN ───────────────────────────────────────────────────────────────────

const MCS_STEPS = [5, 10, 15, 20, 30, 50];
const MS_STEPS  = [1,  3,  5, 10, 15, 20];

async function fetchAndCluster() {
  if (!state.dataset || !currentEmb) return;
  els.loading.style.display = 'block';
  try {
    const params = new URLSearchParams({
      method: state.method,
      n_neighbors: state.nNeighbors,
      min_dist: state.minDist,
      n_components: 2,
      metric: state.metric,
      scale: state.scale,
      min_cluster_size: state.minClusterSize,
      min_samples: state.minSamples,
      cluster_selection_method: state.clusterSelectionMethod,
    });
    const resp = await fetch(`/api/cluster/${state.dataset}?${params}`);
    if (!resp.ok) throw new Error(`API error ${resp.status}`);
    state.clusterResult = await resp.json();
    const { n_clusters, n_noise } = state.clusterResult;
    const pct = ((n_noise / currentEmb.x.length) * 100).toFixed(1);
    els.clusterStat.textContent =
      `${n_clusters} cluster${n_clusters !== 1 ? 's' : ''}  ·  ${n_noise} noise points (${pct}%)`;
    els.clusterStat.hidden = false;
    rerenderColors();
  } catch (e) {
    console.error('Clustering failed:', e);
  } finally {
    els.loading.style.display = 'none';
  }
}

let clusterTimer = null;
function scheduleCluster() {
  clearTimeout(clusterTimer);
  clusterTimer = setTimeout(fetchAndCluster, 300);
}

function switchTab(tab) {
  state.tab = tab;
  state.highlightedCluster = null;
  els.tabUmap.classList.toggle('active', tab === 'umap');
  els.tabHdbscan.classList.toggle('active', tab === 'hdbscan');
  els.contentUmap.hidden = tab !== 'umap';
  els.contentHdbscan.hidden = tab !== 'hdbscan';

  if (tab === 'hdbscan') {
    // Ticks were hidden while tab was invisible — reposition now that they're visible
    requestAnimationFrame(positionAllTicks);
    fetchAndCluster();
  } else {
    state.clusterResult = null;
    els.clusterStat.hidden = true;
    rerenderColors();
  }
}

els.tabUmap.addEventListener('click',    () => switchTab('umap'));
els.tabHdbscan.addEventListener('click', () => switchTab('hdbscan'));

els.mcsSlider.addEventListener('input', () => {
  state.minClusterSize = MCS_STEPS[parseInt(els.mcsSlider.value)];
  els.mcsValue.value = state.minClusterSize;
  scheduleCluster();
});

els.mcsValue.addEventListener('change', () => {
  const v = Math.max(2, parseInt(els.mcsValue.value) || 2);
  state.minClusterSize = v;
  els.mcsValue.value = v;
  const idx = MCS_STEPS.reduce((bi, s, i) => Math.abs(s - v) < Math.abs(MCS_STEPS[bi] - v) ? i : bi, 0);
  els.mcsSlider.value = idx;
  scheduleCluster();
});

els.msSlider.addEventListener('input', () => {
  state.minSamples = MS_STEPS[parseInt(els.msSlider.value)];
  els.msValue.value = state.minSamples;
  scheduleCluster();
});

els.msValue.addEventListener('change', () => {
  const v = Math.max(1, parseInt(els.msValue.value) || 1);
  state.minSamples = v;
  els.msValue.value = v;
  const idx = MS_STEPS.reduce((bi, s, i) => Math.abs(s - v) < Math.abs(MS_STEPS[bi] - v) ? i : bi, 0);
  els.msSlider.value = idx;
  scheduleCluster();
});

els.csmEom.addEventListener('click', () => {
  if (state.clusterSelectionMethod === 'eom') return;
  state.clusterSelectionMethod = 'eom';
  els.csmEom.classList.add('active'); els.csmLeaf.classList.remove('active');
  fetchAndCluster();
});

els.csmLeaf.addEventListener('click', () => {
  if (state.clusterSelectionMethod === 'leaf') return;
  state.clusterSelectionMethod = 'leaf';
  els.csmLeaf.classList.add('active'); els.csmEom.classList.remove('active');
  fetchAndCluster();
});

// ── Highlight ─────────────────────────────────────────────────────────────────

function toggleHighlight(label) {
  state.highlightedLabel = state.highlightedLabel === label ? null : label;
  rerenderColors();
}

function toggleClusterHighlight(label) {
  state.highlightedCluster = state.highlightedCluster === label ? null : label;
  rerenderColors();
}

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetParams() {
  state.nNeighbors = 15; state.minDist = 0.1;
  state.metric = 'euclidean'; state.scale = 'scaled';
  state.method = 'umap'; state.highlightedLabel = null; state.colorBy = 'class';

  els.nnSlider.value = N_NEIGHBORS_STEPS.indexOf(15); els.nnValue.textContent = 15;
  els.mdSlider.value = MIN_DIST_STEPS.indexOf(0.1);   els.mdValue.textContent = 0.1;
  els.metricSelect.value = 'euclidean';
  els.scaleOn.classList.add('active');   els.scaleOff.classList.remove('active');
  els.methodUmap.classList.add('active'); els.methodPca.classList.remove('active');
  els.umapParams.classList.remove('params-disabled');
  if (els.colorBySelect) els.colorBySelect.value = 'class';

  requestAnimationFrame(positionAllTicks);
  state.isFirstRender = true;
  fetchAndRender();
}

// ── Color by feature ──────────────────────────────────────────────────────────

async function ensureFeatureData() {
  const ds = state.dataset, sc = state.scale;
  if (cachedData[ds]?.[sc]) return cachedData[ds][sc];
  els.loading.style.display = 'block';
  try {
    const resp = await fetch(`/api/data/${ds}?scale=${sc}`);
    if (!resp.ok) return null;
    const data = await resp.json();
    cachedData[ds] = cachedData[ds] || {};
    cachedData[ds][sc] = data;
    return data;
  } finally {
    els.loading.style.display = 'none';
  }
}

function updateColorByOptions() {
  if (!els.colorBySelect) return;
  const sel = els.colorBySelect;
  while (sel.options.length > 1) sel.remove(1);
  const names = datasetInfo[state.dataset]?.feature_names;
  if (!names || names.length === 0) {
    if (els.colorByGroup) els.colorByGroup.hidden = true;
    return;
  }
  if (els.colorByGroup) els.colorByGroup.hidden = false;
  names.forEach((name, i) => {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = name;
    sel.appendChild(opt);
  });
  state.colorBy = 'class';
  sel.value = 'class';
}

async function onColorByChange() {
  const val = els.colorBySelect.value;
  state.colorBy = val === 'class' ? 'class' : parseInt(val);
  state.highlightedLabel = null;
  if (state.colorBy !== 'class') await ensureFeatureData();
  rerenderColors();
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
  state.highlightedLabel = null;
  state.highlightedCluster = null;
  state.colorBy = 'class';
  updateDatasetInfo();
  updateColorByOptions();
  fetchAndRender();
});

if (els.colorBySelect) els.colorBySelect.addEventListener('change', onColorByChange);
if (els.resetBtn)      els.resetBtn.addEventListener('click', resetParams);

// Plot click listeners are attached lazily after first Plotly render (see attachPlotListeners)

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
const dataFooter     = document.getElementById('data-footer');
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

  dataTable.innerHTML = `<thead>${headerRow}</thead><tbody>${rows}</tbody>`;
  dataFooter.textContent = n > MAX_ROWS ? `showing ${MAX_ROWS} of ${n} rows` : '';
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
  dataTable.innerHTML = '';
  dataFooter.textContent = '';
}

closeDataBtn.addEventListener('click', closeDataModal);
dataModal.addEventListener('click', e => { if (e.target === dataModal) closeDataModal(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeDataModal();
    codeModal.hidden = true;
    if (state.highlightedLabel  !== null) { state.highlightedLabel  = null; rerenderColors(); }
    if (state.highlightedCluster !== null){ state.highlightedCluster = null; rerenderColors(); }
  }
});

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

copyCodeBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(codeBlock.textContent).then(() => {
    const orig = copyCodeBtn.textContent;
    copyCodeBtn.textContent = 'Copied!';
    setTimeout(() => { copyCodeBtn.textContent = orig; }, 1500);
  });
});

// ── Param tooltips ────────────────────────────────────────────────────────────

function initTooltips() {
  let tip = null;

  document.querySelectorAll('.param-q').forEach(el => {
    el.addEventListener('mouseenter', () => {
      if (tip) tip.remove();
      tip = document.createElement('div');
      tip.className = 'param-tip';
      tip.textContent = el.dataset.tip;
      document.body.appendChild(tip);

      const r  = el.getBoundingClientRect();
      const tw = tip.offsetWidth;
      const th = tip.offsetHeight;
      const top  = r.top - th - 8;
      const left = Math.max(8, Math.min(r.left + r.width / 2 - tw / 2, window.innerWidth - tw - 8));
      tip.style.top  = (top < 8 ? r.bottom + 8 : top) + 'px';
      tip.style.left = left + 'px';
    });

    el.addEventListener('mouseleave', () => { if (tip) { tip.remove(); tip = null; } });
  });
}

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
      updateColorByOptions();
      fetchAndRender();
    }
    requestAnimationFrame(positionAllTicks);
    initTooltips();
  } catch (e) {
    console.error('Failed to load datasets:', e);
    els.loading.textContent = 'Failed to connect to server.';
    els.loading.style.display = 'block';
  }
}

init();
