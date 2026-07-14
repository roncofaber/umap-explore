import { state } from './state.js';
import { els } from './elements.js';
import { MCS_STEPS, MS_STEPS, CSE_STEPS } from './constants.js';
import { fetchClusterResult } from './api.js';
import { updateLegend, rerenderColors } from './legend.js';
import { positionAllTicks, setLoading } from './ui.js';
// ── Condensed tree rendering (icicle plot, mirrors hdbscan reference) ─────────

function renderTreePlot(data) {
  const { bars, lines, selected_clusters, epsilon } = data;

  const MONO = "'JetBrains Mono', monospace";
  const SANS = "'Plus Jakarta Sans', sans-serif";
  const AXIS_FONT = { family: MONO, size: 13, color: '#515978' };
  const TICK_FONT = { family: MONO, size: 11, color: '#8a94b2' };

  // Main bar trace — all bars, colored dark-to-light by cluster size
  const traces = [{
    type: 'bar',
    x: bars.centers,
    y: bars.tops,
    base: bars.bottoms,
    width: bars.widths,
    marker: {
      color: bars.sizes_normalized,
      colorscale: [[0, '#dde2ed'], [1, '#515978']],
      showscale: false,
      line: { width: 0 },
    },
    hovertemplate: '%{width:.0f} pts — λ: %{base:.3f} → %{y:.3f}<extra></extra>',
    showlegend: false,
  }];

  const shapes = [];
  const annotations = [];

  // Horizontal connector lines — black, matching hdbscan reference
  lines.forEach(line => {
    shapes.push({
      type: 'line',
      x0: line.x[0], x1: line.x[1],
      y0: line.y[0], y1: line.y[1],
      line: { color: '#1c2033', width: 1.2 },
    });
  });

  // Selected cluster outlines + labels + legend entries
  selected_clusters.forEach(c => {
    const { x_left, x_right, y_bottom, y_top } = c.bounds;
    shapes.push({
      type: 'rect',
      x0: x_left, x1: x_right, y0: y_bottom, y1: y_top,
      fillcolor: 'transparent',
      line: { color: c.color, width: 2.5 },
    });
    annotations.push({
      x: (x_left + x_right) / 2, y: y_top,
      text: `cluster ${c.label}`,
      showarrow: false, yanchor: 'bottom',
      font: { color: c.color, size: 11, family: SANS },
    });
    traces.push({
      type: 'scatter', x: [null], y: [null], mode: 'markers',
      marker: { color: c.color, size: 10, symbol: 'square' },
      name: `cluster ${c.label}`, showlegend: true,
    });
  });

  // ε threshold — horizontal dashed line at λ = 1/ε
  if (epsilon > 0) {
    const ly = 1.0 / epsilon;
    shapes.push({
      type: 'line', x0: 0, x1: 1, xref: 'paper',
      y0: ly, y1: ly,
      line: { color: '#e05252', width: 1.5, dash: 'dash' },
    });
    annotations.push({
      x: 1, xref: 'paper', y: ly, xanchor: 'right', yanchor: 'bottom',
      text: `ε = ${epsilon}`, showarrow: false,
      font: { color: '#e05252', size: 11, family: MONO },
    });
  }

  Plotly.react(els.treeWrapper, traces, {
    margin: { t: 10, r: 20, b: 60, l: 65 },
    paper_bgcolor: '#eef0f5', plot_bgcolor: '#eef0f5',
    bargap: 0,
    showlegend: selected_clusters.length > 0,
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.22,
              font: { family: SANS, size: 12, color: '#515978' } },
    shapes, annotations,
    xaxis: { visible: false, showgrid: false, zeroline: false },
    yaxis: {
      title: { text: 'λ  (1 / distance)', font: AXIS_FONT, standoff: 6 },
      showgrid: false, zeroline: false, tickfont: TICK_FONT,
      tickformat: '.2f',
    },
  }, { responsive: true });

  // Click → find which selected cluster bounds contain the clicked bar
  els.treeWrapper.on('plotly_click', ev => {
    if (!ev.points.length) return;
    const px = ev.points[0].x;
    // lambda value is base + top (base=bottom, y=height)
    const py = ev.points[0].base + ev.points[0].y * 0.5;
    const hit = selected_clusters.find(c =>
      px >= c.bounds.x_left && px <= c.bounds.x_right &&
      py >= c.bounds.y_bottom && py <= c.bounds.y_top
    );
    if (hit) toggleClusterHighlight(hit.label);
  });
  els.treeWrapper.on('plotly_doubleclick', () => {
    if (state.highlightedCluster !== null) { state.highlightedCluster = null; rerenderColors(); }
  });
}


export async function fetchTree() {
  if (!state.dataset) return;
  setLoading('rendering tree…');
  try {
    const params = new URLSearchParams({
      method: state.method, n_neighbors: state.nNeighbors, min_dist: state.minDist,
      n_components: 2, metric: state.metric, scale: state.scale,
      min_cluster_size: state.minClusterSize, min_samples: state.minSamples,
      cluster_selection_method: state.clusterSelectionMethod,
      cluster_selection_epsilon: state.clusterSelectionEpsilon,
      allow_single_cluster: state.allowSingleCluster,
      cluster_on: state.clusterOn,
    });
    const resp = await fetch(`/api/cluster/${state.dataset}/tree?${params}`);
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const treeData = await resp.json();
    if (!treeData.bars) throw new Error(`unexpected API response: ${JSON.stringify(treeData).slice(0, 200)}`);
    renderTreePlot(treeData);
  } catch (e) {
    console.error('Tree fetch failed:', e);
    els.treeWrapper.innerHTML =
      `<p style="color:var(--text-3);padding:2rem;text-align:center">Tree unavailable: ${e.message}</p>`;
  } finally {
    setLoading(null);
  }
}

export function setClusterView(view) {
  state.clusterView = view;
  const isTree = view === 'tree';
  els.viewScatter.classList.toggle('active', !isTree);
  els.viewTree.classList.toggle('active', isTree);
  els.plot.closest('#plot-wrapper').hidden = isTree;
  els.treeWrapper.hidden = !isTree;
  els.legend.style.display = isTree ? 'none' : '';   // hide scatter legend in tree view
  if (isTree) fetchTree();
}

export async function fetchAndCluster() {
  if (!state.dataset) return;
  const msg = state.clusterOn === 'data'
    ? 'clustering high-dimensional data…'
    : 'clustering…';
  setLoading(msg);
  try {
    const result = await fetchClusterResult();
    state.clusterResult = result;
    const { n_clusters, n_noise } = result;
    const total = state.clusterResult.labels.length;
    const pct   = ((n_noise / total) * 100).toFixed(1);
    els.clusterStat.textContent =
      `${n_clusters} cluster${n_clusters !== 1 ? 's' : ''}  ·  ${n_noise} noise points (${pct}%)`;
    els.clusterStat.hidden = false;
    rerenderColors();
    if (state.clusterView === 'tree') fetchTree();
  } catch (e) {
    console.error('Clustering failed:', e);
  } finally {
    setLoading(null);
  }
}

let clusterTimer = null;
export function scheduleCluster() {
  clearTimeout(clusterTimer);
  clusterTimer = setTimeout(fetchAndCluster, 300);
}

export function switchTab(tab) {
  state.tab = tab;
  state.highlightedCluster = null;
  els.tabUmap.classList.toggle('active', tab === 'umap');
  els.tabHdbscan.classList.toggle('active', tab === 'hdbscan');
  els.contentUmap.hidden    = tab !== 'umap';
  els.contentHdbscan.hidden = tab !== 'hdbscan';

  if (tab === 'hdbscan') {
    els.viewToggle.classList.add('visible');
    requestAnimationFrame(positionAllTicks);
    fetchAndCluster();
  } else {
    els.viewToggle.classList.remove('visible');
    state.clusterView = 'scatter';
    state.clusterResult = null;
    els.clusterStat.hidden = true;
    els.viewScatter.classList.add('active');
    els.viewTree.classList.remove('active');
    els.plot.closest('#plot-wrapper').hidden = false;
    els.treeWrapper.hidden = true;
    els.legend.style.display = '';
    rerenderColors();
  }
}

// ── Slider wiring (called from main.js) ───────────────────────────────────────
export function initClusterControls() {
  els.mcsSlider.addEventListener('input', () => {
    state.minClusterSize  = MCS_STEPS[parseInt(els.mcsSlider.value)];
    els.mcsValue.value    = state.minClusterSize;
    scheduleCluster();
  });

  els.mcsValue.addEventListener('change', () => {
    const v = Math.max(2, parseInt(els.mcsValue.value) || 2);
    state.minClusterSize = v;
    els.mcsValue.value   = v;
    const idx = MCS_STEPS.reduce((bi, s, i) =>
      Math.abs(s - v) < Math.abs(MCS_STEPS[bi] - v) ? i : bi, 0);
    els.mcsSlider.value = idx;
    scheduleCluster();
  });

  els.msSlider.addEventListener('input', () => {
    state.minSamples  = MS_STEPS[parseInt(els.msSlider.value)];
    els.msValue.value = state.minSamples;
    scheduleCluster();
  });

  els.msValue.addEventListener('change', () => {
    const v = Math.max(1, parseInt(els.msValue.value) || 1);
    state.minSamples = v;
    els.msValue.value = v;
    const idx = MS_STEPS.reduce((bi, s, i) =>
      Math.abs(s - v) < Math.abs(MS_STEPS[bi] - v) ? i : bi, 0);
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

  els.cseSlider.addEventListener('input', () => {
    state.clusterSelectionEpsilon = CSE_STEPS[parseInt(els.cseSlider.value)];
    els.cseValue.textContent = state.clusterSelectionEpsilon;
    scheduleCluster();
  });

  els.ascFalse.addEventListener('click', () => {
    if (!state.allowSingleCluster) return;
    state.allowSingleCluster = false;
    els.ascFalse.classList.add('active'); els.ascTrue.classList.remove('active');
    fetchAndCluster();
  });

  els.ascTrue.addEventListener('click', () => {
    if (state.allowSingleCluster) return;
    state.allowSingleCluster = true;
    els.ascTrue.classList.add('active'); els.ascFalse.classList.remove('active');
    fetchAndCluster();
  });

  els.viewScatter.addEventListener('click', () => {
    if (state.clusterView === 'scatter') return;
    setClusterView('scatter');
  });

  els.viewTree.addEventListener('click', () => {
    if (state.clusterView === 'tree') return;
    setClusterView('tree');
  });

  els.coProjection.addEventListener('click', () => {
    if (state.clusterOn === 'projection') return;
    state.clusterOn = 'projection';
    els.coProjection.classList.add('active'); els.coData.classList.remove('active');
    fetchAndCluster();
  });

  els.coData.addEventListener('click', () => {
    if (state.clusterOn === 'data') return;
    state.clusterOn = 'data';
    els.coData.classList.add('active'); els.coProjection.classList.remove('active');
    fetchAndCluster();
  });
}
