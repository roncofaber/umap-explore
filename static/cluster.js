import { state } from './state.js';
import { els } from './elements.js';
import { MCS_STEPS, MS_STEPS, CSE_STEPS } from './constants.js';
import { fetchClusterResult } from './api.js';
import { updateLegend, rerenderColors } from './legend.js';
import { positionAllTicks, setLoading } from './ui.js';
// ── Condensed tree layout & rendering ────────────────────────────────────────

function computeYLayout(nodes) {
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  const childrenOf = {};
  nodes.forEach(n => {
    if (n.parent !== -1) {
      (childrenOf[n.parent] = childrenOf[n.parent] || []).push(n.id);
    }
  });

  const root = nodes.find(n => n.parent === -1);
  const layout = {};

  function assign(id, yStart) {
    const kids = (childrenOf[id] || []).filter(c => nodeMap[c]);
    if (!kids.length) {
      layout[id] = [yStart, yStart + nodeMap[id].size];
      return yStart + nodeMap[id].size;
    }
    kids.sort((a, b) => nodeMap[a].size - nodeMap[b].size);
    let y = yStart;
    for (const kid of kids) y = assign(kid, y);
    layout[id] = [yStart, y];
    return y;
  }

  if (root) assign(root.id, 0);
  return layout;
}

function renderTreePlot(treeData) {
  const { nodes, n_points, epsilon } = treeData;
  const layout = computeYLayout(nodes);
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  const traces = [];

  // One scatter trace per cluster (rectangle drawn via fill='toself')
  nodes.forEach(node => {
    const [y0, y1] = layout[node.id] || [0, node.size];
    const { birth_lambda: x0, death_lambda: x1, selected, label, color, size } = node;
    const alpha = selected ? 'cc' : '55';

    traces.push({
      type: 'scatter',
      x: [x0, x1, x1, x0, x0],
      y: [y0, y0, y1, y1, y0],
      fill: 'toself',
      fillcolor: color + alpha,
      line: { color, width: selected ? 1.5 : 0.5 },
      mode: 'lines',
      customdata: [{ label, selected, size }],
      hovertemplate: selected
        ? `<b>cluster ${label}</b><br>${size} points<br>λ: ${x0.toFixed(3)} → ${x1.toFixed(3)}<extra></extra>`
        : `internal node<br>${size} points<extra></extra>`,
      showlegend: false,
    });
  });

  // Vertical connector lines from parent split to each child
  const shapes = [];
  nodes.forEach(node => {
    if (node.parent === -1 || !layout[node.parent] || !layout[node.id]) return;
    const [py0, py1] = layout[node.parent];
    const [cy0, cy1] = layout[node.id];
    shapes.push({
      type: 'line',
      x0: node.birth_lambda, x1: node.birth_lambda,
      y0: (py0 + py1) / 2, y1: (cy0 + cy1) / 2,
      line: { color: '#8a94b2', width: 0.8 },
      layer: 'below',
    });
  });

  // ε threshold line
  const annotations = [];
  if (epsilon > 0) {
    const lx = 1.0 / epsilon;
    shapes.push({ type: 'line', x0: lx, x1: lx, y0: 0, y1: n_points,
                  line: { color: '#e05252', width: 1.5, dash: 'dash' } });
    annotations.push({ x: lx, y: n_points * 0.98, xanchor: 'left', showarrow: false,
                        text: `ε = ${epsilon}`,
                        font: { color: '#e05252', size: 11, family: "'JetBrains Mono', monospace" } });
  }

  const plotLayout = {
    margin: { t: 20, r: 20, b: 55, l: 60 },
    paper_bgcolor: '#eef0f5', plot_bgcolor: '#eef0f5',
    showlegend: false, shapes, annotations,
    xaxis: { title: { text: 'λ  (1 / distance)',
                       font: { family: "'JetBrains Mono', monospace", size: 13, color: '#515978' } },
             showgrid: false, zeroline: false,
             tickfont: { family: "'JetBrains Mono', monospace", size: 11, color: '#8a94b2' } },
    yaxis: { title: { text: 'cluster size',
                       font: { family: "'JetBrains Mono', monospace", size: 13, color: '#515978' } },
             showgrid: false, zeroline: false,
             tickfont: { family: "'JetBrains Mono', monospace", size: 11, color: '#8a94b2' } },
  };

  Plotly.react(els.treeWrapper, traces, plotLayout, { responsive: true });

  // Click a selected cluster bar → highlight those points in the scatter
  els.treeWrapper.on('plotly_click', data => {
    const cd = data.points[0]?.customdata?.[0];
    if (cd?.selected) toggleClusterHighlight(cd.label);
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
    if (!resp.ok) throw new Error(`Tree API error ${resp.status}`);
    renderTreePlot(await resp.json());
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
    requestAnimationFrame(positionAllTicks);
    fetchAndCluster();
  } else {
    // Reset tree view when leaving HDBSCAN tab
    state.clusterView = 'scatter';
    state.clusterResult = null;
    els.clusterStat.hidden = true;
    els.viewScatter.classList.add('active');
    els.viewTree.classList.remove('active');
    els.plot.closest('#plot-wrapper').hidden = false;
    els.treeWrapper.hidden = true;
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
