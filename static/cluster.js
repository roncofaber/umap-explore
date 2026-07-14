import { state } from './state.js';
import { els } from './elements.js';
import { MCS_STEPS, MS_STEPS, CSE_STEPS } from './constants.js';
import { fetchClusterResult } from './api.js';
import { updateLegend, rerenderColors } from './legend.js';
import { positionAllTicks } from './ui.js';

export async function fetchAndCluster() {
  if (!state.dataset) return;
  els.loading.style.display = 'block';
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
  } catch (e) {
    console.error('Clustering failed:', e);
  } finally {
    els.loading.style.display = 'none';
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
    state.clusterResult = null;
    els.clusterStat.hidden = true;
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
}
