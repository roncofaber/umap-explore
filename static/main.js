import { state, datasetInfo } from './state.js';
import { els } from './elements.js';
import { N_NEIGHBORS_STEPS, MIN_DIST_STEPS } from './constants.js';
import { fetchEmbedding, ensureFeatureData } from './api.js';
import { renderPlot, getCurrentEmb, setPlotCallbacks } from './plot.js';
import { updateLegend, toggleHighlight, toggleClusterHighlight, rerenderColors } from './legend.js';
import { fetchAndCluster, switchTab, initClusterControls } from './cluster.js';
import { initCodeModal, initDataModal } from './modals.js';
import {
  updateParamStatus, updateDatasetInfo, updateColorByOptions, onColorByChange,
  initSidebarToggle, positionAllTicks, initTooltips, setLoading,
} from './ui.js';

// ── Fetch + render orchestrator ───────────────────────────────────────────────
async function fetchAndRender() {
  setLoading('computing…');
  updateParamStatus();
  try {
    const emb = await fetchEmbedding();
    state.explainedVarianceRatio = emb.explained_variance_ratio ?? null;
    renderPlot(emb);
    if (state.tab === 'hdbscan') {
      await fetchAndCluster();
    } else {
      updateLegend(emb);
    }
  } catch (e) {
    console.error('Failed to load embedding:', e);
  } finally {
    setLoading(null);
  }
}

let renderTimer = null;
function scheduleRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(fetchAndRender, 300);
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetParams() {
  state.nNeighbors = 15; state.minDist = 0.1;
  state.metric = 'euclidean'; state.scale = 'scaled';
  state.method = 'umap'; state.highlightedLabel = null;
  state.highlightedCluster = null; state.colorBy = 'class';

  els.nnSlider.value = N_NEIGHBORS_STEPS.indexOf(15); els.nnValue.textContent  = 15;
  els.mdSlider.value = MIN_DIST_STEPS.indexOf(0.1);   els.mdValue.textContent  = 0.1;
  els.metricSelect.value = 'euclidean';
  els.scaleOn.classList.add('active');    els.scaleOff.classList.remove('active');
  els.methodUmap.classList.add('active'); els.methodPca.classList.remove('active');
  els.umapParams.classList.remove('params-disabled');
  if (els.colorBySelect) els.colorBySelect.value = 'class';

  requestAnimationFrame(positionAllTicks);
  state.isFirstRender = true;
  fetchAndRender();
}

// ── Event listeners ───────────────────────────────────────────────────────────
function wireUmapControls() {
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
    els.methodUmap.classList.add('active'); els.methodPca.classList.remove('active');
    els.umapParams.classList.remove('params-disabled');
    fetchAndRender();
  });

  els.methodPca.addEventListener('click', () => {
    if (state.method === 'pca') return;
    state.method = 'pca';
    els.methodPca.classList.add('active'); els.methodUmap.classList.remove('active');
    els.umapParams.classList.add('params-disabled');
    fetchAndRender();
  });

  els.scaleOn.addEventListener('click', () => {
    if (state.scale === 'scaled') return;
    state.scale = 'scaled';
    els.scaleOn.classList.add('active'); els.scaleOff.classList.remove('active');
    fetchAndRender();
  });

  els.scaleOff.addEventListener('click', () => {
    if (state.scale === 'raw') return;
    state.scale = 'raw';
    els.scaleOff.classList.add('active'); els.scaleOn.classList.remove('active');
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

  els.tabUmap.addEventListener('click',    () => switchTab('umap'));
  els.tabHdbscan.addEventListener('click', () => switchTab('hdbscan'));
}

function wirePlotCallbacks() {
  setPlotCallbacks(
    (data) => {
      if (!data.points.length) return;
      const idx = data.points[0].pointIndex;
      if (state.tab === 'hdbscan' && state.clusterResult) {
        toggleClusterHighlight(state.clusterResult.labels[idx]);
        return;
      }
      const emb = getCurrentEmb();
      if (!emb || emb.label_names === null || state.colorBy !== 'class') return;
      toggleHighlight(emb.labels[idx]);
    },
    () => {
      if (state.highlightedLabel   !== null) { state.highlightedLabel   = null; rerenderColors(); }
      if (state.highlightedCluster !== null) { state.highlightedCluster = null; rerenderColors(); }
    },
  );
}

function wireGlobalKeys() {
  const { closeModal: closeDataModal } = initDataModal();
  const codeModal = document.getElementById('code-modal');

  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    closeDataModal();
    codeModal.hidden = true;
    if (state.highlightedLabel   !== null) { state.highlightedLabel   = null; rerenderColors(); }
    if (state.highlightedCluster !== null) { state.highlightedCluster = null; rerenderColors(); }
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const datasets = await fetch('/api/datasets').then(r => r.json());
    datasets.forEach(ds => {
      datasetInfo[ds.name] = ds;
      const opt = document.createElement('option');
      opt.value = ds.name; opt.textContent = ds.label;
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

// ── Bootstrap ─────────────────────────────────────────────────────────────────
wireUmapControls();
wirePlotCallbacks();
initClusterControls();
initCodeModal();
wireGlobalKeys();
initSidebarToggle(() => rerenderColors());
init();
