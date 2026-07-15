import { state, datasetInfo } from './state.js';
import { els } from './elements.js';
import { N_NEIGHBORS_STEPS, MIN_DIST_STEPS, PERPLEXITY_STEPS } from './constants.js';
import { fetchEmbedding, ensureFeatureData } from './api.js';
import { renderPlot, getCurrentEmb, setPlotCallbacks } from './plot.js';
import { updateLegend, toggleHighlight, toggleClusterHighlight, rerenderColors } from './legend.js';
import { fetchAndCluster, switchTab, initClusterControls, setClusterView } from './cluster.js';
import { initCodeModal, initDataModal } from './modals.js';
import {
  updateDatasetInfo, updateColorByOptions, onColorByChange,
  initSidebarToggle, positionAllTicks, initTooltips, setLoading,
  showMethodParams,
} from './ui.js';

// ── Fetch + render orchestrator ───────────────────────────────────────────────
async function fetchAndRender() {
  setLoading('computing…');
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
  state.perplexity = 30; state.pcX = 0; state.pcY = 1;

  els.nnSlider.value = N_NEIGHBORS_STEPS.indexOf(15); els.nnValue.textContent = 15;
  els.mdSlider.value = MIN_DIST_STEPS.indexOf(0.1);   els.mdValue.textContent = 0.1;
  els.metricSelect.value = 'euclidean';
  els.perpSlider.value = PERPLEXITY_STEPS.indexOf(30); els.perpValue.textContent = 30;
  els.scaleOn.classList.add('active');    els.scaleOff.classList.remove('active');
  els.methodUmap.classList.add('active');
  els.methodPca.classList.remove('active');
  els.methodTsne.classList.remove('active');
  els.pc12.classList.add('active');
  els.pc13.classList.remove('active');
  els.pc23.classList.remove('active');
  showMethodParams('umap');
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

  function setMethod(m) {
    if (state.method === m) return;
    state.method = m;
    els.methodUmap.classList.toggle('active', m === 'umap');
    els.methodPca.classList.toggle('active',  m === 'pca');
    els.methodTsne.classList.toggle('active', m === 'tsne');
    if (m === 'tsne') els.tsneMetricSelect.value = state.metric;
    if (m === 'umap') els.metricSelect.value = state.metric;
    showMethodParams(m);
    requestAnimationFrame(positionAllTicks);
    fetchAndRender();
  }
  els.methodUmap.addEventListener('click', () => setMethod('umap'));
  els.methodPca.addEventListener('click',  () => setMethod('pca'));
  els.methodTsne.addEventListener('click', () => setMethod('tsne'));

  els.perpSlider.addEventListener('input', () => {
    state.perplexity = PERPLEXITY_STEPS[parseInt(els.perpSlider.value)];
    els.perpValue.textContent = state.perplexity;
    scheduleRender();
  });

  els.tsneMetricSelect.addEventListener('change', () => {
    state.metric = els.tsneMetricSelect.value;
    fetchAndRender();
  });

  function setPcPair(x, y) {
    state.pcX = x; state.pcY = y;
    els.pc12.classList.toggle('active', x === 0 && y === 1);
    els.pc13.classList.toggle('active', x === 0 && y === 2);
    els.pc23.classList.toggle('active', x === 1 && y === 2);
    fetchAndRender();
  }
  els.pc12.addEventListener('click', () => setPcPair(0, 1));
  els.pc13.addEventListener('click', () => setPcPair(0, 2));
  els.pc23.addEventListener('click', () => setPcPair(1, 2));

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
  function clearHighlights() {
    if (state.highlightedLabel   !== null) { state.highlightedLabel   = null; rerenderColors(); }
    if (state.highlightedCluster !== null) { state.highlightedCluster = null; rerenderColors(); }
  }

  setPlotCallbacks(
    // plotly_click — point in scatter
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
    // plotly_doubleclick — clear highlight
    clearHighlights,
    // plotly_legendclick — legend entry clicked
    // Dummy traces start at curveNumber=1; curveNumber-1 = class/cluster index.
    (data) => {
      const dummyIdx = data.curveNumber - 1;
      if (dummyIdx < 0) return; // main trace, no legend entry
      if (state.tab === 'hdbscan' && state.clusterResult) {
        const uniqueClusters = [...new Set(
          state.clusterResult.labels.filter(l => l >= 0)
        )].sort((a, b) => a - b);
        const label = dummyIdx < uniqueClusters.length ? uniqueClusters[dummyIdx] : -1;
        toggleClusterHighlight(label);
      } else if (state.colorBy === 'class') {
        toggleHighlight(dummyIdx);
      }
    },
    // plotly_legenddoubleclick — clear highlight
    clearHighlights,
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
    els.perpSlider.value = PERPLEXITY_STEPS.indexOf(state.perplexity);
    els.perpValue.textContent = state.perplexity;
    showMethodParams(state.method);

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

// ── Home reset (title click) ──────────────────────────────────────────────────
function resetToHome() {
  els.plotSettings.hidden = true;
  els.settingsBtn.classList.remove('active');
  if (state.tab === 'hdbscan') switchTab('umap');
  showMethodParams('umap');
  const firstOption = els.datasetSelect.options[0];
  if (firstOption && firstOption.value !== state.dataset) {
    state.dataset = firstOption.value;
    els.datasetSelect.value = firstOption.value;
    state.highlightedLabel = null;
    state.highlightedCluster = null;
    state.colorBy = 'class';
    updateDatasetInfo();
    updateColorByOptions();
  }
  resetParams();
}

// ── Plot settings ─────────────────────────────────────────────────────────────
function wireSettings() {
  els.homeBtn.addEventListener('click', resetToHome);

  els.settingsBtn.addEventListener('click', e => {
    e.stopPropagation();
    els.plotSettings.hidden = !els.plotSettings.hidden;
    els.settingsBtn.classList.toggle('active', !els.plotSettings.hidden);
  });

  // Close dropdown when clicking anywhere outside it
  els.plotSettings.addEventListener('click', e => e.stopPropagation());
  document.addEventListener('click', () => {
    els.plotSettings.hidden = true;
    els.settingsBtn.classList.remove('active');
  });

  els.psSize.addEventListener('input', () => {
    state.pointSize = parseInt(els.psSize.value);
    els.psSizeVal.textContent = state.pointSize;
    rerenderColors();
  });
  els.psOpacity.addEventListener('input', () => {
    state.pointOpacity = parseFloat(els.psOpacity.value);
    els.psOpacityVal.textContent = Math.round(state.pointOpacity * 100) + '%';
    rerenderColors();
  });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
wireUmapControls();
wirePlotCallbacks();
initClusterControls();
initCodeModal();
wireGlobalKeys();
wireSettings();
initSidebarToggle(() => rerenderColors());
init();
