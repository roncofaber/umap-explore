import { state, datasetInfo } from './state.js';
import { els } from './elements.js';
import { makeTrace, makeLayout, getCurrentEmb } from './plot.js';

export function updateLegend(emb) {
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

  if (!emb || emb.label_names === null || state.colorBy !== 'class') {
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

export function rerenderColors() {
  const emb = getCurrentEmb();
  if (!emb) return;
  Plotly.react(els.plot, [makeTrace(emb)], makeLayout(emb));
  updateLegend(emb);
}

export function toggleHighlight(label) {
  state.highlightedLabel = state.highlightedLabel === label ? null : label;
  rerenderColors();
}

export function toggleClusterHighlight(label) {
  state.highlightedCluster = state.highlightedCluster === label ? null : label;
  rerenderColors();
}
