import { state } from './state.js';
import { els } from './elements.js';
import { makeTrace, makeLayout, getCurrentEmb } from './plot.js';

// Legend is now handled entirely by Plotly (dummy traces + native legend).
// This module only manages the HDBSCAN tree-view cluster interactions (clicking
// clusters in the tree to highlight them in the scatter plot).

export function updateLegend(_emb) {
  // No-op for scatter — Plotly's native legend handles class/cluster display.
  // Called from fetchAndRender but nothing to do here anymore.
}

export function rerenderColors() {
  const emb = getCurrentEmb();
  if (!emb) return;
  Plotly.react(els.plot, makeTrace(emb), makeLayout(emb));
}

export function toggleHighlight(label) {
  state.highlightedLabel = state.highlightedLabel === label ? null : label;
  rerenderColors();
}

export function toggleClusterHighlight(label) {
  state.highlightedCluster = state.highlightedCluster === label ? null : label;
  rerenderColors();
}
