import { state, cachedData } from './state.js';
import { els } from './elements.js';
import { setLoading } from './ui.js';

export async function fetchEmbedding() {
  const params = new URLSearchParams({
    method:       state.method,
    n_neighbors:  state.nNeighbors,
    min_dist:     state.minDist,
    n_components: 2,
    metric:       state.metric,
    scale:        state.scale,
    perplexity:   state.perplexity,
    pc_x:         state.pcX,
    pc_y:         state.pcY,
  });
  const resp = await fetch(`/api/embeddings/${state.dataset}?${params}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

export async function fetchClusterResult() {
  const params = new URLSearchParams({
    method:                     state.method,
    n_neighbors:                state.nNeighbors,
    min_dist:                   state.minDist,
    n_components:               2,
    metric:                     state.metric,
    scale:                      state.scale,
    perplexity:                 state.perplexity,
    pc_x:                       state.pcX,
    pc_y:                       state.pcY,
    min_cluster_size:           state.minClusterSize,
    min_samples_auto:           state.minSamplesAuto,
    min_samples:                state.minSamples,
    cluster_selection_method:   state.clusterSelectionMethod,
    cluster_selection_epsilon:  state.clusterSelectionEpsilon,
    allow_single_cluster:       state.allowSingleCluster,
    cluster_on:                 state.clusterOn,
  });
  const resp = await fetch(`/api/cluster/${state.dataset}?${params}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

export async function ensureFeatureData() {
  // Always fetch raw values — scaled (z-score) values are uninterpretable as a colour axis.
  const key = `${state.dataset}_raw`;
  if (cachedData[key]) return cachedData[key];
  setLoading('loading feature data…');
  try {
    const resp = await fetch(`/api/data/${state.dataset}?scale=raw`);
    if (!resp.ok) return null;
    cachedData[key] = await resp.json();
    return cachedData[key];
  } finally {
    setLoading(null);
  }
}
