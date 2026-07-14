// Shared mutable state — ES module singleton, safe to import from multiple files.

export const state = {
  dataset:               null,
  method:                'umap',      // 'umap' | 'pca'
  nNeighbors:            15,
  minDist:               0.1,
  metric:                'euclidean',
  scale:                 'scaled',    // 'scaled' | 'raw'
  isFirstRender:         true,
  highlightedLabel:      null,        // class highlight (UMAP tab)
  highlightedCluster:    null,        // cluster highlight (HDBSCAN tab)
  colorBy:               'class',     // 'class' | feature index
  tab:                   'umap',      // 'umap' | 'hdbscan'
  minClusterSize:        15,
  minSamples:            5,
  clusterSelectionMethod: 'eom',
  clusterResult:         null,
};

// Dataset metadata from /api/datasets (populated on init)
export const datasetInfo = {};

// Feature data cache keyed by `${dataset}_${scale}`
export const cachedData = {};
