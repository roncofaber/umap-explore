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
  tab:                      'umap',   // 'umap' | 'hdbscan'
  minClusterSize:           15,
  minSamples:               5,
  minSamplesAuto:           true,   // when true, min_samples = None → HDBSCAN defaults to min_cluster_size
  clusterSelectionMethod:   'eom',
  clusterSelectionEpsilon:  0.0,
  allowSingleCluster:       false,
  clusterOn:                'projection', // 'projection' | 'data'
  clusterView:              'scatter',    // 'scatter' | 'tree'
  hdbscanColor:             'cluster',   // 'cluster' | 'probability'
  clusterResult:            null,
  explainedVarianceRatio:   null,    // [pc1, pc2, pc3] from PCA, null otherwise
  perplexity:               30,      // t-SNE perplexity
  pcX:                      0,       // PCA: which component on x-axis (0=PC1, 1=PC2, 2=PC3)
  pcY:                      1,       // PCA: which component on y-axis
  pointSize:                5,
  pointOpacity:             0.8,
};

// Dataset metadata from /api/datasets (populated on init)
export const datasetInfo = {};

// Feature data cache keyed by `${dataset}_${scale}`
export const cachedData = {};
