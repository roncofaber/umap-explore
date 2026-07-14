// DOM element references — queried once at module load (safe because
// <script type="module"> is deferred and runs after the DOM is parsed).

const q = id => document.getElementById(id);

export const els = {
  // Dataset
  datasetSelect:    q('dataset-select'),
  datasetInfoCard:  q('dataset-info'),    // renamed from datasetInfo to avoid collision
  datasetStats:     q('dataset-stats'),
  datasetDesc:      q('dataset-desc'),
  // Tabs
  tabUmap:          q('tab-umap'),
  tabHdbscan:       q('tab-hdbscan'),
  contentUmap:      q('tab-content-umap'),
  contentHdbscan:   q('tab-content-hdbscan'),
  // UMAP controls
  methodUmap:       q('method-umap'),
  methodPca:        q('method-pca'),
  umapParams:       q('umap-params'),
  scaleOn:          q('scale-on'),
  scaleOff:         q('scale-off'),
  nnSlider:         q('n-neighbors-slider'),
  nnValue:          q('n-neighbors-value'),
  mdSlider:         q('min-dist-slider'),
  mdValue:          q('min-dist-value'),
  metricSelect:     q('metric-select'),
  colorBySelect:    q('color-by-select'),
  colorByGroup:     q('color-by-group'),
  resetBtn:         q('reset-btn'),
  // HDBSCAN controls
  mcsSlider:        q('mcs-slider'),
  mcsValue:         q('mcs-value'),
  msSlider:         q('ms-slider'),
  msValue:          q('ms-value'),
  csmEom:           q('csm-eom'),
  csmLeaf:          q('csm-leaf'),
  clusterStat:      q('cluster-stat'),
  // Plot
  plot:             q('plot'),
  legend:           q('legend'),
  loading:          q('loading'),
  paramStatus:      q('param-status'),
  // Sidebar
  sidebarToggle:    q('sidebar-toggle'),
  sidebar:          q('controls'),
  // Modals
  showCodeBtn:      q('show-code-btn'),
  viewDataBtn:      q('view-data-btn'),
};
