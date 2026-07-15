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
  methodTsne:       q('method-tsne'),
  umapParams:       q('umap-params'),
  tsneParams:       q('tsne-params'),
  pcaParams:        q('pca-params'),
  scaleOn:          q('scale-on'),
  scaleOff:         q('scale-off'),
  nnSlider:         q('n-neighbors-slider'),
  nnValue:          q('n-neighbors-value'),
  mdSlider:         q('min-dist-slider'),
  mdValue:          q('min-dist-value'),
  metricSelect:       q('metric-select'),
  tsneMetricSelect:   q('tsne-metric-select'),
  perpSlider:         q('perp-slider'),
  perpValue:          q('perp-value'),
  pc12:               q('pc-12'),
  pc13:               q('pc-13'),
  pc23:               q('pc-23'),
  colorBySelect:    q('color-by-select'),
  colorByGroup:     q('color-by-group'),
  resetBtn:         q('reset-btn'),
  // HDBSCAN controls
  mcsSlider:        q('mcs-slider'),
  mcsValue:         q('mcs-value'),
  msSlider:         q('ms-slider'),
  msValue:          q('ms-value'),
  msAuto:           q('ms-auto'),
  csmEom:           q('csm-eom'),
  csmLeaf:          q('csm-leaf'),
  cseSlider:        q('cse-slider'),
  cseValue:         q('cse-value'),
  ascFalse:         q('asc-false'),
  ascTrue:          q('asc-true'),
  coProjection:     q('co-projection'),
  coData:           q('co-data'),
  viewScatter:      q('view-scatter'),
  viewTree:         q('view-tree'),
  treeWrapper:      q('tree-wrapper'),
  hcSection:        q('hdbscan-color-section'),
  hcCluster:        q('hc-cluster'),
  hcProbability:    q('hc-probability'),
  // Plot
  plot:             q('plot'),
  loading:          q('loading'),
  loadingMsg:       q('loading-msg'),
  // Sidebar
  sidebarToggle:    q('sidebar-toggle'),
  sidebar:          q('controls'),
  // Header
  homeBtn:          q('home-btn'),
  // Modals
  showCodeBtn:      q('show-code-btn'),
  viewDataBtn:      q('view-data-btn'),
  // Plot settings
  settingsBtn:      q('settings-btn'),
  plotSettings:     q('plot-settings'),
  psSize:           q('ps-size'),
  psSizeVal:        q('ps-size-val'),
  psOpacity:        q('ps-opacity'),
  psOpacityVal:     q('ps-opacity-val'),
};
