import { state, datasetInfo, cachedData } from './state.js';
import { els } from './elements.js';
import { MARGIN, AXIS_LABEL_FONT, TICK_FONT, AXIS_BOX } from './constants.js';

// ── Module-level animation state ──────────────────────────────────────────────
let currentEmb = null;
let animFrame  = null;
let plotListenersAttached = false;

export const getCurrentEmb = () => currentEmb;

// ── Axis range ─────────────────────────────────────────────────────────────────
function axisRange(arr) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < mn) mn = arr[i];
    if (arr[i] > mx) mx = arr[i];
  }
  const pad = (mx - mn) * 0.06;
  return [mn - pad, mx + pad];
}

// ── Trace builder ─────────────────────────────────────────────────────────────
export function makeTrace(emb) {
  const isContinuous = emb.label_names === null;
  const hoverText = isContinuous
    ? emb.labels.map(v => `value: ${v.toFixed(2)}`)
    : emb.labels.map(l => emb.label_names[l]);

  const marker = { size: 5, opacity: 0.8 };

  if (state.tab === 'hdbscan' && state.clusterResult) {
    const { colors, labels } = state.clusterResult;
    const hl = state.highlightedCluster;
    marker.color = hl !== null
      ? labels.map((l, i) => l === hl ? colors[i] : '#d0d5e8')
      : colors;
    hoverText.splice(0, hoverText.length,
      ...labels.map(l => l >= 0 ? `cluster ${l}` : 'noise'));

  } else if (state.colorBy !== 'class') {
    const fd = cachedData[`${state.dataset}_${state.scale}`];
    const vals = fd ? fd.X.map(row => row[state.colorBy]) : emb.labels;
    const fname = datasetInfo[state.dataset]?.feature_names?.[state.colorBy]
                  || `feature ${state.colorBy}`;
    const W = els.plot.offsetWidth  || 700;
    const H = els.plot.offsetHeight || 700;
    marker.color     = vals;
    marker.colorscale = 'Viridis';
    marker.showscale = true;
    // Horizontal colorbar sits in the bottom margin above the x-axis label.
    // Title on top so it doesn't collide with axis tick labels below.
    marker.colorbar  = {
      orientation: 'h',
      x: 0.5, xanchor: 'center',
      y: MARGIN.b / H * 0.65, yanchor: 'middle',
      thickness: 10, len: 0.55,
      tickfont: TICK_FONT,
      title: { text: fname, font: TICK_FONT, side: 'top' },
    };

  } else if (!isContinuous && state.highlightedLabel !== null) {
    const palette = datasetInfo[state.dataset]?.label_colors;
    marker.color = emb.labels.map(l =>
      l === state.highlightedLabel ? (palette ? palette[l] : '#5469d4') : '#d0d5e8'
    );

  } else if (isContinuous) {
    marker.color     = emb.labels;
    marker.colorscale = 'Viridis';
    marker.showscale = true;

  } else {
    const palette = datasetInfo[state.dataset]?.label_colors;
    marker.color = palette
      ? emb.labels.map(l => palette[l])
      : emb.labels.map(l => l / Math.max(emb.label_names.length - 1, 1));
    if (!palette) { marker.colorscale = 'Turbo'; marker.showscale = false; }
  }

  return {
    type: 'scattergl', mode: 'markers',
    x: emb.x, y: emb.y,
    text: hoverText,
    hovertemplate: '%{text}<extra></extra>',
    marker,
  };
}

// ── Layout builder ────────────────────────────────────────────────────────────
export function makeLayout(emb) {
  const W = els.plot.offsetWidth  || 700;
  const H = els.plot.offsetHeight || 700;
  return {
    margin: MARGIN,
    paper_bgcolor: '#eef0f5',
    plot_bgcolor:  '#eef0f5',
    showlegend: false,
    xaxis: {
      ...AXIS_BOX,
      domain: [MARGIN.l / W, 1 - MARGIN.r / W],
      range: axisRange(emb.x),
      title: { text: 'coord 1', font: AXIS_LABEL_FONT, standoff: 6 },
    },
    yaxis: {
      ...AXIS_BOX,
      domain: [MARGIN.b / H, 1 - MARGIN.t / H],
      range: axisRange(emb.y),
      title: { text: 'coord 2', font: AXIS_LABEL_FONT, standoff: 6 },
    },
  };
}

// ── Animation ─────────────────────────────────────────────────────────────────
function cubicInOut(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function interpolateEmb(from, to, e) {
  const n = to.x.length;
  const x = new Array(n);
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = from.x[i] + (to.x[i] - from.x[i]) * e;
    y[i] = from.y[i] + (to.y[i] - from.y[i]) * e;
  }
  return { ...to, x, y };
}

export function renderPlot(emb) {
  const isFullRender = state.isFirstRender
    || !currentEmb
    || currentEmb.x.length !== emb.x.length;

  if (isFullRender) {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    Plotly.react(els.plot, [makeTrace(emb)], makeLayout(emb), { responsive: true });
    attachPlotListeners();
    state.isFirstRender = false;
    currentEmb = emb;
    return;
  }

  const from  = currentEmb;
  const start = performance.now();
  const DURATION = 600;

  if (animFrame) cancelAnimationFrame(animFrame);

  (function tick() {
    const t      = Math.min((performance.now() - start) / DURATION, 1);
    const interp = interpolateEmb(from, emb, cubicInOut(t));
    currentEmb   = interp;
    Plotly.react(els.plot, [makeTrace(interp)], makeLayout(interp));
    animFrame = t < 1 ? requestAnimationFrame(tick) : null;
    if (t >= 1) currentEmb = emb;
  })();
}

// ── Plotly event listeners (attached once after first render) ─────────────────
// Callbacks are injected from main.js to avoid circular imports.
let _onPointClick  = null;
let _onDoubleClick = null;

export function setPlotCallbacks(onPointClick, onDoubleClick) {
  _onPointClick  = onPointClick;
  _onDoubleClick = onDoubleClick;
}

function attachPlotListeners() {
  if (plotListenersAttached) return;
  plotListenersAttached = true;
  els.plot.on('plotly_click',       data => _onPointClick  && _onPointClick(data));
  els.plot.on('plotly_doubleclick', ()   => _onDoubleClick && _onDoubleClick());
}
