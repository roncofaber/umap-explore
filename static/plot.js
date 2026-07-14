import { state, datasetInfo, cachedData } from './state.js';
import { els } from './elements.js';
import { MARGIN, AXIS_LABEL_FONT, TICK_FONT, AXIS_BOX } from './constants.js';

// ── Module-level state ────────────────────────────────────────────────────────
let currentEmb = null;
let animFrame  = null;
let plotListenersAttached = false;

export const getCurrentEmb = () => currentEmb;

// ── Helpers ───────────────────────────────────────────────────────────────────
function axisRange(arr) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < mn) mn = arr[i];
    if (arr[i] > mx) mx = arr[i];
  }
  const pad = (mx - mn) * 0.06;
  return [mn - pad, mx + pad];
}

// Horizontal bar (legend or colorbar) positioned in the bottom margin,
// below the x-axis tick labels.  Scales with plot height so it's always
// in the right place regardless of window size.
function belowAxisY(H) {
  return (MARGIN.b * 0.38) / H;  // ~42% up from bottom of paper
}

// Dummy scatter trace: one null point used only as a Plotly legend entry.
function legendEntry(name, color) {
  return {
    type: 'scatter', x: [null], y: [null], mode: 'markers',
    marker: { color, size: 9, symbol: 'circle', opacity: 0.85 },
    name, showlegend: true,
  };
}

// Horizontal colorbar config, placed below the axes.
function hColorbar(H, titleText) {
  return {
    orientation: 'h',
    x: 0.5, xanchor: 'center',
    y: belowAxisY(H), yanchor: 'top',
    thickness: 12, len: 0.55,
    tickfont: TICK_FONT,
    ...(titleText ? { title: { text: titleText, font: TICK_FONT, side: 'top' } } : {}),
  };
}

// ── Trace builder ─────────────────────────────────────────────────────────────
// Always returns an ARRAY: [mainScatterTrace, ...dummyLegendTraces].
// Dummy traces have a single null point and appear only in the Plotly legend.
export function makeTrace(emb) {
  const isContinuous = emb.label_names === null;
  const hoverText = isContinuous
    ? emb.labels.map(v => `value: ${v.toFixed(2)}`)
    : emb.labels.map(l => emb.label_names[l]);

  const marker  = { size: 5, opacity: 0.8 };
  const dummies = [];
  const H = els.plot.offsetHeight || 700;

  if (state.tab === 'hdbscan' && state.clusterResult) {
    // ── HDBSCAN scatter: color by cluster ──────────────────────────────────
    const { colors, labels, cluster_colors } = state.clusterResult;
    const hl = state.highlightedCluster;
    marker.color = hl !== null
      ? labels.map((l, i) => l === hl ? colors[i] : '#d0d5e8')
      : colors;
    hoverText.splice(0, hoverText.length,
      ...labels.map(l => l >= 0 ? `cluster ${l}` : 'noise'));

    const uniqueClusters = [...new Set(labels.filter(l => l >= 0))].sort((a, b) => a - b);
    uniqueClusters.forEach((cl, i) =>
      dummies.push(legendEntry(`cluster ${cl}`, cluster_colors[i])));
    if (state.clusterResult.n_noise > 0)
      dummies.push(legendEntry('noise', '#c0c8d8'));

  } else if (state.colorBy !== 'class') {
    // ── Color by feature: horizontal Viridis colorbar below ───────────────
    const fd    = cachedData[`${state.dataset}_${state.scale}`];
    const vals  = fd ? fd.X.map(row => row[state.colorBy]) : emb.labels;
    const fname = datasetInfo[state.dataset]?.feature_names?.[state.colorBy]
                  || `feature ${state.colorBy}`;
    marker.color      = vals;
    marker.colorscale = 'Viridis';
    marker.showscale  = true;
    marker.colorbar   = hColorbar(H, fname);

  } else if (!isContinuous && state.highlightedLabel !== null) {
    // ── Highlight one class: mute the rest ────────────────────────────────
    const palette = datasetInfo[state.dataset]?.label_colors;
    marker.color = emb.labels.map(l =>
      l === state.highlightedLabel ? (palette ? palette[l] : '#5469d4') : '#d0d5e8'
    );

  } else if (isContinuous) {
    // ── Continuous data (Swiss Roll): Viridis colorbar below ──────────────
    marker.color      = emb.labels;
    marker.colorscale = 'Viridis';
    marker.showscale  = true;
    marker.colorbar   = hColorbar(H, null);

  } else {
    // ── Categorical: hex colors per point + legend dummy entries ──────────
    const palette = datasetInfo[state.dataset]?.label_colors;
    marker.color = palette
      ? emb.labels.map(l => palette[l])
      : emb.labels.map(l => l / Math.max(emb.label_names.length - 1, 1));
    if (!palette) { marker.colorscale = 'Turbo'; marker.showscale = false; }

    if (emb.label_names)
      emb.label_names.forEach((name, i) =>
        dummies.push(legendEntry(name, palette ? palette[i] : '#888')));
  }

  return [
    {
      type: 'scattergl', mode: 'markers',
      x: emb.x, y: emb.y,
      text: hoverText,
      hovertemplate: '%{text}<extra></extra>',
      marker, showlegend: false,
    },
    ...dummies,
  ];
}

// ── Layout builder ────────────────────────────────────────────────────────────
export function makeLayout(emb) {
  const W = els.plot.offsetWidth  || 700;
  const H = els.plot.offsetHeight || 700;
  return {
    margin: MARGIN,
    paper_bgcolor: '#eef0f5',
    plot_bgcolor:  '#eef0f5',
    showlegend: true,
    legend: {
      orientation: 'h',
      x: 0.5, xanchor: 'center',
      y: belowAxisY(H), yanchor: 'top',
      font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, color: '#515978' },
      itemsizing: 'constant',
      tracegroupgap: 0,
    },
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
  const traces = makeTrace(emb);
  const layout = makeLayout(emb);

  const isFullRender = state.isFirstRender
    || !currentEmb
    || currentEmb.x.length !== emb.x.length;

  if (isFullRender) {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    Plotly.react(els.plot, traces, layout, { responsive: true });
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
    Plotly.react(els.plot, makeTrace(interp), makeLayout(interp));
    animFrame = t < 1 ? requestAnimationFrame(tick) : null;
    if (t >= 1) currentEmb = emb;
  })();
}

// ── Plotly event listeners + resize observer ──────────────────────────────────
let _onPointClick    = null;
let _onDoubleClick   = null;
let _onLegendClick   = null;
let _onLegendDblClick = null;

export function setPlotCallbacks(onPointClick, onDoubleClick, onLegendClick, onLegendDblClick) {
  _onPointClick     = onPointClick;
  _onDoubleClick    = onDoubleClick;
  _onLegendClick    = onLegendClick;
  _onLegendDblClick = onLegendDblClick;
}

function attachPlotListeners() {
  if (plotListenersAttached) return;
  plotListenersAttached = true;

  els.plot.on('plotly_click',       data => _onPointClick  && _onPointClick(data));
  els.plot.on('plotly_doubleclick', ()   => _onDoubleClick && _onDoubleClick());

  // Intercept legend-item clicks: trigger highlight instead of Plotly's hide/show.
  // Returning false cancels Plotly's default visibility toggle.
  els.plot.on('plotly_legendclick', data => {
    if (_onLegendClick) _onLegendClick(data);
    return false;
  });
  els.plot.on('plotly_legenddoubleclick', () => {
    if (_onLegendDblClick) _onLegendDblClick();
    return false;
  });

  // Recalculate domains when the plot container is resized so the axes box
  // stays correctly positioned at all screen sizes.
  let resizeTimer = null;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (currentEmb) Plotly.relayout(els.plot, makeLayout(currentEmb));
    }, 120);
  }).observe(els.plot);
}
