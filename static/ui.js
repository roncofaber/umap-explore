import { state, datasetInfo } from './state.js';
import { els } from './elements.js';
import { ensureFeatureData } from './api.js';
import { rerenderColors } from './legend.js';

// ── Loading indicator ─────────────────────────────────────────────────────────
export function setLoading(message = null) {
  const active = message !== null;
  els.loading.style.display    = active ? 'block' : 'none';
  els.loadingMsg.style.display = active ? 'block' : 'none';
  if (active) els.loadingMsg.textContent = message;
}


// ── Dataset info card ─────────────────────────────────────────────────────────
export function updateDatasetInfo() {
  const ds = datasetInfo[state.dataset];
  if (!ds) { els.datasetInfoCard.hidden = true; return; }

  const nClasses   = ds.label_colors ? ds.label_colors.length : '—';
  const classLabel = ds.has_labels ? `${nClasses} class${nClasses !== 1 ? 'es' : ''}` : 'continuous';
  // <details>: summary = stats line, body = description
  els.datasetStats.textContent = `${ds.n_points} pts · ${ds.n_features} features · ${classLabel}`;
  els.datasetDesc.textContent  = ds.description || '';
  els.datasetInfoCard.hidden   = false;
}

// ── Color-by selector ─────────────────────────────────────────────────────────
export function updateColorByOptions() {
  if (!els.colorBySelect) return;
  const sel = els.colorBySelect;
  while (sel.options.length > 1) sel.remove(1);
  const names = datasetInfo[state.dataset]?.feature_names;
  if (!names || names.length === 0) {
    if (els.colorByGroup) els.colorByGroup.hidden = true;
    return;
  }
  if (els.colorByGroup) els.colorByGroup.hidden = false;
  names.forEach((name, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = name;
    sel.appendChild(opt);
  });
  state.colorBy = 'class';
  sel.value = 'class';
}

export async function onColorByChange() {
  const val = els.colorBySelect.value;
  state.colorBy = val === 'class' ? 'class' : parseInt(val);
  state.highlightedLabel = null;
  if (state.colorBy !== 'class') await ensureFeatureData();
  rerenderColors();
}

// ── Sidebar toggle ────────────────────────────────────────────────────────────
const isMobile = () => window.innerWidth <= 640;

export function updateToggleLabel() {
  const hidden = isMobile()
    ? !els.sidebar.classList.contains('mobile-open')
    : els.sidebar.classList.contains('collapsed');
  els.sidebarToggle.textContent = hidden ? '›' : '‹';
}

export function initSidebarToggle(onToggle) {
  els.sidebarToggle.addEventListener('click', () => {
    if (isMobile()) {
      const open = els.sidebar.classList.toggle('mobile-open');
      els.sidebarToggle.classList.toggle('mobile-open', open);
      if (open) requestAnimationFrame(positionAllTicks);
    } else {
      const collapsed = els.sidebar.classList.toggle('collapsed');
      els.sidebarToggle.classList.toggle('collapsed', collapsed);
    }
    updateToggleLabel();
    requestAnimationFrame(() => { Plotly.relayout(els.plot, {}); onToggle?.(); });
  });
  updateToggleLabel();
}

// ── Slider tick positioning ───────────────────────────────────────────────────
export function positionTicks(slider) {
  const container = slider.nextElementSibling;
  if (!container || !container.classList.contains('slider-ticks')) return;
  const w = slider.getBoundingClientRect().width;
  if (!w) return;
  const steps     = parseInt(slider.max) - parseInt(slider.min);
  const halfThumb = 6.5;
  container.querySelectorAll('span').forEach((span, i) => {
    span.style.left = (halfThumb + i * (w - 2 * halfThumb) / steps) + 'px';
  });
}

export function positionAllTicks() {
  [els.nnSlider, els.mdSlider, els.mcsSlider, els.msSlider, els.cseSlider]
    .filter(Boolean)
    .forEach(positionTicks);
}

// ── Tooltips ──────────────────────────────────────────────────────────────────
export function initTooltips() {
  let tip = null;
  document.querySelectorAll('.param-q').forEach(el => {
    el.addEventListener('mouseenter', () => {
      if (tip) tip.remove();
      tip = document.createElement('div');
      tip.className   = 'param-tip';
      tip.textContent = el.dataset.tip;
      document.body.appendChild(tip);
      const r  = el.getBoundingClientRect();
      const tw = tip.offsetWidth;
      const th = tip.offsetHeight;
      const top  = r.top - th - 8;
      const left = Math.max(8, Math.min(r.left + r.width / 2 - tw / 2, window.innerWidth - tw - 8));
      tip.style.top  = (top < 8 ? r.bottom + 8 : top) + 'px';
      tip.style.left = left + 'px';
    });
    el.addEventListener('mouseleave', () => { if (tip) { tip.remove(); tip = null; } });
  });
}
