const lut = {
  jobId: null,
  items: [],
  pollTimer: null,
  running: false,
  selectedId: null,
  library: [],
  activeId: "",
  settleTimer: null,
  jobSig: "",
  stack: [],
  soloKey: "",
  custom: {},
  checked: new Set(),
  lastCheckId: "",
  stackSeq: 1,
  editSeq: 0,
  didSeed: false,
  grade: {},
  suggest: {},
  layerTune: {},
};

const LUT_LAYER_COLORS = ["#b5431f", "#2f6b4f", "#3d5a80", "#8a5a2b", "#6b4f8a", "#1f6b6b"];
const LUT_HUE_BANDS = [
  ["hue_red", "红", "#d64545"],
  ["hue_yellow", "黄", "#d6b400"],
  ["hue_green", "绿", "#3a9a4a"],
  ["hue_cyan", "青", "#2aa3b0"],
  ["hue_blue", "蓝", "#3d6fd6"],
  ["hue_magenta", "品红", "#b44aa8"],
];
const LUT_HUE_KEYS = LUT_HUE_BANDS.map((row) => row[0]);
const LUT_COMPARE_W = 1200;
const LUT_DRAG_W = 640;
const LUT_CARD_W = 420;
const lutAfter = { want: "", wantSeq: 0, loading: "", itemId: "" };
const lutLive = { renderer: null, itemId: "", mode: "lut", allowed: false, raf: 0 };
const LUT_DEFAULT_PARAMS = {
  black: 0,
  gamma: 1,
  contrast: 1,
  sparkle: 0,
  sh_tint: [0, 0, 0],
  hi_tint: [0, 0, 0],
  mid_tint: [0, 0, 0],
  hue_moves: [],
  sat_moves: [],
  val_moves: [],
  gray_protect: 0.12,
};

function lutPct(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function lutDefaultStrength() {
  return Number($("lutDefault").value) / 100;
}

function lutItemOf(id) {
  return lut.items.find((it) => it.id === id);
}

function lutById(id) {
  return lut.library.find((item) => item.id === id) || null;
}

function lutSelectedItem() {
  return lutItemOf(lut.selectedId);
}

function lutCheckedItems() {
  return lut.items.filter((item) => lut.checked.has(item.id));
}

function lutCheckedIds() {
  return lutCheckedItems().map((item) => item.id);
}

function lutTargetItems() {
  const checked = lutCheckedItems();
  if (checked.length) return checked;
  const selected = lutSelectedItem();
  return selected ? [selected] : [];
}

function lutSliderItem() {
  if (lut.selectedId && lut.checked.has(lut.selectedId)) return lutSelectedItem();
  return lutTargetItems()[0] || lutSelectedItem();
}

function layerName(layer) {
  const item = lutById(layer.lut_id);
  return item ? item.name : "未知 LUT";
}

function layerSummary(layer) {
  const item = lutById(layer.lut_id);
  if (!item) return "";
  return item.summary || item.description || "";
}

function itemOverrides(id) {
  return lut.custom[id] || null;
}

function layerStrength(layer, item) {
  const overrides = item ? itemOverrides(item.id) : null;
  if (overrides && overrides[layer.key] != null) return overrides[layer.key];
  return layer.strength;
}

function itemHasCustom(item) {
  if (!item) return false;
  const overrides = itemOverrides(item.id);
  if (overrides && lut.stack.some((layer) => overrides[layer.key] != null)) return true;
  const g = lut.grade[item.id];
  if (g && ((g.mode && g.mode !== "auto") || g.desat)) return true;
  const tunes = lut.layerTune[item.id];
  return Boolean(tunes && Object.keys(tunes).length);
}

function pickTuneNum(layer, lookTune, key, fallback) {
  if (layer[key] != null) return layer[key];
  if (lookTune && lookTune[key] != null) return lookTune[key];
  return fallback;
}

function layerTuneDefaults(layer) {
  const look = lutById(layer.lut_id);
  const t = (look && look.tune) || {};
  const out = {
    apply: layer.apply || "auto",
    hue: pickTuneNum(layer, t, "hue", 0),
    sat: pickTuneNum(layer, t, "sat", 0),
    gamma: pickTuneNum(layer, t, "gamma", 1),
    contrast: pickTuneNum(layer, t, "contrast", 1),
  };
  LUT_HUE_KEYS.forEach((key) => {
    out[key] = pickTuneNum(layer, t, key, 0);
  });
  return out;
}

function layerTuneOf(layer, item) {
  const base = layerTuneDefaults(layer);
  const o = item && lut.layerTune[item.id] && lut.layerTune[item.id][layer.key];
  if (!o) return base;
  const out = {
    apply: o.apply || base.apply,
    hue: o.hue != null ? o.hue : base.hue,
    sat: o.sat != null ? o.sat : base.sat,
    gamma: o.gamma != null ? o.gamma : base.gamma,
    contrast: o.contrast != null ? o.contrast : base.contrast,
  };
  LUT_HUE_KEYS.forEach((key) => {
    out[key] = o[key] != null ? o[key] : base[key];
  });
  return out;
}

function newLayerFromLut(lutId) {
  const look = lutById(lutId);
  const t = (look && look.tune) || {};
  const row = {
    key: `L${lut.stackSeq++}`,
    lut_id: lutId,
    strength: lutDefaultStrength(),
    enabled: true,
    apply: "auto",
    hue: t.hue != null ? t.hue : 0,
    sat: t.sat != null ? t.sat : 0,
    gamma: t.gamma != null ? t.gamma : 1,
    contrast: t.contrast != null ? t.contrast : 1,
  };
  LUT_HUE_KEYS.forEach((key) => {
    row[key] = t[key] != null ? t[key] : 0;
  });
  return row;
}

function gradeOf(item) {
  const s = (item && lut.suggest[item.id]) || {};
  const u = (item && lut.grade[item.id]) || {};
  return {
    mode: u.mode || "auto",
    desat: Boolean(u.desat),
    suggested: s.mode || "lut",
    reason: s.reason || "",
  };
}

function gradeQuery(item) {
  const g = gradeOf(item);
  return `grade=${encodeURIComponent(g.mode)}${g.desat ? "&desat=1" : ""}`;
}

function gradePayload(item) {
  const g = gradeOf(item);
  return { mode: g.mode, desat: g.desat };
}

function exportItemGrades() {
  const out = {};
  lut.items.forEach((item) => {
    out[item.id] = gradePayload(item);
  });
  return out;
}

function setGradeMode(mode) {
  lutTargetItems().forEach((item) => {
    lut.grade[item.id] = { ...(lut.grade[item.id] || {}), mode };
  });
  const selected = lutSelectedItem();
  if (selected && !lut.grade[selected.id]) lut.grade[selected.id] = { mode };
  renderLutTune();
  renderLutStats();
  scheduleLutPreview();
}

function toggleDesat() {
  const on = !gradeOf(lutSliderItem()).desat;
  const targets = lutTargetItems();
  if (!targets.length) return;
  targets.forEach((item) => {
    lut.grade[item.id] = { ...(lut.grade[item.id] || {}), desat: on };
  });
  renderLutTune();
  renderLutStats();
  scheduleLutPreview();
}

function formatTune(key, value) {
  if (key === "hue" || key.indexOf("hue_") === 0) return `${Math.round(value)}°`;
  if (key === "sat") return `${Math.round(value * 100)}%`;
  return Number(value).toFixed(2);
}

async function fetchLutSuggest(item) {
  if (!item || !lut.jobId || !lut.stack.length) return;
  const stack = stackQuery(item);
  if (!stack) return;
  try {
    const res = await fetch(`/api/lut/suggest/${item.id}?job_id=${lut.jobId}&stack=${encodeURIComponent(stack)}`);
    if (!res.ok) return;
    lut.suggest[item.id] = await res.json();
    renderLutTune();
  } catch (_err) {
    /* keep last suggestion */
  }
}

function layerEnabled(layer, honorSolo) {
  if (layer.enabled === false) return false;
  if (honorSolo && lut.soloKey) return layer.key === lut.soloKey;
  return true;
}

function effectiveLayers(item, honorSolo) {
  return lut.stack
    .map((layer) => {
      const tune = layerTuneOf(layer, item);
      const row = {
        lut_id: layer.lut_id,
        key: layer.key,
        strength: layerStrength(layer, item),
        enabled: layerEnabled(layer, honorSolo !== false),
        apply: tune.apply,
        hue: tune.hue,
        sat: tune.sat,
        gamma: tune.gamma,
        contrast: tune.contrast,
      };
      LUT_HUE_KEYS.forEach((key) => {
        row[key] = tune[key];
      });
      return row;
    })
    .filter((layer) => layer.enabled && layer.strength > 1e-6);
}

function stackQuery(item) {
  return effectiveLayers(item, true)
    .map(
      (layer) =>
        `${layer.lut_id}:${Math.round(layer.strength * 100)}:${layer.apply}:${Number(layer.hue).toFixed(1)}:${Number(layer.sat).toFixed(3)}:${Number(layer.gamma).toFixed(3)}:${Number(layer.contrast).toFixed(3)}:` +
        LUT_HUE_KEYS.map((key) => Number(layer[key] || 0).toFixed(1)).join(":")
    )
    .join(",");
}

function stackPayload(item) {
  return lut.stack.map((layer) => {
    const tune = layerTuneOf(layer, item);
    const row = {
      lut_id: layer.lut_id,
      strength: layerStrength(layer, item),
      enabled: layerEnabled(layer, false),
      apply: tune.apply,
      hue: tune.hue,
      sat: tune.sat,
      gamma: tune.gamma,
      contrast: tune.contrast,
    };
    LUT_HUE_KEYS.forEach((key) => {
      row[key] = tune[key];
    });
    return row;
  });
}

function exportItemLayers() {
  const out = {};
  lut.items.forEach((item) => {
    out[item.id] = stackPayload(item);
  });
  return out;
}

function lutPreviewUrl(item, w) {
  if (!lut.jobId || !item || !lut.stack.length) return "";
  const stack = stackQuery(item);
  if (!stack) return "";
  return `/api/lut/preview/${item.id}?job_id=${lut.jobId}&stack=${encodeURIComponent(stack)}&w=${w}&${gradeQuery(item)}`;
}

const lutInspectState = { itemId: "", scale: 1, drag: false, sx: 0, sy: 0, sl: 0, st: 0 };

function lutInspectUrl(item) {
  if (!item || !lut.jobId) return "";
  const w = Math.min(Math.max(item.width || 1600, 1600), 3600);
  return lutPreviewUrl(item, w);
}

function applyLutInspectSize(scale) {
  const img = $("lutInspectImg");
  if (!img || !img.naturalWidth) return;
  lutInspectState.scale = Math.max(0.12, Math.min(6, scale));
  img.style.width = `${Math.round(img.naturalWidth * lutInspectState.scale)}px`;
}

function applyLutInspectFit() {
  const view = $("lutInspectView");
  const img = $("lutInspectImg");
  if (!view || !img || !img.naturalWidth) return;
  const scale = Math.min(1, (view.clientWidth - 24) / img.naturalWidth, (view.clientHeight - 24) / img.naturalHeight);
  applyLutInspectSize(scale);
  view.scrollLeft = 0;
  view.scrollTop = 0;
}

function applyLutInspectOne() {
  const view = $("lutInspectView");
  const img = $("lutInspectImg");
  applyLutInspectSize(1);
  if (!view || !img) return;
  view.scrollLeft = Math.max(0, (img.clientWidth - view.clientWidth) / 2);
  view.scrollTop = Math.max(0, (img.clientHeight - view.clientHeight) / 2);
}

function openLutInspect(item) {
  if (!item) return;
  const url = lutInspectUrl(item);
  if (!url) return;
  if (item.id !== lut.selectedId) selectLutItem(item.id);
  lutInspectState.itemId = item.id;
  $("lutInspectName").textContent = item.name;
  $("lutInspectMeta").textContent = `${item.width}×${item.height} · 滚轮缩放，拖动查看细节`;
  const img = $("lutInspectImg");
  img.style.width = "";
  img.onload = () => applyLutInspectFit();
  img.src = url;
  $("lutInspect").showModal();
  if (img.complete && img.naturalWidth) applyLutInspectFit();
}

function stackChipText(item) {
  const layers = effectiveLayers(item);
  if (!layers.length) return lut.stack.length ? "层已关闭" : "未叠加";
  const g = gradeOf(item);
  const steps = layers.map((layer) => {
    const used = layer.apply === "auto" ? (g.suggested === "tune" ? "曲线" : "LUT") : layer.apply === "tune" ? "曲线" : "LUT";
    return `${layerName(layer)} ${used} ${lutPct(layer.strength)}`;
  });
  if (g.desat) steps.unshift("去色");
  return steps.join(" → ");
}

function setLutDefaultLabel() {
  const value = Number($("lutDefault").value);
  $("lutDefaultLabel").textContent = `${value}%`;
  if (Number($("lutDefaultNum").value) !== value) $("lutDefaultNum").value = String(value);
}

function addLutLayer(lutId) {
  const item = lutById(lutId);
  if (!item) return;
  lut.activeId = lutId;
  lut.stack.push(newLayerFromLut(lutId));
  lut.suggest = {};
  renderLut();
  if (lut.selectedId) fetchLutSuggest(lutSelectedItem());
}

function removeLutLayer(key) {
  lut.stack = lut.stack.filter((layer) => layer.key !== key);
  if (lut.soloKey === key) lut.soloKey = "";
  Object.keys(lut.custom).forEach((id) => {
    if (lut.custom[id]) delete lut.custom[id][key];
    if (lut.custom[id] && !Object.keys(lut.custom[id]).length) delete lut.custom[id];
  });
  Object.keys(lut.layerTune).forEach((id) => {
    if (lut.layerTune[id]) delete lut.layerTune[id][key];
    if (lut.layerTune[id] && !Object.keys(lut.layerTune[id]).length) delete lut.layerTune[id];
  });
  renderLut();
}

function moveLutLayer(key, delta) {
  const idx = lut.stack.findIndex((layer) => layer.key === key);
  const next = idx + delta;
  if (idx < 0 || next < 0 || next >= lut.stack.length) return;
  const copy = lut.stack.slice();
  const [row] = copy.splice(idx, 1);
  copy.splice(next, 0, row);
  lut.stack = copy;
  renderLut();
}

function toggleLayerEnabled(key) {
  const layer = lut.stack.find((item) => item.key === key);
  if (!layer) return;
  layer.enabled = layer.enabled === false;
  if (layer.enabled === false && lut.soloKey === key) lut.soloKey = "";
  renderLut();
}

function toggleLayerSolo(key) {
  lut.soloKey = lut.soloKey === key ? "" : key;
  renderLut();
}

function clearLutStack() {
  if (!lut.stack.length) return;
  if (!window.confirm("清空全部叠加层？各张图的单独强度也会去掉。")) return;
  lut.stack = [];
  lut.soloKey = "";
  lut.custom = {};
  lut.grade = {};
  lut.suggest = {};
  lut.layerTune = {};
  renderLut();
}

function setLayerStrength(key, strength, persistItem) {
  const layer = lut.stack.find((item) => item.key === key);
  if (!layer) return;
  const targets = lutTargetItems();
  if (targets.length) {
    targets.forEach((item) => {
      lut.custom[item.id] = { ...(lut.custom[item.id] || {}), [key]: strength };
    });
  } else {
    layer.strength = strength;
  }
  const label = document.querySelector(`[data-layer-label="${key}"]`);
  if (label) label.textContent = lutPct(strength);
  renderLutStats();
  renderLutStackHead();
  scheduleLutPreview();
  if (persistItem) return;
}

function writeLayerTune(key, patch) {
  const layer = lut.stack.find((item) => item.key === key);
  if (!layer) return;
  const targets = lutTargetItems();
  if (targets.length) {
    targets.forEach((item) => {
      const prev = (lut.layerTune[item.id] && lut.layerTune[item.id][key]) || {};
      lut.layerTune[item.id] = { ...(lut.layerTune[item.id] || {}), [key]: { ...prev, ...patch } };
    });
  } else {
    Object.assign(layer, patch);
  }
}

function setLayerApply(key, apply) {
  writeLayerTune(key, { apply });
  renderLutStack();
  renderLutStats();
  scheduleLutPreview();
}

function setLayerTune(key, field, value) {
  writeLayerTune(key, { [field]: value });
  const label = document.querySelector(`[data-ltune-label="${key}-${field}"]`);
  if (label) label.textContent = formatTune(field, value);
  renderLutStats();
  scheduleLutPreview();
}

function resetSelectedItem() {
  const targets = lutTargetItems();
  if (!targets.length) return;
  targets.forEach((item) => {
    delete lut.custom[item.id];
    delete lut.grade[item.id];
    delete lut.layerTune[item.id];
  });
  renderLut();
}

function renderLutLibrary() {
  const box = $("lutLibrary");
  if (!lut.library.length) {
    box.innerHTML = `<p class="hint">还没有保存的 LUT。</p>`;
    return;
  }
  box.innerHTML = lut.library
    .map((item) => {
      const used = lut.stack.filter((layer) => layer.lut_id === item.id);
      const active = used.length || item.id === lut.activeId ? " active" : "";
      const stacked = used.length ? " in-stack" : "";
      const swatches = (item.swatches || [])
        .slice(0, 5)
        .map((rgb) => `<i class="swatch" style="background:rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></i>`)
        .join("");
      const hald = item.has_hald ? `<div class="lut-lib-hald" style="background-image:url('/api/lut/${item.id}/hald')"></div>` : "";
      const badges = used.length
        ? `<div class="lut-chips">${used.map((layer) => `<span class="lut-chip">第${lut.stack.indexOf(layer) + 1}层</span>`).join("")}</div>`
        : "";
      return `<article class="type-card${active}${stacked}" data-lut="${item.id}">
        <div class="lut-lib-row">
          ${hald}
          <div>
            <strong>${item.name}</strong>
            <small>${item.summary || item.description || "已保存"}</small>
            <div class="swatches">${swatches}</div>
            ${badges}
          </div>
        </div>
        <div class="type-actions">
          <button type="button" data-add="${item.id}">${used.length ? "再加一层" : "加入叠加"}</button>
          <button type="button" data-edit="${item.id}">填入表单</button>
          <button type="button" data-del="${item.id}">删除</button>
        </div>
      </article>`;
    })
    .join("");
  box.querySelectorAll("[data-add]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      addLutLayer(btn.dataset.add);
    });
  });
  box.querySelectorAll("[data-edit]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      fillLutForm(btn.dataset.edit);
    });
  });
  box.querySelectorAll("[data-del]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteLut(btn.dataset.del);
    });
  });
  box.querySelectorAll(".type-card").forEach((card) => {
    card.addEventListener("click", () => addLutLayer(card.dataset.lut));
  });
}

function fillLutForm(id) {
  const item = lutById(id);
  if (!item) return;
  lut.activeId = id;
  $("lutName").value = item.name;
  $("lutDesc").value = item.description || "";
  if (item.ref_folder) $("lutRefFolder").value = item.ref_folder;
  renderLutLibrary();
}

function renderLutStackHead() {
  const targets = lutTargetItems();
  const n = lut.stack.length;
  const customCount = targets.filter(itemHasCustom).length;
  $("lutStackEmpty").hidden = n > 0;
  $("lutResetItemBtn").hidden = !customCount;
  $("lutResetItemBtn").textContent = targets.length > 1 ? "勾选恢复默认" : "这张恢复默认";
  if (!n) {
    $("lutStackTitle").textContent = "叠加层";
    $("lutStackHint").textContent = "从上到下依次套色。每层可套自己的 LUT，或改成该层对应的色相/曲线。";
    return;
  }
  if (targets.length > 1) {
    $("lutStackTitle").textContent = `叠加层 · 正在调已勾选的 ${targets.length} 张`;
    $("lutStackHint").textContent = "拖每层滑杆会同时改这些勾选图。按住 Shift 点勾选可连选。";
  } else if (targets[0]) {
    const item = targets[0];
    $("lutStackTitle").textContent = itemHasCustom(item)
      ? `叠加层 · 正在调「${item.name}」（仅这张）`
      : `叠加层 · 正在调「${item.name}」`;
    $("lutStackHint").textContent = "拖每层滑杆改这张图。勾选多张后可一起调。要当默认值，点左侧「同步到全部图」。";
  } else {
    $("lutStackTitle").textContent = `叠加层 · ${n} 层默认叠加`;
    $("lutStackHint").textContent = "从上到下依次套色。勾选图片后，滑杆会同时改这些图。";
  }
}

function renderLutStack() {
  renderLutStackHead();
  const box = $("lutStackList");
  const item = lutSliderItem();
  if (!lut.stack.length) {
    box.innerHTML = "";
    return;
  }
  box.innerHTML = lut.stack
    .map((layer, idx) => {
      const look = lutById(layer.lut_id);
      const strength = layerStrength(layer, item);
      const color = LUT_LAYER_COLORS[idx % LUT_LAYER_COLORS.length];
      const off = layer.enabled === false;
      const solo = lut.soloKey === layer.key;
      const dim = lut.soloKey && !solo;
      const place = idx === 0 ? "先套" : idx === lut.stack.length - 1 ? "后套" : `第 ${idx + 1} 层`;
      const hald = look && look.has_hald
        ? `style="background-image:url('/api/lut/${layer.lut_id}/hald')"`
        : "";
      const spec = layerTuneOf(layer, item);
      const cls = `lut-layer${off ? " is-off" : ""}${solo ? " is-solo" : ""}${dim ? " is-dim" : ""}`;
      const tuneRows = [
        ["hue", "全图", -60, 60, 1, spec.hue],
        ["sat", "饱和", -0.5, 0.3, 0.02, spec.sat],
        ["gamma", "曲线", 0.8, 1.26, 0.02, spec.gamma],
        ["contrast", "对比", 0.86, 1.18, 0.02, spec.contrast],
      ]
        .map(
          ([field, name, min, max, step, value]) => `<label class="lut-tune-row">
          <span>${name}</span>
          <input type="range" min="${min}" max="${max}" step="${step}" value="${value}" data-ltune="${layer.key}" data-ltune-k="${field}" ${off ? "disabled" : ""} />
          <b data-ltune-label="${layer.key}-${field}">${formatTune(field, value)}</b>
        </label>`
        )
        .join("");
      const hueBands = LUT_HUE_BANDS.map(
        ([field, name, color]) => `<label class="lut-hue-band">
          <i style="background:${color}"></i>
          <span>${name}</span>
          <span class="lut-hue-track">
            <input type="range" min="-60" max="60" step="1" value="${spec[field]}" data-ltune="${layer.key}" data-ltune-k="${field}" ${off ? "disabled" : ""} />
          </span>
          <b data-ltune-label="${layer.key}-${field}">${formatTune(field, spec[field])}</b>
        </label>`
      ).join("");
      return `<article class="${cls}" data-layer="${layer.key}" style="border-left-color:${color}">
        <span class="lut-layer-index" style="background:${color}">${idx + 1}</span>
        <div class="lut-layer-hald" ${hald}></div>
        <div class="lut-layer-meta">
          <strong>${layerName(layer)}</strong>
          <small>${place}${layerSummary(layer) ? ` · ${layerSummary(layer)}` : ""}</small>
          <div class="lut-layer-slider">
            <input type="range" min="0" max="150" step="5" value="${Math.round(strength * 100)}" data-str="${layer.key}" ${off ? "disabled" : ""} />
            <b data-layer-label="${layer.key}">${lutPct(strength)}</b>
          </div>
          <div class="lut-layer-apply">
            <button type="button" data-lapply="${layer.key}" data-lapply-mode="auto" class="${spec.apply === "auto" ? "is-on" : ""}">自动</button>
            <button type="button" data-lapply="${layer.key}" data-lapply-mode="lut" class="${spec.apply === "lut" ? "is-on" : ""}">套 LUT</button>
            <button type="button" data-lapply="${layer.key}" data-lapply-mode="tune" class="${spec.apply === "tune" ? "is-on" : ""}">色相/曲线</button>
          </div>
          <div class="lut-layer-tunes${spec.apply === "lut" ? " is-dim" : ""}">
            ${tuneRows}
            <div class="lut-hue-bands">
              <small>分通道色相</small>
              ${hueBands}
            </div>
          </div>
        </div>
        <div class="lut-layer-ops">
          <button type="button" data-on="${layer.key}" class="${off ? "" : "is-on"}">${off ? "已关" : "开"}</button>
          <button type="button" data-solo="${layer.key}" class="${solo ? "is-on" : ""}">独看</button>
          <button type="button" data-up="${layer.key}" ${idx === 0 ? "disabled" : ""}>上移</button>
          <button type="button" data-down="${layer.key}" ${idx === lut.stack.length - 1 ? "disabled" : ""}>下移</button>
          <button type="button" class="is-danger" data-rm="${layer.key}">去掉</button>
        </div>
      </article>`;
    })
    .join("");
  box.querySelectorAll("[data-str]").forEach((input) => {
    input.addEventListener("input", (event) => {
      event.stopPropagation();
      setLayerStrength(input.dataset.str, Number(input.value) / 100, false);
    });
  });
  box.querySelectorAll("[data-on]").forEach((btn) => {
    btn.addEventListener("click", () => toggleLayerEnabled(btn.dataset.on));
  });
  box.querySelectorAll("[data-solo]").forEach((btn) => {
    btn.addEventListener("click", () => toggleLayerSolo(btn.dataset.solo));
  });
  box.querySelectorAll("[data-up]").forEach((btn) => {
    btn.addEventListener("click", () => moveLutLayer(btn.dataset.up, -1));
  });
  box.querySelectorAll("[data-down]").forEach((btn) => {
    btn.addEventListener("click", () => moveLutLayer(btn.dataset.down, 1));
  });
  box.querySelectorAll("[data-rm]").forEach((btn) => {
    btn.addEventListener("click", () => removeLutLayer(btn.dataset.rm));
  });
  box.querySelectorAll("[data-lapply]").forEach((btn) => {
    btn.addEventListener("click", () => setLayerApply(btn.dataset.lapply, btn.dataset.lapplyMode));
  });
  box.querySelectorAll("[data-ltune]").forEach((input) => {
    input.addEventListener("input", () => setLayerTune(input.dataset.ltune, input.dataset.ltuneK, Number(input.value)));
  });
}

function renderLutTune() {
  const item = lutSliderItem();
  const hint = $("lutTuneHint");
  const desatBtn = $("lutDesatBtn");
  if (desatBtn) {
    const on = Boolean(item) && gradeOf(item).desat;
    desatBtn.classList.toggle("is-on", on);
    desatBtn.setAttribute("aria-pressed", on ? "true" : "false");
    desatBtn.disabled = !item;
  }
  if (!item || !lut.stack.length) {
    if (hint) hint.textContent = "先加入叠加层并点开一张图。自动时，每层按空间决定套 LUT 还是用该层色相/曲线。";
    document.querySelectorAll("#lutTuneModes [data-gmode]").forEach((btn) => {
      btn.classList.toggle("is-on", btn.dataset.gmode === "auto");
    });
    return;
  }
  const g = gradeOf(item);
  const used = g.mode === "auto" ? g.suggested : g.mode;
  if (hint) {
    hint.textContent = g.reason
      ? `${g.reason}${g.mode === "auto" ? "（自动，仅作用于设为自动的层）" : "（这张图的层默认）"}`
      : used === "tune"
        ? "这张图默认用各层自己的色相/曲线，避免马赛克。"
        : "这张图默认逐层套 LUT。每层仍可单独改成色相/曲线。";
  }
  const cap = $("lutAfterCaption");
  if (cap) cap.textContent = "套色后";
  document.querySelectorAll("#lutTuneModes [data-gmode]").forEach((btn) => {
    btn.classList.toggle("is-on", btn.dataset.gmode === g.mode);
  });
}

function renderLutStats() {
  const items = lut.items;
  const live = effectiveLayers(lutSliderItem());
  const checked = lutCheckedItems();
  const canLayer = Boolean(lut.stack.length);
  $("lutStatTotal").textContent = items.length || "—";
  $("lutStatName").textContent = lut.stack.length ? `${lut.stack.length} 层` : "—";
  $("lutStatStrength").textContent = live.length ? live.map((layer) => lutPct(layer.strength)).join(" · ") : "—";
  $("lutStatCustom").textContent = items.length ? items.filter(itemHasCustom).length : "—";
  $("lutGalleryCount").textContent = items.length
    ? `${items.length} 张${checked.length ? ` · 已勾选 ${checked.length}` : ""}`
    : "";
  const allBox = $("lutCheckAll");
  if (allBox) {
    allBox.checked = items.length > 0 && checked.length === items.length;
    allBox.indeterminate = checked.length > 0 && checked.length < items.length;
  }
  $("lutExportOneBtn").disabled = !(lut.selectedId && canLayer);
  $("lutExportCheckedBtn").disabled = !(checked.length && canLayer);
  $("lutExportCheckedZipBtn").disabled = !(checked.length && canLayer);
  $("lutExportAllBtn").disabled = !(items.length && canLayer);
  $("lutExportZipBtn").disabled = !(items.length && canLayer);
}

function lutCardChips(item) {
  const chips = lut.stack
    .map((layer, idx) => {
      const color = LUT_LAYER_COLORS[idx % LUT_LAYER_COLORS.length];
      const on = layer.enabled !== false && (!lut.soloKey || lut.soloKey === layer.key);
      return `<span class="lut-chip" style="${on ? "" : "opacity:.45"}"><i style="background:${color}"></i>${layerName(layer)} ${lutPct(layerStrength(layer, item))}</span>`;
    })
    .join("");
  const pre = gradeOf(item).desat ? `<span class="lut-chip"><i style="background:#888"></i>去色</span>` : "";
  return pre + (chips || `<span class="lut-chip">未叠加</span>`);
}

function lutCardBadge(item) {
  return `${item.width}×${item.height}${itemHasCustom(item) ? " · 已单独调" : ""}`;
}

function refreshLutCards() {
  const cards = $("lutGrid").querySelectorAll(".lut-card");
  if (cards.length !== lut.items.length) {
    renderLutGrid();
    return;
  }
  for (const card of cards) {
    const item = lutItemOf(card.dataset.id);
    const img = card.querySelector("[data-card-img]");
    const url = item ? lutPreviewUrl(item, LUT_CARD_W) : "";
    if (!item || !img || Boolean(url) !== img.hasAttribute("data-inspect")) {
      renderLutGrid();
      return;
    }
    if (img.getAttribute("src") !== url) img.src = url;
    const badge = card.querySelector("[data-card-badge]");
    badge.textContent = lutCardBadge(item);
    badge.classList.toggle("is-custom", itemHasCustom(item));
    card.querySelector("[data-card-chips]").innerHTML = lutCardChips(item);
    card.querySelector("[data-card-text]").textContent = stackChipText(item);
  }
}

function renderLutGrid() {
  const grid = $("lutGrid");
  if (!lut.items.length) {
    grid.innerHTML = "";
    return;
  }
  grid.innerHTML = lut.items
    .map((item) => {
      const selected = item.id === lut.selectedId ? " is-selected" : "";
      const checked = lut.checked.has(item.id) ? " is-checked" : "";
      const after = lutPreviewUrl(item, LUT_CARD_W);
      const thumb = `/api/thumb/${item.id}?job_id=${lut.jobId}&w=${LUT_CARD_W}`;
      const custom = itemHasCustom(item);
      const size = item.width && item.height ? `width="${item.width}" height="${item.height}"` : "";
      return `<article class="card lut-card${selected}${checked}" data-id="${item.id}">
        <div class="lut-masonry">
          <label class="lut-check" title="勾选后可一起调、一起导出">
            <input type="checkbox" data-check="${item.id}" ${lut.checked.has(item.id) ? "checked" : ""} />
          </label>
          <img src="${after || thumb}" alt="" loading="lazy" decoding="async" ${size} data-card-img ${after ? `data-inspect="${item.id}" title="点击放大看细节"` : ""} />
          ${after ? `<i class="lut-orig-inset" style="background-image:url('${thumb}')" title="原图"></i>` : ""}
        </div>
        <div class="meta">
          <strong>${item.name}</strong>
          <span class="badge${custom ? " is-custom" : ""}" data-card-badge>${lutCardBadge(item)}</span>
          <div class="lut-chips" data-card-chips>${lutCardChips(item)}</div>
          <div class="lut-card-row">
            <span class="muted" style="font-size:12px" data-card-text>${stackChipText(item)}</span>
            <button type="button" data-dl="${item.id}">导出</button>
          </div>
        </div>
      </article>`;
    })
    .join("");
  grid.querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.target.closest("button, .lut-check, input, [data-inspect]")) return;
      selectLutItem(card.dataset.id);
    });
  });
  grid.querySelectorAll("[data-inspect]").forEach((img) => {
    img.addEventListener("click", (event) => {
      event.stopPropagation();
      openLutInspect(lutItemOf(img.dataset.inspect));
    });
  });
  grid.querySelectorAll("[data-check]").forEach((box) => {
    box.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      toggleLutCheck(box.dataset.check, event);
      box.checked = lut.checked.has(box.dataset.check);
    });
  });
  grid.querySelectorAll("[data-dl]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      downloadOne(btn.dataset.dl);
    });
  });
}

function renderLutCompare() {
  const item = lutSelectedItem();
  $("lutCompareEmpty").hidden = Boolean(item);
  $("lutCompareBody").hidden = !item;
  if (!item || !lut.jobId) return;
  $("lutCompareName").textContent = item.name;
  $("lutCompareSummary").textContent = lut.stack.length
    ? `${item.width}×${item.height} · ${stackChipText(item)}${itemHasCustom(item) ? " · 仅这张" : ""}`
    : `${item.width}×${item.height} · 先加入叠加层再看套色效果`;
  const orig = `/api/preview/${item.id}?job_id=${lut.jobId}&w=${LUT_COMPARE_W}`;
  if ($("lutOrigImg").getAttribute("src") !== orig) $("lutOrigImg").src = orig;
  if (lutLive.itemId !== item.id) prepareLutLive(item);
  loadLutAfter(item, lutPreviewUrl(item, LUT_COMPARE_W));
}

function loadLutAfter(item, url) {
  const img = $("lutAfterImg");
  if (!item || !url) {
    lutAfter.want = "";
    img.removeAttribute("src");
    hideLutLive();
    return;
  }
  if (lutAfter.itemId !== item.id) {
    lutAfter.itemId = item.id;
    img.removeAttribute("src");
  }
  lutAfter.want = url;
  lutAfter.wantSeq = lut.editSeq;
  if (!lutAfter.loading) pumpLutAfter();
}

function pumpLutAfter() {
  const url = lutAfter.want;
  const img = $("lutAfterImg");
  if (!url || img.getAttribute("src") === url) {
    lutAfter.loading = "";
    if (url && lutAfter.wantSeq === lut.editSeq) hideLutLive();
    return;
  }
  lutAfter.loading = url;
  const itemId = lutAfter.itemId;
  const seq = lutAfter.wantSeq;
  const pre = new Image();
  pre.onload = pre.onerror = () => {
    if (pre.naturalWidth && lutAfter.itemId === itemId) {
      img.src = url;
      if (url === lutAfter.want && seq === lut.editSeq) {
        const done = () => {
          if (seq === lut.editSeq && img.getAttribute("src") === url) hideLutLive();
        };
        img.decode().then(done, done);
      }
    }
    lutAfter.loading = "";
    if (lutAfter.want && lutAfter.want !== url) pumpLutAfter();
  };
  pre.src = url;
}

function lutLiveRenderer() {
  if (!lutLive.renderer) {
    lutLive.renderer = window.LutLiveRenderer ? new window.LutLiveRenderer($("lutLiveCanvas")) : { ok: false };
  }
  return lutLive.renderer;
}

function hideLutLive() {
  $("lutLiveCanvas").hidden = true;
}

async function prepareLutLive(item) {
  hideLutLive();
  lutLive.itemId = item ? item.id : "";
  lutLive.allowed = false;
  const renderer = lutLiveRenderer();
  if (!item || !renderer.ok || !lut.jobId) return;
  const q = `job_id=${lut.jobId}&w=${LUT_COMPARE_W}`;
  try {
    const res = await fetch(`/api/lut/live/${item.id}?${q}`);
    const info = await res.json();
    if (!res.ok || !info.live || lutLive.itemId !== item.id) return;
    lutLive.mode = info.mode;
    const ok = await renderer.prepare(
      `${lut.jobId}:${item.id}`,
      `/api/lut/live/${item.id}/src?${q}`,
      `/api/lut/live/${item.id}/room?${q}`
    );
    lutLive.allowed = Boolean(ok) && lutLive.itemId === item.id;
  } catch (_err) {
    lutLive.allowed = false;
  }
}

function lutLiveSpec(item) {
  const used = gradeOf(item).mode === "auto" ? lutLive.mode : gradeOf(item).mode;
  const round = (value, digits) => Number(Number(value || 0).toFixed(digits));
  const layers = [];
  let allTune = true;
  let greenLock = false;
  for (const layer of effectiveLayers(item, true)) {
    const look = lutById(layer.lut_id);
    if (!look) return null;
    const kind = layer.apply === "auto" ? (used === "tune" ? "tune" : "lut") : layer.apply;
    const green = (look.name || "").includes("绿调");
    if (kind !== "tune") allTune = false;
    if (green) greenLock = true;
    const strength = Math.max(0, Math.min(1.5, Math.round(layer.strength * 100) / 100));
    if (strength <= 1e-6) continue;
    if (kind === "tune") {
      layers.push({
        kind,
        strength,
        tune: {
          ...(look.tune || {}),
          hue: round(layer.hue, 1),
          sat: round(layer.sat, 3),
          gamma: round(layer.gamma, 3),
          contrast: round(layer.contrast, 3),
        },
        bands: LUT_HUE_KEYS.map((key) => round(layer[key], 1)),
      });
    } else {
      if (!look.params) return null;
      layers.push({ kind, strength, params: { ...LUT_DEFAULT_PARAMS, ...look.params }, protectWash: !green });
    }
  }
  return { layers, skinOnly: allTune || greenLock, desat: gradeOf(item).desat };
}

function lutLiveCan(item) {
  return (
    lutLive.allowed &&
    item &&
    item.id === lutLive.itemId &&
    window.LutLiveRenderer &&
    window.LutLiveRenderer.supports(lutLiveSpec(item))
  );
}

function drawLutLive() {
  lutLive.raf = 0;
  const item = lutSelectedItem();
  if (!lutLiveCan(item) || !lutLiveRenderer().render(lutLiveSpec(item))) {
    hideLutLive();
    return;
  }
  $("lutLiveCanvas").hidden = false;
}

function renderLut() {
  setLutDefaultLabel();
  renderLutLibrary();
  renderLutStack();
  renderLutTune();
  renderLutStats();
  renderLutGrid();
  renderLutCompare();
}

window.renderLut = renderLut;

async function loadLutLibrary() {
  const res = await fetch("/api/lut/library");
  const data = await res.json();
  applyLutLibrary(data);
}

function applyLutLibrary(data) {
  lut.library = data.luts || [];
  lut.activeId = data.active_id || lut.activeId || (lut.library[0] && lut.library[0].id) || "";
  if (data.ref_folder && !$("lutRefFolder").value) $("lutRefFolder").value = data.ref_folder;
  lut.stack = lut.stack.filter((layer) => lutById(layer.lut_id));
  if (!lut.didSeed && !lut.stack.length && lut.activeId && lutById(lut.activeId)) {
    lut.stack.push(newLayerFromLut(lut.activeId));
  }
  lut.didSeed = true;
  renderLut();
}

async function deleteLut(id) {
  if (!window.confirm("删除这张已保存的 LUT？叠加里用到它的层也会去掉。")) return;
  const data = await (await fetch(`/api/lut/${id}`, { method: "DELETE" })).json();
  lut.stack = lut.stack.filter((layer) => layer.lut_id !== id);
  applyLutLibrary(data);
}

async function generateLut(regenerate) {
  const name = $("lutName").value.trim();
  const folder = $("lutRefFolder").value.trim();
  const description = $("lutDesc").value.trim();
  $("lutGenHint").textContent = "正在用参考图和描述一起生成 LUT…";
  $("lutGenBtn").disabled = true;
  $("lutRegenBtn").disabled = true;
  try {
    const res = await fetch("/api/lut/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        folder,
        description,
        lut_id: regenerate ? lut.activeId : null,
        regenerate: Boolean(regenerate),
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("lutGenHint").textContent = errorDetail(data, "生成失败");
      return;
    }
    applyLutLibrary(data.library);
    lut.activeId = data.lut.id;
    if (!lut.stack.some((layer) => layer.lut_id === data.lut.id)) {
      addLutLayer(data.lut.id);
    } else {
      renderLut();
    }
    $("lutGenHint").textContent = `${data.lut.name} 已保存并加入叠加。${data.lut.summary || ""}`;
  } catch (err) {
    $("lutGenHint").textContent = `生成失败：${err.message || err}`;
  } finally {
    $("lutGenBtn").disabled = false;
    $("lutRegenBtn").disabled = false;
  }
}

function applyLutJob(job) {
  lut.jobId = job.id;
  lut.items = (job.items || [])
    .map((item) => item)
    .sort((a, b) => String(a.rel || a.name).localeCompare(String(b.rel || b.name), "zh"));
  lut.running = job.status === "running" || job.status === "queued" || job.status === "listing";
  if (job.status === "listing") {
    $("lutProgressText").textContent = job.total ? `正在列出 ${job.total} 张` : "正在列出文件…";
  } else if (lut.running) {
    $("lutProgressText").textContent = _paceText("正在载入", job);
  } else if (job.status === "done") {
    $("lutProgressText").textContent = `已载入 ${job.processed} 张`;
  } else if (job.status === "stopped") {
    $("lutProgressText").textContent = `已停止 ${job.processed} / ${job.total || job.processed}`;
  }
  if (!lut.selectedId && lut.items[0]) lut.selectedId = lut.items[0].id;
  const liveIds = new Set(lut.items.map((item) => item.id));
  lut.checked = new Set([...lut.checked].filter((id) => liveIds.has(id)));
  const sig = `${job.id}:${lut.items.length}`;
  if (sig === lut.jobSig) {
    renderLutStats();
    return;
  }
  lut.jobSig = sig;
  renderLut();
}

async function pollLutJob() {
  if (!lut.jobId) return;
  const res = await fetch(`/api/jobs/${lut.jobId}`);
  const job = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("lutScanHint").textContent = "读取进度失败，请重新载入";
    lut.running = false;
    if (lut.pollTimer) clearInterval(lut.pollTimer);
    lut.pollTimer = null;
    $("lutStopBtn").disabled = true;
    return;
  }
  applyLutJob(job);
  if (job.status === "done" || job.status === "stopped") {
    lut.running = false;
    $("lutStopBtn").disabled = true;
    if (lut.pollTimer) clearInterval(lut.pollTimer);
    lut.pollTimer = null;
    $("lutScanHint").textContent = "载入完成。点一张图对比，上方可分开调节每一层。";
  }
}

async function startLutScan() {
  const folder = $("folder").value.trim();
  if (!folder) {
    $("lutScanHint").textContent = "请填写待处理图片文件夹。";
    return;
  }
  lut.custom = {};
  lut.grade = {};
  lut.suggest = {};
  lut.layerTune = {};
  lut.checked = new Set();
  lut.lastCheckId = "";
  lut.selectedId = null;
  lut.running = true;
  $("lutStopBtn").disabled = false;
  $("lutScanHint").textContent = "正在载入待处理图…";
  const res = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder, recursive: $("recursive").checked, kind: "lut" }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("lutScanHint").textContent = errorDetail(data, "载入失败");
    lut.running = false;
    $("lutStopBtn").disabled = true;
    return;
  }
  lut.jobId = data.job_id;
  if (lut.pollTimer) clearInterval(lut.pollTimer);
  lut.pollTimer = setInterval(pollLutJob, 400);
  pollLutJob();
}

async function stopLutScan() {
  if (!lut.jobId || !lut.running) return;
  $("lutStopBtn").disabled = true;
  await fetch(`/api/jobs/${lut.jobId}/stop`, { method: "POST" });
  pollLutJob();
}

function selectLutItem(id) {
  lut.selectedId = id;
  renderLutStack();
  renderLutTune();
  renderLutGrid();
  renderLutCompare();
  renderLutStats();
  fetchLutSuggest(lutItemOf(id));
}

function toggleLutCheck(id, event) {
  const ids = lut.items.map((item) => item.id);
  if (event && event.shiftKey && lut.lastCheckId) {
    const a = ids.indexOf(lut.lastCheckId);
    const b = ids.indexOf(id);
    if (a >= 0 && b >= 0) {
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      const shouldOn = !lut.checked.has(id);
      for (let i = lo; i <= hi; i += 1) {
        if (shouldOn) lut.checked.add(ids[i]);
        else lut.checked.delete(ids[i]);
      }
    }
  } else if (lut.checked.has(id)) {
    lut.checked.delete(id);
  } else {
    lut.checked.add(id);
  }
  lut.lastCheckId = id;
  if (!lut.selectedId) lut.selectedId = id;
  renderLutStack();
  renderLutStats();
  $("lutGrid").querySelectorAll(".lut-card").forEach((card) => {
    card.classList.toggle("is-checked", lut.checked.has(card.dataset.id));
    const box = card.querySelector("[data-check]");
    if (box) box.checked = lut.checked.has(card.dataset.id);
  });
}

function setAllLutChecked(on) {
  lut.checked = new Set(on ? lut.items.map((item) => item.id) : []);
  lut.lastCheckId = "";
  renderLut();
}

function scheduleLutPreview() {
  if (lut.settleTimer) clearTimeout(lut.settleTimer);
  lut.editSeq += 1;
  const item = lutSelectedItem();
  if (lutLiveCan(item)) {
    if (!lutLive.raf) lutLive.raf = requestAnimationFrame(drawLutLive);
  } else {
    loadLutAfter(item, lutPreviewUrl(item, LUT_DRAG_W));
  }
  lut.settleTimer = setTimeout(() => {
    renderLutCompare();
    refreshLutCards();
  }, 380);
}

async function applyCurrentToAll() {
  if (!lut.stack.length) return;
  const item = lutSelectedItem();
  lut.stack.forEach((layer) => {
    layer.strength = layerStrength(layer, item);
    const spec = layerTuneOf(layer, item);
    layer.apply = spec.apply;
    layer.hue = spec.hue;
    layer.sat = spec.sat;
    layer.gamma = spec.gamma;
    layer.contrast = spec.contrast;
    LUT_HUE_KEYS.forEach((key) => {
      layer[key] = spec[key];
    });
  });
  lut.custom = {};
  lut.layerTune = {};
  renderLut();
}

async function downloadOne(id) {
  const item = lutItemOf(id);
  const layers = stackPayload(item);
  const live = layers.filter((layer) => layer.enabled !== false && layer.strength > 1e-6);
  if (!item || !lut.jobId || !live.length) {
    $("lutExportHint").textContent = "请先加入叠加层并载入图片。";
    return;
  }
  $("lutExportHint").textContent = `正在导出 ${item.name}…`;
  try {
    const res = await fetch("/api/lut/export-one", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: lut.jobId,
        item_id: id,
        layers,
        grade: gradePayload(item),
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      $("lutExportHint").textContent = errorDetail(data, "导出失败");
      return;
    }
    const blob = await res.blob();
    const cd = res.headers.get("content-disposition") || "";
    const match = /filename\*?=(?:UTF-8'')?["']?([^"';]+)/i.exec(cd);
    const fallback = `${String(item.name || "image").replace(/\.[^.]+$/, "")}_lut`;
    const name = match ? decodeURIComponent(match[1]) : fallback;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    $("lutExportHint").textContent = `已导出 ${item.name}`;
  } catch (err) {
    $("lutExportHint").textContent = `导出失败：${err.message || err}`;
  }
}

function lutNav(delta) {
  if (!lut.items.length) return;
  const idx = Math.max(0, lut.items.findIndex((it) => it.id === lut.selectedId));
  const next = lut.items[(idx + delta + lut.items.length) % lut.items.length];
  selectLutItem(next.id);
}

async function exportLutFolder(onlyChecked) {
  if (!lut.jobId || !lut.stack.length) return;
  const ids = onlyChecked ? lutCheckedIds() : null;
  if (onlyChecked && !ids.length) {
    $("lutExportHint").textContent = "请先勾选要导出的图。";
    return;
  }
  $("lutExportHint").textContent = onlyChecked
    ? `正在导出勾选的 ${ids.length} 张…`
    : "正在按各张、各层强度导出到文件夹…";
  $("lutExportAllBtn").disabled = true;
  $("lutExportCheckedBtn").disabled = true;
  try {
    const res = await fetch("/api/lut/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: lut.jobId,
        dest: $("destLut").value.trim() || null,
        item_ids: ids,
        layers: stackPayload(null),
        item_layers: exportItemLayers(),
        item_grades: exportItemGrades(),
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("lutExportHint").textContent = errorDetail(data, "导出失败");
      return;
    }
    $("lutExportHint").textContent = `已导出 ${data.count} 张到 ${data.dest}`;
  } catch (err) {
    $("lutExportHint").textContent = `导出失败：${err.message || err}`;
  } finally {
    renderLutStats();
  }
}

async function exportLutZip(onlyChecked) {
  if (!lut.jobId || !lut.stack.length) return;
  const ids = onlyChecked ? lutCheckedIds() : null;
  if (onlyChecked && !ids.length) {
    $("lutExportHint").textContent = "请先勾选要导出的图。";
    return;
  }
  $("lutExportHint").textContent = onlyChecked
    ? `正在打包勾选的 ${ids.length} 张…`
    : "正在打包 ZIP，大图会稍等…";
  $("lutExportZipBtn").disabled = true;
  $("lutExportCheckedZipBtn").disabled = true;
  try {
    const res = await fetch("/api/lut/export-zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: lut.jobId,
        item_ids: ids,
        layers: stackPayload(null),
        item_layers: exportItemLayers(),
        item_grades: exportItemGrades(),
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      $("lutExportHint").textContent = errorDetail(data, "打包失败");
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = onlyChecked ? "lut_checked.zip" : "lut_export.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    $("lutExportHint").textContent = "ZIP 已开始下载。";
  } catch (err) {
    $("lutExportHint").textContent = `打包失败：${err.message || err}`;
  } finally {
    renderLutStats();
  }
}

$("lutDefault").addEventListener("input", setLutDefaultLabel);
$("lutDefaultNum").addEventListener("input", () => {
  const value = Math.min(150, Math.max(0, Number($("lutDefaultNum").value)));
  if (Number.isNaN(value)) return;
  $("lutDefault").value = String(value);
  setLutDefaultLabel();
});
$("lutGenBtn").addEventListener("click", () => generateLut(false));
$("lutRegenBtn").addEventListener("click", () => generateLut(true));
$("lutScanBtn").addEventListener("click", startLutScan);
$("lutStopBtn").addEventListener("click", stopLutScan);
$("lutApplyAllBtn").addEventListener("click", applyCurrentToAll);
$("lutClearStackBtn").addEventListener("click", clearLutStack);
$("lutResetItemBtn").addEventListener("click", resetSelectedItem);
$("lutCheckAll").addEventListener("change", (event) => setAllLutChecked(event.target.checked));
$("lutCheckNone").addEventListener("click", () => setAllLutChecked(false));
$("lutExportOneBtn").addEventListener("click", () => downloadOne(lut.selectedId));
$("lutExportCheckedBtn").addEventListener("click", () => exportLutFolder(true));
$("lutExportCheckedZipBtn").addEventListener("click", () => exportLutZip(true));
$("lutExportAllBtn").addEventListener("click", () => exportLutFolder(false));
$("lutExportZipBtn").addEventListener("click", () => exportLutZip(false));
$("lutPrevBtn").addEventListener("click", () => lutNav(-1));
$("lutNextBtn").addEventListener("click", () => lutNav(1));
$("lutDownloadBtn").addEventListener("click", () => downloadOne(lut.selectedId));
document.querySelectorAll("#lutTuneModes [data-gmode]").forEach((btn) => {
  btn.addEventListener("click", () => setGradeMode(btn.dataset.gmode));
});
$("lutDesatBtn").addEventListener("click", toggleDesat);
$("lutAfterImg").addEventListener("click", () => openLutInspect(lutSelectedItem()));
$("lutInspectFit").addEventListener("click", applyLutInspectFit);
$("lutInspectOne").addEventListener("click", applyLutInspectOne);
$("lutInspectClose").addEventListener("click", () => $("lutInspect").close());
$("lutInspect").addEventListener("click", (event) => {
  if (event.target === $("lutInspect")) $("lutInspect").close();
});
$("lutInspectView").addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    const next = lutInspectState.scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12);
    const view = $("lutInspectView");
    const img = $("lutInspectImg");
    const rect = img.getBoundingClientRect();
    const rx = (event.clientX - rect.left) / Math.max(1, rect.width);
    const ry = (event.clientY - rect.top) / Math.max(1, rect.height);
    applyLutInspectSize(next);
    const after = img.getBoundingClientRect();
    view.scrollLeft += after.left + after.width * rx - event.clientX;
    view.scrollTop += after.top + after.height * ry - event.clientY;
  },
  { passive: false }
);
$("lutInspectView").addEventListener("mousedown", (event) => {
  if (event.button !== 0) return;
  const view = $("lutInspectView");
  lutInspectState.drag = true;
  lutInspectState.sx = event.clientX;
  lutInspectState.sy = event.clientY;
  lutInspectState.sl = view.scrollLeft;
  lutInspectState.st = view.scrollTop;
  view.classList.add("is-drag");
});
window.addEventListener("mousemove", (event) => {
  if (!lutInspectState.drag) return;
  const view = $("lutInspectView");
  view.scrollLeft = lutInspectState.sl - (event.clientX - lutInspectState.sx);
  view.scrollTop = lutInspectState.st - (event.clientY - lutInspectState.sy);
});
window.addEventListener("mouseup", () => {
  lutInspectState.drag = false;
  const view = $("lutInspectView");
  if (view) view.classList.remove("is-drag");
});

loadLutLibrary();
