const crop = {
  jobId: null,
  items: [],
  filter: "all",
  pollTimer: null,
  running: false,
  selectedId: null,
  edits: {},
  compareImg: null,
  compareItemId: null,
  fit: null,
  drag: null,
  saveTimer: null,
};

function cropNeed(item) {
  return Boolean(item && item.cropped);
}

function visibleCropItems() {
  if (crop.filter === "need") return crop.items.filter(cropNeed);
  if (crop.filter === "skip") return crop.items.filter((it) => !cropNeed(it));
  return crop.items;
}

function cropBorderName(item) {
  return { black: "黑边", white: "白边", mixed: "黑白边", manual: "自选" }[item.border] || "无边";
}

function cropBoxOf(item) {
  const w = item.width || 0;
  const h = item.height || 0;
  const box = item.crop_box || [0, 0, w, h];
  return [box[0] || 0, box[1] || 0, box[2] || w, box[3] || h];
}

function applyBoxLocal(item, box, edited) {
  const w = item.width || 0;
  const h = item.height || 0;
  if (!item.auto_crop_box) item.auto_crop_box = cropBoxOf(item).slice();
  let [left, top, right, bottom] = box;
  left = Math.min(Math.max(0, Math.round(left)), Math.max(0, w - 1));
  top = Math.min(Math.max(0, Math.round(top)), Math.max(0, h - 1));
  right = Math.min(w, Math.max(left + 1, Math.round(right)));
  bottom = Math.min(h, Math.max(top + 1, Math.round(bottom)));
  item.crop_box = [left, top, right, bottom];
  item.crop_width = right - left;
  item.crop_height = bottom - top;
  item.trim_left = left;
  item.trim_top = top;
  item.trim_right = w - right;
  item.trim_bottom = h - bottom;
  item.cropped = !(left === 0 && top === 0 && right === w && bottom === h);
  item.edited = Boolean(edited);
  if (item.cropped && (!item.border || item.border === "none")) item.border = "manual";
  if (!item.cropped) item.border = "none";
  return item;
}

function cropMetrics(item) {
  const w = item.width || 0;
  const h = item.height || 0;
  const [left, top, right, bottom] = cropBoxOf(item);
  const cw = Math.max(0, right - left);
  const ch = Math.max(0, bottom - top);
  const area = w * h;
  return {
    left,
    top,
    right,
    bottom,
    trimL: left,
    trimT: top,
    trimR: w - right,
    trimB: h - bottom,
    cropW: cw,
    cropH: ch,
    removed: area ? 1 - (cw * ch) / area : 0,
  };
}

function sideLabel(px, full) {
  if (!px) return "0";
  const pct = full ? ((px / full) * 100).toFixed(1) : "0";
  return `${px}px · ${pct}%`;
}

function cropBadge(item) {
  const m = cropMetrics(item);
  if (!cropNeed(item)) return item.edited ? "已改回原图" : "无需裁剪";
  const extra = item.edited ? " · 已改框" : "";
  return `${cropBorderName(item)} 裁掉 ${(m.removed * 100).toFixed(1)}%${extra}`;
}

function cropSideText(item) {
  if (!cropNeed(item)) return "将原样复制";
  const m = cropMetrics(item);
  return `上${m.trimT} 下${m.trimB} 左${m.trimL} 右${m.trimR} · ${item.width}×${item.height} → ${m.cropW}×${m.cropH}`;
}

function sameBox(a, b) {
  if (!a || !b || a.length < 4 || b.length < 4) return false;
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}

function layoutFit(nw, nh, cw, ch) {
  const scale = Math.min(cw / nw, ch / nh);
  return {
    x: (cw - nw * scale) / 2,
    y: (ch - nh * scale) / 2,
    w: nw * scale,
    h: nh * scale,
    scale,
  };
}

function renderCropStats() {
  const items = crop.items;
  const need = items.filter(cropNeed);
  $("cropStatTotal").textContent = items.length || "—";
  $("cropStatNeed").textContent = items.length ? need.length : "—";
  $("cropStatSkip").textContent = items.length ? items.length - need.length : "—";
  $("cropStatSaved").textContent = items.length ? items.filter((it) => it.saved).length : "—";
}

function renderCropChart() {
  const canvas = $("cropHist");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const height = 140;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const rows = [
    ["黑边", crop.items.filter((it) => it.border === "black").length],
    ["白边", crop.items.filter((it) => it.border === "white").length],
    ["黑白边", crop.items.filter((it) => it.border === "mixed").length],
    ["自选", crop.items.filter((it) => it.border === "manual").length],
    ["无需裁剪", crop.items.filter((it) => !cropNeed(it)).length],
  ];
  const max = Math.max(1, ...rows.map((r) => r[1]));
  const rowH = height / rows.length;
  rows.forEach(([name, count], i) => {
    const bar = ((width - 140) * count) / max;
    const y = i * rowH + 8;
    ctx.fillStyle = "#c9b8a4";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(130, y, Math.max(bar, 0), rowH - 16, 5);
    else ctx.rect(130, y, Math.max(bar, 0), rowH - 16);
    ctx.fill();
    ctx.fillStyle = "#6f665d";
    ctx.font = "12px sans-serif";
    ctx.fillText(name, 4, y + rowH * 0.55);
    ctx.fillText(String(count), 138 + bar, y + rowH * 0.55);
  });
}

function containedRect(img) {
  const nw = img.naturalWidth || 1;
  const nh = img.naturalHeight || 1;
  const cw = img.clientWidth || 1;
  const ch = img.clientHeight || 1;
  return layoutFit(nw, nh, cw, ch);
}

function boxToDisplay(item, img, box) {
  const nw = img.naturalWidth || item.width || 1;
  const nh = img.naturalHeight || item.height || 1;
  const sx = nw / Math.max(1, item.width || nw);
  const sy = nh / Math.max(1, item.height || nh);
  return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy];
}

function layoutCardCrop(card) {
  const item = crop.items.find((it) => it.id === card.dataset.id);
  const img = card.querySelector("img");
  const rect = card.querySelector(".crop-rect");
  if (!item || !img || !rect || !img.naturalWidth || !item.width) return;
  if (!cropNeed(item)) {
    rect.hidden = true;
    return;
  }
  const fit = containedRect(img);
  const [left, top, right, bottom] = boxToDisplay(item, img, cropBoxOf(item));
  rect.hidden = false;
  rect.style.left = `${fit.x + left * fit.scale}px`;
  rect.style.top = `${fit.y + top * fit.scale}px`;
  rect.style.width = `${Math.max(2, (right - left) * fit.scale)}px`;
  rect.style.height = `${Math.max(2, (bottom - top) * fit.scale)}px`;
}

function renderCropGrid() {
  const items = visibleCropItems();
  $("cropGalleryCount").textContent = crop.items.length
    ? `当前视图 ${items.length} 张`
    : "还没有去边分析结果";
  $("cropGrid").innerHTML = items
    .map((item) => {
      const hit = cropNeed(item);
      const selected = item.id === crop.selectedId ? " is-selected" : "";
      return `
        <article class="card ${hit ? "is-text" : ""}${selected}" data-id="${item.id}">
          <div class="crop-thumb">
            <img src="/api/thumb/${item.id}?job_id=${crop.jobId}&w=420" alt="" />
            <div class="crop-rect" hidden></div>
          </div>
          <div class="meta">
            <strong title="${item.rel}">${item.name}</strong>
            <span class="badge ${hit ? "hot" : ""}">${cropBadge(item)}</span>
            <span class="crop-sides">${cropSideText(item)}</span>
          </div>
        </article>`;
    })
    .join("");
  $("cropGrid").querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => selectCropItem(card.dataset.id));
    const img = card.querySelector("img");
    if (img.complete) layoutCardCrop(card);
    else img.addEventListener("load", () => layoutCardCrop(card));
  });
}

function setCropScanControls() {
  $("cropScanBtn").disabled = false;
  $("cropScanBtn").textContent = crop.items.length || crop.running ? "重新分析" : "开始分析";
  $("cropStopBtn").disabled = !crop.running;
}

function restoreCropExportButton() {
  $("exportCropBtn").disabled = !crop.items.length;
}

function selectedCropItem() {
  return crop.items.find((it) => it.id === crop.selectedId) || null;
}

function updateCropCompareMeta() {
  const item = selectedCropItem();
  if (!item) return;
  const m = cropMetrics(item);
  $("cropCompareName").textContent = item.name;
  const parts = [
    `${item.width}×${item.height} → ${m.cropW}×${m.cropH}`,
    cropNeed(item) ? `裁掉 ${(m.removed * 100).toFixed(1)}% 面积` : "不裁，原样保留",
  ];
  if (item.edited) parts.push("已手动改框");
  if (item.saved) parts.push("已另存");
  $("cropCompareSummary").textContent = parts.join(" · ");
  $("cropCompareStats").innerHTML = [
    ["上边", sideLabel(m.trimT, item.height)],
    ["下边", sideLabel(m.trimB, item.height)],
    ["左边", sideLabel(m.trimL, item.width)],
    ["右边", sideLabel(m.trimR, item.width)],
  ]
    .map(([name, value]) => `<div><dt>${name}</dt><dd>${value}</dd></div>`)
    .join("");
}

function paneHeight(width, ratio) {
  const maxH = Math.min(window.innerHeight * 0.48, 520);
  return Math.min(maxH, Math.max(160, Math.round(width * ratio)));
}

function sizeCropPanes(img, item) {
  const origPane = $("cropOrigPane");
  const afterPane = $("cropAfterPane");
  if (!origPane || !img || !img.naturalWidth) return;
  const width = origPane.clientWidth || 480;
  origPane.style.height = `${paneHeight(width, img.naturalHeight / img.naturalWidth)}px`;
  if (afterPane && item) {
    const m = cropMetrics(item);
    const ratio = m.cropW && m.cropH ? m.cropH / m.cropW : img.naturalHeight / img.naturalWidth;
    afterPane.style.height = `${paneHeight(width, ratio)}px`;
  }
  void origPane.offsetHeight;
}

function placeCropFrame() {
  const item = selectedCropItem();
  const img = $("cropOrigImg");
  const layer = $("cropFrameLayer");
  const frame = $("cropFrame");
  if (!item || !img || !layer || !frame || !img.naturalWidth) return;
  crop.fit = containedRect(img);
  layer.style.left = `${crop.fit.x}px`;
  layer.style.top = `${crop.fit.y}px`;
  layer.style.width = `${crop.fit.w}px`;
  layer.style.height = `${crop.fit.h}px`;
  layer.style.right = "auto";
  layer.style.bottom = "auto";
  const [left, top, right, bottom] = cropBoxOf(item);
  frame.hidden = false;
  frame.style.left = `${(left / item.width) * 100}%`;
  frame.style.top = `${(top / item.height) * 100}%`;
  frame.style.width = `${((right - left) / item.width) * 100}%`;
  frame.style.height = `${((bottom - top) / item.height) * 100}%`;
}

function drawAfterPreview() {
  const item = selectedCropItem();
  const img = crop.compareImg || $("cropOrigImg");
  const pane = $("cropAfterPane");
  const canvas = $("cropAfterCanvas");
  if (!item || !img || !img.naturalWidth || !pane || !canvas) return;
  const cssW = pane.clientWidth || 1;
  const cssH = pane.clientHeight || 1;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(cssW * dpr));
  canvas.height = Math.max(1, Math.round(cssH * dpr));
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#161310";
  ctx.fillRect(0, 0, cssW, cssH);
  const [left, top, right, bottom] = boxToDisplay(item, img, cropBoxOf(item));
  const sw = Math.max(1, right - left);
  const sh = Math.max(1, bottom - top);
  ctx.drawImage(img, left, top, sw, sh, 0, 0, cssW, cssH);
}

function drawCropCompare() {
  const item = selectedCropItem();
  const img = crop.compareImg || $("cropOrigImg");
  if (!item || !img || !img.naturalWidth) return;
  $("cropCompareEmpty").hidden = true;
  $("cropCompareBody").hidden = false;
  sizeCropPanes(img, item);
  placeCropFrame();
  drawAfterPreview();
  updateCropCompareMeta();
}

function cropPreviewUrl(item) {
  return `/api/preview/${item.id}?job_id=${crop.jobId}&w=1600`;
}

function selectCropItem(id) {
  const item = crop.items.find((it) => it.id === id);
  if (!item) return;
  crop.selectedId = id;
  renderCropGrid();
  $("cropCompareEmpty").hidden = true;
  $("cropCompareBody").hidden = false;
  updateCropCompareMeta();
  const img = $("cropOrigImg");
  const url = cropPreviewUrl(item);
  if (crop.compareItemId === id && img.complete && img.naturalWidth && img.src.includes(item.id)) {
    crop.compareImg = img;
    drawCropCompare();
    return;
  }
  crop.compareItemId = id;
  img.onload = () => {
    if (crop.selectedId !== id) return;
    crop.compareImg = img;
    drawCropCompare();
  };
  if (img.src !== new URL(url, window.location.origin).href) img.src = url;
  else if (img.complete) img.onload();
}

function stepCropItem(delta) {
  const items = visibleCropItems();
  if (!items.length) return;
  const idx = Math.max(0, items.findIndex((it) => it.id === crop.selectedId));
  const next = items[(idx + delta + items.length) % items.length];
  selectCropItem(next.id);
}

function persistCropBox(item) {
  if (!crop.jobId || !item) return;
  crop.edits[item.id] = cropBoxOf(item);
  clearTimeout(crop.saveTimer);
  crop.saveTimer = setTimeout(async () => {
    try {
      await fetch(`/api/jobs/${crop.jobId}/crop/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ box: cropBoxOf(item) }),
      });
    } catch (_) {
      /* 另存时会再带一次框 */
    }
  }, 280);
}

function commitCropBox(box, edited) {
  const item = selectedCropItem();
  if (!item) return;
  applyBoxLocal(item, box, edited);
  crop.edits[item.id] = cropBoxOf(item);
  drawCropCompare();
  renderCropStats();
  renderCropChart();
  renderCropGrid();
  persistCropBox(item);
}

function imagePoint(clientX, clientY) {
  const item = selectedCropItem();
  const layer = $("cropFrameLayer");
  if (!item || !layer) return null;
  const rect = layer.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return null;
  return {
    x: Math.min(item.width, Math.max(0, ((clientX - rect.left) / rect.width) * item.width)),
    y: Math.min(item.height, Math.max(0, ((clientY - rect.top) / rect.height) * item.height)),
  };
}

function minCropSize(item) {
  return Math.max(8, Math.round(Math.min(item.width, item.height) * 0.01));
}

function resizeBox(box, handle, point, item) {
  const min = minCropSize(item);
  let [left, top, right, bottom] = box;
  if (handle.includes("w")) left = Math.min(point.x, right - min);
  if (handle.includes("e")) right = Math.max(point.x, left + min);
  if (handle.includes("n")) top = Math.min(point.y, bottom - min);
  if (handle.includes("s")) bottom = Math.max(point.y, top + min);
  return [left, top, right, bottom];
}

function onCropPointerDown(event) {
  const item = selectedCropItem();
  if (!item || event.button) return;
  const handle = event.target.dataset && event.target.dataset.h;
  const point = imagePoint(event.clientX, event.clientY);
  if (!point) return;
  event.preventDefault();
  if (handle) {
    crop.drag = { mode: "resize", handle, start: point, box: cropBoxOf(item) };
  } else if (event.target.id === "cropFrame") {
    crop.drag = { mode: "move", start: point, box: cropBoxOf(item) };
  } else {
    crop.drag = { mode: "draw", start: point, box: [point.x, point.y, point.x, point.y] };
  }
  $("cropOrigPane").setPointerCapture(event.pointerId);
}

function onCropPointerMove(event) {
  if (!crop.drag) return;
  const item = selectedCropItem();
  const point = imagePoint(event.clientX, event.clientY);
  if (!item || !point) return;
  const { mode, handle, start, box } = crop.drag;
  if (mode === "draw") {
    const min = minCropSize(item);
    const left = Math.min(start.x, point.x);
    const top = Math.min(start.y, point.y);
    const right = Math.max(start.x, point.x);
    const bottom = Math.max(start.y, point.y);
    applyBoxLocal(item, [left, top, Math.max(right, left + min), Math.max(bottom, top + min)], true);
  } else if (mode === "move") {
    const [left, top, right, bottom] = box;
    let dx = point.x - start.x;
    let dy = point.y - start.y;
    dx = Math.min(item.width - right, Math.max(-left, dx));
    dy = Math.min(item.height - bottom, Math.max(-top, dy));
    applyBoxLocal(item, [left + dx, top + dy, right + dx, bottom + dy], true);
  } else {
    applyBoxLocal(item, resizeBox(box, handle, point, item), true);
  }
  const afterPane = $("cropAfterPane");
  const origPane = $("cropOrigPane");
  if (afterPane && origPane) {
    const m = cropMetrics(item);
    const ratio = m.cropW && m.cropH ? m.cropH / m.cropW : 1;
    afterPane.style.height = `${paneHeight(origPane.clientWidth || 480, ratio)}px`;
  }
  placeCropFrame();
  drawAfterPreview();
  updateCropCompareMeta();
}

function onCropPointerUp() {
  if (!crop.drag) return;
  crop.drag = null;
  const item = selectedCropItem();
  if (!item) return;
  crop.edits[item.id] = cropBoxOf(item);
  persistCropBox(item);
  renderCropStats();
  renderCropChart();
  renderCropGrid();
}

function resetSelectedCrop() {
  const item = selectedCropItem();
  if (!item) return;
  const auto = item.auto_crop_box || [0, 0, item.width, item.height];
  applyBoxLocal(item, auto, false);
  delete crop.edits[item.id];
  drawCropCompare();
  renderCrop();
  if (!crop.jobId) return;
  fetch(`/api/jobs/${crop.jobId}/crop/${item.id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reset: true }),
  }).catch(() => {});
}

function renderCrop() {
  renderCropStats();
  renderCropChart();
  renderCropGrid();
  restoreCropExportButton();
  setCropScanControls();
  if (crop.selectedId && selectedCropItem()) {
    $("cropCompareEmpty").hidden = true;
    $("cropCompareBody").hidden = false;
    if (crop.compareImg && crop.compareItemId === crop.selectedId) drawCropCompare();
    else updateCropCompareMeta();
  } else {
    $("cropCompareEmpty").hidden = false;
    $("cropCompareBody").hidden = true;
  }
}

function applyCropJob(job) {
  const prevSelected = crop.selectedId;
  crop.jobId = job.id;
  crop.items = (job.items || []).map((it) => {
    if (!it.auto_crop_box) it.auto_crop_box = cropBoxOf(it).slice();
    const edit = crop.edits[it.id];
    return edit && !sameBox(edit, cropBoxOf(it)) ? applyBoxLocal(it, edit, true) : it;
  });
  crop.running = job.status === "running" || job.status === "queued" || job.status === "listing";
  if (job.status === "listing") {
    $("cropProgressText").textContent = job.total
      ? `正在列出文件，已找到 ${job.total} 张`
      : "正在列出文件…";
  } else if (job.status === "running" || job.status === "queued") {
    let text = `正在看边 ${job.processed || 0} / ${job.total || "?"}`;
    if (job.processed && job.started_at) {
      const sec = (Date.now() / 1000 - job.started_at) / job.processed;
      text += ` · ${sec.toFixed(1)} 秒/张`;
    }
    if (job.current) text += ` · ${job.current}`;
    $("cropProgressText").textContent = text;
  } else if (job.status === "done") {
    $("cropProgressText").textContent = `完成 ${job.processed} 张`;
  } else if (job.status === "stopped") {
    $("cropProgressText").textContent = `已停止 ${job.processed} / ${job.total || job.processed}`;
  } else if (job.status === "error") {
    $("cropProgressText").textContent = job.error || "分析失败";
  }
  if (prevSelected && crop.items.some((it) => it.id === prevSelected)) {
    crop.selectedId = prevSelected;
  } else if (!crop.selectedId && crop.items.length) {
    const first = crop.items.find(cropNeed) || crop.items[0];
    crop.selectedId = first.id;
  }
  renderCrop();
  if (crop.selectedId && (!crop.compareImg || crop.compareItemId !== crop.selectedId)) {
    selectCropItem(crop.selectedId);
  }
}

function stopCropPolling() {
  if (crop.pollTimer) {
    clearInterval(crop.pollTimer);
    crop.pollTimer = null;
  }
}

async function pollCropJob() {
  if (!crop.jobId) return;
  const res = await fetch(`/api/jobs/${crop.jobId}`);
  const job = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("cropProgressText").textContent = res.status === 404 ? "任务已丢失，请重新分析" : "读取进度失败";
    stopCropPolling();
    crop.running = false;
    setCropScanControls();
    return;
  }
  applyCropJob(job);
  if (job.status === "done" || job.status === "stopped" || job.status === "error") {
    stopCropPolling();
    crop.running = false;
    setCropScanControls();
  }
}

async function startCropScan() {
  const folder = $("folder").value.trim();
  if (!folder) {
    $("cropScanHint").textContent = "请填写本地文件夹路径。";
    return;
  }
  if (crop.running) await stopCropScan();
  stopCropPolling();
  crop.running = true;
  crop.selectedId = null;
  crop.edits = {};
  crop.compareImg = null;
  crop.compareItemId = null;
  $("cropStopBtn").disabled = false;
  $("cropExportHint").textContent = "";
  $("cropScanHint").textContent = "正在查找黑边白边…";
  const res = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      folder,
      recursive: $("recursive").checked,
      kind: "crop",
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "扫描失败" }));
    $("cropScanHint").textContent = err.detail || "扫描失败";
    crop.running = false;
    setCropScanControls();
    return;
  }
  const data = await res.json();
  crop.jobId = data.job_id;
  crop.items = [];
  crop.running = true;
  $("cropScanHint").textContent = "去边分析进行中，可随时停止。点图可对比并改框。";
  renderCrop();
  crop.pollTimer = setInterval(pollCropJob, 400);
  pollCropJob();
}

async function stopCropScan() {
  if (!crop.jobId || !crop.running) return;
  $("cropStopBtn").disabled = true;
  await fetch(`/api/jobs/${crop.jobId}/stop`, { method: "POST" });
  $("cropScanHint").textContent = "已请求停止。";
  await pollCropJob();
}

function cropBoxPayload() {
  const boxes = {};
  crop.items.forEach((item) => {
    if (crop.edits[item.id] || item.edited) boxes[item.id] = cropBoxOf(item);
  });
  return boxes;
}

async function exportCrop() {
  if (!crop.jobId) {
    $("cropExportHint").textContent = "请先分析完再另存。";
    return;
  }
  $("exportCropBtn").disabled = true;
  $("cropExportHint").textContent = "正在按原画质写入新文件夹…";
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: crop.jobId,
        threshold: 0,
        dest: $("destCrop").value.trim() || null,
        dest_uncropped: $("destCropSkip").value.trim() || null,
        kind: "crop",
        crop_boxes: cropBoxPayload(),
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("cropExportHint").textContent = errorDetail(data, `保存失败（${res.status}）`);
      return;
    }
    const extra = data.failed ? `，失败 ${data.failed} 张` : "";
    $("cropExportHint").textContent =
      `已写入 ${data.count} 张：裁切 ${data.cropped || 0} 张到 ${data.dest}，无需裁剪 ${data.copied || 0} 张到 ${data.dest_uncropped || "—"}${extra}`;
    const jobRes = await fetch(`/api/jobs/${crop.jobId}`);
    if (jobRes.ok) applyCropJob(await jobRes.json());
  } catch (err) {
    $("cropExportHint").textContent = `保存失败：${err.message || err}`;
  } finally {
    restoreCropExportButton();
  }
}

window.renderCrop = renderCrop;

$("cropScanBtn").addEventListener("click", startCropScan);
$("cropStopBtn").addEventListener("click", stopCropScan);
$("exportCropBtn").addEventListener("click", exportCrop);
$("cropPrevBtn").addEventListener("click", () => stepCropItem(-1));
$("cropNextBtn").addEventListener("click", () => stepCropItem(1));
$("cropResetBtn").addEventListener("click", resetSelectedCrop);
$("cropOrigPane").addEventListener("pointerdown", onCropPointerDown);
$("cropOrigPane").addEventListener("pointermove", onCropPointerMove);
$("cropOrigPane").addEventListener("pointerup", onCropPointerUp);
$("cropOrigPane").addEventListener("pointercancel", onCropPointerUp);
document.querySelectorAll("#cropStage .tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll("#cropStage .tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    crop.filter = tab.dataset.cropfilter;
    renderCropGrid();
    const still = visibleCropItems().some((it) => it.id === crop.selectedId);
    if (!still && visibleCropItems()[0]) selectCropItem(visibleCropItems()[0].id);
  });
});
window.addEventListener("resize", () => {
  if ($("cropStage").hidden) return;
  renderCropChart();
  if (crop.compareImg) drawCropCompare();
  $("cropGrid").querySelectorAll(".card").forEach(layoutCardCrop);
});
document.addEventListener("keydown", (event) => {
  if ($("cropStage").hidden) return;
  if (event.target.closest("input, textarea, select, dialog")) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    stepCropItem(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    stepCropItem(1);
  }
});
renderCrop();
