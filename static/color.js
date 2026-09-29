const color = {
  jobId: null,
  items: [],
  filter: "all",
  pollTimer: null,
  running: false,
};

function overTh() {
  return Number($("overThreshold").value) / 100;
}

function underTh() {
  return Number($("underThreshold").value) / 100;
}

function neonTh() {
  return Number($("neonThreshold").value) / 100;
}

function brightRatio(item) {
  if (item.bright_ratio != null) return Number(item.bright_ratio || 0);
  return Math.max(Number(item.white_ratio || 0), Number(item.over_ratio || 0));
}

function darkRatio(item) {
  if (item.dark_ratio != null) return Number(item.dark_ratio || 0);
  return Math.max(Number(item.black_ratio || 0), Number(item.under_ratio || 0));
}

const MAX_PASS_COLORS = 3;

function colorCount(item) {
  return (item.colors || []).length;
}

function harmonyOk(item) {
  if ((item.harmony_detail || "") === "无法读取图片") return false;
  if ((item.harmony || "none") !== "none") return true;
  return colorCount(item) <= MAX_PASS_COLORS;
}

function colorReasons(item) {
  const reasons = [];
  if (brightRatio(item) >= overTh()) reasons.push("over");
  if (darkRatio(item) >= underTh()) reasons.push("under");
  if (Number(item.neon_ratio || 0) >= neonTh()) reasons.push("neon");
  if (!harmonyOk(item)) reasons.push("harmony");
  return reasons;
}

function colorFails(item) {
  return colorReasons(item).length > 0;
}

function fmtPct(ratio) {
  return `${(Number(ratio || 0) * 100).toFixed(1)}%`;
}

function bindPair(rangeId, numId) {
  const range = $(rangeId);
  const num = $(numId);
  const min = Number(range.min);
  const max = Number(range.max);
  range.addEventListener("input", () => {
    num.value = range.value;
    renderColor();
  });
  num.addEventListener("input", () => {
    const raw = Number(num.value);
    if (Number.isNaN(raw)) return;
    const value = Math.min(max, Math.max(min, raw));
    range.value = String(value);
    renderColor();
  });
}

function updateColorLabels() {
  const over = Number($("overThreshold").value);
  const under = Number($("underThreshold").value);
  const neon = Number($("neonThreshold").value);
  $("overLabel").textContent = `${over.toFixed(1)}%`;
  $("underLabel").textContent = `${under.toFixed(1)}%`;
  $("neonLabel").textContent = `${neon.toFixed(1)}%`;
  if ($("overNum") && Number($("overNum").value) !== over) $("overNum").value = String(over);
  if ($("underNum") && Number($("underNum").value) !== under) $("underNum").value = String(under);
  if ($("neonNum") && Number($("neonNum").value) !== neon) $("neonNum").value = String(neon);
}

function visibleColorItems() {
  if (color.filter === "fail") return color.items.filter(colorFails);
  if (color.filter === "ok") return color.items.filter((it) => !colorFails(it));
  return color.items;
}

function renderColorStats() {
  const items = color.items;
  const fails = items.filter(colorFails);
  const harmonyFails = items.filter((it) => !harmonyOk(it));
  $("colorStatTotal").textContent = items.length || "—";
  $("colorStatFail").textContent = items.length ? fails.length : "—";
  $("colorStatOk").textContent = items.length ? items.length - fails.length : "—";
  $("colorStatHarmony").textContent = items.length ? harmonyFails.length : "—";
}

function renderColorChart() {
  const canvas = $("colorHist");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const height = 168;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const rows = [
    ["过曝", color.items.filter((it) => brightRatio(it) >= overTh()).length],
    ["欠曝", color.items.filter((it) => darkRatio(it) >= underTh()).length],
    ["高饱和纯色", color.items.filter((it) => Number(it.neon_ratio || 0) >= neonTh()).length],
    ["配色不达标", color.items.filter((it) => !harmonyOk(it)).length],
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

function reasonBadge(item) {
  const tags = [];
  if (brightRatio(item) >= overTh()) tags.push(`过曝 ${fmtPct(brightRatio(item))}`);
  if (darkRatio(item) >= underTh()) tags.push(`欠曝 ${fmtPct(darkRatio(item))}`);
  if (Number(item.neon_ratio || 0) >= neonTh()) tags.push(`高饱和纯色 ${fmtPct(item.neon_ratio)}`);
  if (!harmonyOk(item)) tags.push(`配色不达标 · 主色 ${colorCount(item)} 块`);
  else if ((item.harmony || "none") === "none") tags.push(`主色 ${colorCount(item)} 块，通过`);
  else tags.push(item.harmony_name || "配色合格");
  return tags.join(" · ");
}

function renderColorGrid() {
  const items = visibleColorItems();
  $("colorGalleryCount").textContent = color.items.length
    ? `当前视图 ${items.length} 张`
    : "还没有色彩分析结果";
  $("colorGrid").innerHTML = items
    .map((item) => {
      const fail = colorFails(item);
      const swatches = (item.colors || [])
        .slice(0, 4)
        .map((c) => `<i class="swatch" style="background:${c.hex}" title="${c.hex}"></i>`)
        .join("");
      return `
        <article class="card ${fail ? "is-text" : ""}" data-id="${item.id}">
          <div class="thumb" style="background-image:url('/api/thumb/${item.id}?job_id=${color.jobId}&w=420')"></div>
          <div class="meta">
            <strong title="${item.rel}">${item.name}</strong>
            <span class="badge ${fail ? "hot" : ""}">${reasonBadge(item)}</span>
            <div class="swatches">${swatches}</div>
          </div>
        </article>`;
    })
    .join("");
  $("colorGrid").querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => openColorLightbox(card.dataset.id));
  });
}

function setColorScanControls() {
  $("colorScanBtn").disabled = false;
  $("colorScanBtn").textContent = color.items.length || color.running ? "重新分析" : "开始分析";
  $("colorStopBtn").disabled = !color.running;
}

function renderColor() {
  updateColorLabels();
  renderColorStats();
  renderColorChart();
  renderColorGrid();
  restoreColorExportButtons();
  setColorScanControls();
}

function restoreColorExportButtons() {
  $("exportColorBtn").disabled = !color.items.some(colorFails);
  $("exportColorPassBtn").disabled = !color.items.some((it) => !colorFails(it));
}

function openColorLightbox(id) {
  const item = color.items.find((it) => it.id === id);
  if (!item) return;
  const img = new Image();
  img.onload = () => {
    const canvas = $("preview");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
  };
  img.src = `/api/image/${item.id}?job_id=${color.jobId}`;
  $("lbName").textContent = item.name;
  const status = colorFails(item) ? "不达标，将移到 fail" : "合格，将移到 pass";
  const colors = (item.colors || []).map((c) => c.hex).join("  ");
  $("lbMeta").textContent =
    `${item.width}×${item.height}  ·  ${status}\n` +
    `过曝 ${fmtPct(brightRatio(item))}（只计失去细节；固有白 ${fmtPct(item.white_ratio)} 已忽略）\n` +
    `欠曝 ${fmtPct(darkRatio(item))}（只计失去细节；固有黑 ${fmtPct(item.black_ratio)} 已忽略）\n` +
    `高饱和纯色 ${fmtPct(item.neon_ratio)}\n` +
    (harmonyOk(item) && (item.harmony || "none") === "none"
      ? `主色 ${colorCount(item)} 块（不超过3块），配色通过\n`
      : `${item.harmony_name}：${item.harmony_detail || ""}\n`) +
    (colors ? `主色 ${colors}` : "");
  $("lightbox").showModal();
}

function applyColorJob(job) {
  color.jobId = job.id;
  color.items = job.items || [];
  color.running = job.status === "running" || job.status === "queued" || job.status === "listing";
  if (job.status === "listing") {
    $("colorProgressText").textContent = job.total
      ? `正在列出文件，已找到 ${job.total} 张（还没开始分析）`
      : "正在列出文件，只收集路径，尚未分析…";
  } else if (job.status === "running" || job.status === "queued") {
    const processed = job.processed || 0;
    const total = job.total || "?";
    let text = `正在分析 ${processed} / ${total}`;
    if (processed && job.started_at) {
      const sec = (Date.now() / 1000 - job.started_at) / processed;
      text += ` · ${sec.toFixed(1)} 秒/张`;
    }
    if (job.current) text += ` · ${job.current}`;
    $("colorProgressText").textContent = text;
  } else if (job.status === "done") {
    $("colorProgressText").textContent = `完成 ${job.processed} 张`;
  } else if (job.status === "stopped") {
    $("colorProgressText").textContent = `已停止 ${job.processed} / ${job.total || job.processed}`;
  }
  renderColor();
}

function stopColorPolling() {
  if (color.pollTimer) {
    clearInterval(color.pollTimer);
    color.pollTimer = null;
  }
}

async function pollColorJob() {
  if (!color.jobId) return;
  const res = await fetch(`/api/jobs/${color.jobId}`);
  const job = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("colorProgressText").textContent =
      res.status === 404 ? "任务已丢失，请重新分析" : "读取进度失败，请重新分析";
    stopColorPolling();
    color.running = false;
    setColorScanControls();
    return;
  }
  applyColorJob(job);
  if (job.status === "done" || job.status === "stopped") {
    stopColorPolling();
    color.running = false;
    setColorScanControls();
  }
}

async function stopColorScan() {
  if (!color.jobId || !color.running) return;
  $("colorStopBtn").disabled = true;
  await fetch(`/api/jobs/${color.jobId}/stop`, { method: "POST" });
  $("colorScanHint").textContent = "已请求停止。之后可改阈值再重新分析。";
  await pollColorJob();
}

async function startColorScan() {
  const folder = $("folder").value.trim();
  if (!folder) {
    $("colorScanHint").textContent = "请填写本地文件夹路径。";
    return;
  }
  if (color.running) await stopColorScan();
  stopColorPolling();
  color.running = true;
  $("colorStopBtn").disabled = false;
  $("colorExportHint").textContent = "";
  $("colorScanHint").textContent = "正在分析色彩…";
  const res = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      folder,
      recursive: $("recursive").checked,
      kind: "color",
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "扫描失败" }));
    $("colorScanHint").textContent = err.detail || "扫描失败";
    color.running = false;
    setColorScanControls();
    return;
  }
  const data = await res.json();
  color.jobId = data.job_id;
  color.items = [];
  color.running = true;
  $("colorScanHint").textContent = "色彩分析进行中，可随时停止。";
  renderColor();
  color.pollTimer = setInterval(pollColorJob, 400);
  pollColorJob();
}

async function exportColor(kind) {
  if (!color.jobId) {
    $("colorExportHint").textContent = "请先分析完再移动。";
    return;
  }
  const passing = kind === "color_pass";
  $("exportColorBtn").disabled = true;
  $("exportColorPassBtn").disabled = true;
  $("colorExportHint").textContent = passing
    ? "正在挪走合格图片并写 JSON…"
    : "正在挪走不达标图片并写 JSON…";
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: color.jobId,
        threshold: 0,
        over_threshold: overTh(),
        under_threshold: underTh(),
        neon_threshold: neonTh(),
        dest: $(passing ? "destColorPass" : "destColor").value.trim() || null,
        kind,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("colorExportHint").textContent = errorDetail(data, `移动失败（${res.status}）`);
      return;
    }
    const extra = data.skipped ? `，另有 ${data.skipped} 张已在目标目录` : "";
    $("colorExportHint").textContent = `已挪走 ${data.count} 张到 ${data.dest}${extra}`;
    const jobRes = await fetch(`/api/jobs/${color.jobId}`);
    if (jobRes.ok) applyColorJob(await jobRes.json());
  } catch (err) {
    $("colorExportHint").textContent = `移动失败：${err.message || err}`;
  } finally {
    restoreColorExportButtons();
  }
}

window.renderColor = renderColor;

bindPair("overThreshold", "overNum");
bindPair("underThreshold", "underNum");
bindPair("neonThreshold", "neonNum");
$("colorScanBtn").addEventListener("click", startColorScan);
$("colorStopBtn").addEventListener("click", stopColorScan);
$("exportColorBtn").addEventListener("click", () => exportColor("color_fail"));
$("exportColorPassBtn").addEventListener("click", () => exportColor("color_pass"));
document.querySelectorAll("#colorStage .tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll("#colorStage .tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    color.filter = tab.dataset.cfilter;
    renderColorGrid();
  });
});
window.addEventListener("resize", () => {
  if (!$("colorStage").hidden) renderColorChart();
});
renderColor();
