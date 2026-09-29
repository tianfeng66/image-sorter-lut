const state = {
  jobId: null,
  items: [],
  filter: "all",
  pollTimer: null,
  running: false,
};

const $ = (id) => document.getElementById(id);

function thresholdValue() {
  return Number($("threshold").value) / 100;
}

function classified(item) {
  return item.text_ratio >= thresholdValue();
}

function fmtPct(ratio) {
  return `${(ratio * 100).toFixed(1)}%`;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function updateThresholdLabel() {
  const value = Number($("threshold").value);
  $("thresholdLabel").textContent = `${value.toFixed(1)}%`;
  if ($("thresholdNum") && Number($("thresholdNum").value) !== value) {
    $("thresholdNum").value = String(value);
  }
}

function renderStats() {
  const items = state.items;
  const textItems = items.filter(classified);
  const ratios = items.map((it) => it.text_ratio);
  const avg = ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0;
  $("statTotal").textContent = items.length || "—";
  $("statText").textContent = items.length ? textItems.length : "—";
  $("statRest").textContent = items.length ? items.length - textItems.length : "—";
  $("statAvg").textContent = items.length ? fmtPct(avg) : "—";
  $("statMed").textContent = items.length ? fmtPct(median(ratios)) : "—";
}

function renderHistogram() {
  const canvas = $("hist");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const height = 120;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const bins = 10;
  const counts = new Array(bins).fill(0);
  for (const item of state.items) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(item.text_ratio * bins)));
    counts[idx] += 1;
  }
  const max = Math.max(1, ...counts);
  const gap = 8;
  const barW = (width - gap * (bins + 1)) / bins;
  const thresholdBin = Math.min(bins - 1, Math.floor(thresholdValue() * bins));

  counts.forEach((count, i) => {
    const h = (count / max) * (height - 18);
    const x = gap + i * (barW + gap);
    const y = height - h - 2;
    ctx.fillStyle = i >= thresholdBin ? "#b5431f" : "#c9b8a4";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, barW, h, 6);
    else ctx.rect(x, y, barW, h);
    ctx.fill();
  });

  $("histAxis").innerHTML = `<span>0%</span><span>当前阈值 ${fmtPct(thresholdValue())}</span><span>100%</span>`;
}

function visibleItems() {
  if (state.filter === "text") return state.items.filter(classified);
  if (state.filter === "rest") return state.items.filter((it) => !classified(it));
  return state.items;
}

function renderGrid() {
  const items = visibleItems();
  $("galleryCount").textContent = state.items.length
    ? `当前视图 ${items.length} 张`
    : "还没有分析结果";
  $("grid").innerHTML = items
    .map((item) => {
      const hot = classified(item);
      return `
        <article class="card ${hot ? "is-text" : ""}" data-id="${item.id}">
          <div class="thumb" style="background-image:url('/api/thumb/${item.id}?job_id=${state.jobId}&w=420')"></div>
          <div class="meta">
            <strong title="${item.rel}">${item.name}</strong>
            <span class="badge ${hot ? "hot" : ""}">${fmtPct(item.text_ratio)} · ${item.box_count} 处文字</span>
          </div>
        </article>`;
    })
    .join("");

  $("grid").querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => openLightbox(card.dataset.id));
  });
}

function setScanControls() {
  $("scanBtn").disabled = false;
  $("scanBtn").textContent = state.items.length || state.running ? "重新分析" : "开始分析";
  $("stopBtn").disabled = !state.running;
}

function renderAll() {
  updateThresholdLabel();
  renderStats();
  renderHistogram();
  renderGrid();
  $("exportTextBtn").disabled = !state.items.some(classified);
  $("exportRestBtn").disabled = !state.items.some((it) => !classified(it));
  setScanControls();
}

function openLightbox(id) {
  const item = state.items.find((it) => it.id === id);
  if (!item) return;
  const img = new Image();
  img.onload = () => {
    const canvas = $("preview");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.strokeStyle = "rgba(181, 67, 31, 0.95)";
    ctx.lineWidth = Math.max(2, img.naturalWidth / 280);
    for (const [x1, y1, x2, y2] of item.boxes || []) {
      ctx.strokeRect(
        x1 * img.naturalWidth,
        y1 * img.naturalHeight,
        (x2 - x1) * img.naturalWidth,
        (y2 - y1) * img.naturalHeight
      );
    }
  };
  img.src = `/api/image/${item.id}?job_id=${state.jobId}`;
  $("lbName").textContent = item.name;
  const snippets = (item.texts || []).slice(0, 6).join("  ·  ");
  $("lbMeta").textContent = `${item.width}×${item.height}  ·  文字占比 ${fmtPct(item.text_ratio)}\n${snippets || "未识别到可读文本"}`;
  $("lightbox").showModal();
}

async function warmup() {
  try {
    const res = await fetch("/api/warmup", { method: "POST" });
    const data = await res.json();
    $("scanHint").textContent = data.ready
      ? `检测引擎已就绪（${data.engine}）。选择文件夹后开始分析。`
      : `将使用备用检测（${data.engine}）。${data.error || ""}`;
  } catch {
    $("scanHint").textContent = "模型预热失败，仍可尝试扫描。";
  }
}

async function loadDefaults() {
  const res = await fetch("/api/defaults");
  const data = await res.json();
  if (data.folder) $("folder").value = data.folder;
}

function applyJob(job) {
  state.jobId = job.id;
  state.items = job.items || [];
  state.running = job.status === "running" || job.status === "queued" || job.status === "listing";
  if (job.status === "listing") {
    $("progressText").textContent = job.total
      ? `正在列出文件，已找到 ${job.total} 张（还没开始分析）`
      : "正在列出文件，只收集路径，尚未分析…";
  } else if (job.status === "running" || job.status === "queued") {
    $("progressText").textContent = _paceText("正在分析", job);
  } else if (job.status === "done") {
    $("progressText").textContent = `完成 ${job.processed} 张`;
  } else if (job.status === "stopped") {
    $("progressText").textContent = `已停止 ${job.processed} / ${job.total || job.processed}`;
  }
  renderAll();
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollJob() {
  if (!state.jobId) return;
  const res = await fetch(`/api/jobs/${state.jobId}`);
  const job = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("progressText").textContent =
      res.status === 404 ? "任务已丢失，请重新分析" : "读取进度失败，请重新分析";
    stopPolling();
    state.running = false;
    setScanControls();
    return;
  }
  applyJob(job);
  if (job.status === "done" || job.status === "stopped") {
    stopPolling();
    state.running = false;
    setScanControls();
  }
}

async function stopScan() {
  if (!state.jobId || !state.running) return;
  $("stopBtn").disabled = true;
  await fetch(`/api/jobs/${state.jobId}/stop`, { method: "POST" });
  $("scanHint").textContent = "已请求停止，当前这张处理完就会停下。之后可改阈值再重新分析。";
  await pollJob();
}

async function startScan() {
  const folder = $("folder").value.trim();
  if (!folder) {
    $("scanHint").textContent = "请填写本地文件夹路径。";
    return;
  }
  if (state.running) {
    await stopScan();
  }
  stopPolling();
  state.running = true;
  $("scanBtn").disabled = false;
  $("scanBtn").textContent = "重新分析";
  $("stopBtn").disabled = false;
  $("exportHint").textContent = "";
  $("scanHint").textContent = "正在按当前阈值准备分析…";
  warmup();
  const res = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      folder,
      recursive: $("recursive").checked,
      kind: "text",
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "扫描失败" }));
    $("scanHint").textContent = err.detail || "扫描失败";
    state.running = false;
    setScanControls();
    return;
  }
  const data = await res.json();
  state.jobId = data.job_id;
  state.items = [];
  state.running = true;
  $("scanHint").textContent = "分析进行中，可随时停止，或改完阈值后点「重新分析」。";
  renderAll();
  state.pollTimer = setInterval(pollJob, 400);
  pollJob();
}

function restoreExportButtons() {
  $("exportTextBtn").disabled = !state.items.some(classified);
  $("exportRestBtn").disabled = !state.items.some((it) => !classified(it));
}

function errorDetail(data, fallback) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("；");
  return fallback;
}

async function exportImages(kind) {
  if (!state.jobId) {
    $("exportHint").textContent = "请先分析完再移动。";
    return;
  }
  const destInput = kind === "text" ? $("destText") : $("destRest");
  $("exportTextBtn").disabled = true;
  $("exportRestBtn").disabled = true;
  $("exportHint").textContent = "正在从原文件夹挪走…";
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: state.jobId,
        threshold: thresholdValue(),
        dest: destInput.value.trim() || null,
        kind,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("exportHint").textContent = errorDetail(data, `移动失败（${res.status}）`);
      return;
    }
    const label = kind === "text" ? "含文字" : "不含文字";
    const extra = data.skipped ? `，另有 ${data.skipped} 张已在目标目录` : "";
    $("exportHint").textContent = `已挪走 ${data.count} 张${label}图片到 ${data.dest}${extra}`;
    const jobRes = await fetch(`/api/jobs/${state.jobId}`);
    if (jobRes.ok) applyJob(await jobRes.json());
  } catch (err) {
    $("exportHint").textContent = `移动失败：${err.message || err}`;
  } finally {
    restoreExportButtons();
  }
}

$("threshold").addEventListener("input", renderAll);
$("thresholdNum").addEventListener("input", () => {
  const raw = Number($("thresholdNum").value);
  if (Number.isNaN(raw)) return;
  const value = Math.min(40, Math.max(0, raw));
  $("threshold").value = String(value);
  renderAll();
});
$("scanBtn").addEventListener("click", startScan);
$("stopBtn").addEventListener("click", stopScan);
$("exportTextBtn").addEventListener("click", () => exportImages("text"));
$("exportRestBtn").addEventListener("click", () => exportImages("rest"));
$("closeLb").addEventListener("click", () => $("lightbox").close());
$("lightbox").addEventListener("click", (event) => {
  if (event.target === $("lightbox")) $("lightbox").close();
});
document.querySelectorAll("#textStage .tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll("#textStage .tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    state.filter = tab.dataset.filter;
    renderGrid();
  });
});
window.addEventListener("resize", renderHistogram);

document.querySelectorAll(".task").forEach((btn) => {
  btn.addEventListener("click", () => {
    const task = btn.dataset.task;
    document.querySelectorAll(".task").forEach((t) => t.classList.toggle("active", t === btn));
    const panels = [
      ["textControls", "textStage", "text"],
      ["colorControls", "colorStage", "color"],
      ["compControls", "compStage", "composition"],
      ["cropControls", "cropStage", "crop"],
      ["lutControls", "lutStage", "lut"],
    ];
    panels.forEach(([ctrl, stage, id]) => {
      const on = task === id;
      $(ctrl).hidden = !on;
      $(stage).hidden = !on;
      $(ctrl).inert = !on;
      $(stage).inert = !on;
    });
    const meta = {
      text: ["Aa", "文字分拣", "按面积占比挑出含字图片"],
      color: ["色", "色彩分析", "过曝欠曝忽略固有色；纯色、配色不达标则移出"],
      composition: ["构", "构图分析", "先只做中心构图，可加参考图和定义"],
      crop: ["裁", "去边裁剪", "对比原图和裁切结果，可拖框或框选后另存"],
      lut: ["LUT", "LUT 调色", "可叠多层 LUT，每层单独调强度"],
    }[task];
    $("brandMark").textContent = meta[0];
    $("brandTitle").textContent = meta[1];
    $("brandSub").textContent = meta[2];
    if (task === "color" && window.renderColor) window.renderColor();
    if (task === "composition" && window.renderComp) window.renderComp();
    if (task === "crop" && window.renderCrop) window.renderCrop();
    if (task === "lut" && window.renderLut) window.renderLut();
    if (task === "text") renderAll();
  });
});

function _paceText(prefix, job) {
  const processed = job.processed || 0;
  const total = job.total || "?";
  if (!processed || !job.started_at) {
    return job.current ? `${prefix} ${processed} / ${total} · ${job.current}` : `${prefix} ${processed} / ${total}`;
  }
  const sec = (Date.now() / 1000 - job.started_at) / processed;
  const name = job.current ? ` · ${job.current}` : "";
  return `${prefix} ${processed} / ${total} · ${sec.toFixed(1)} 秒/张${name}`;
}

loadDefaults();
renderAll();
