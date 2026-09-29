const BUILTIN_IDS = new Set(["center"]);

const comp = {
  jobId: null,
  items: [],
  filter: "all",
  pollTimer: null,
  running: false,
  types: [],
  llm: {},
  busy: false,
  thTimers: {},
  refJobId: null,
  refPollTimer: null,
  refCurrent: "",
};

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function typeById(id) {
  return (comp.types || []).find((typ) => typ.id === id) || { id, name: id, refs: [], threshold: 0.58 };
}

function typeName(id) {
  if (id === "none") return "未归类";
  const typ = (comp.types || []).find((item) => item.id === id);
  return (typ && typ.name) || id;
}

function typeThreshold(typeId) {
  const typ = typeById(typeId);
  const value = Number(typ.threshold);
  return Number.isFinite(value) ? value : 0.58;
}

function itemScore(item, typeId) {
  const row = (item.scores || {})[typeId];
  if (row == null) return 0;
  if (typeof row === "number") return row;
  return Number(row.score || 0);
}

function itemKinds(item) {
  const known = [];
  const kinds = Array.isArray(item.qa_kinds) && item.qa_kinds.length
    ? item.qa_kinds
    : Array.isArray(item.compositions)
      ? item.compositions
      : [];
  kinds.forEach((kind) => {
    if (kind && kind !== "none" && !known.includes(kind)) known.push(kind);
  });
  if (!known.length && item.composition && item.composition !== "none") known.push(item.composition);
  (comp.types || []).forEach((typ) => {
    if (BUILTIN_IDS.has(typ.id)) return;
    if (itemScore(item, typ.id) + 1e-12 >= typeThreshold(typ.id) && !known.includes(typ.id)) {
      known.push(typ.id);
    }
  });
  return known;
}

function hasKind(item, kind) {
  if (kind === "none") return itemKinds(item).length === 0;
  return itemKinds(item).includes(kind);
}

function kindNames(item) {
  return itemKinds(item).map(typeName);
}

function visibleCompItems() {
  if (comp.filter === "all") return comp.items;
  return comp.items.filter((it) => hasKind(it, comp.filter));
}

function snapshotTypeFields() {
  const saved = { folders: {}, descs: {}, dests: {} };
  (comp.types || []).forEach((typ) => {
    const folder = $(`refFolder_${typ.id}`);
    const desc = $(`desc_${typ.id}`);
    const dest = $(`destComp_${typ.id}`);
    if (folder) saved.folders[typ.id] = folder.value;
    if (desc) saved.descs[typ.id] = desc.value;
    if (dest) saved.dests[typ.id] = dest.value;
  });
  return saved;
}

function restoreTypeFields(saved) {
  (comp.types || []).forEach((typ) => {
    const folder = $(`refFolder_${typ.id}`);
    const desc = $(`desc_${typ.id}`);
    const dest = $(`destComp_${typ.id}`);
    if (folder && saved.folders[typ.id] != null) folder.value = saved.folders[typ.id];
    if (desc && saved.descs[typ.id] != null) desc.value = saved.descs[typ.id];
    if (dest && saved.dests[typ.id] != null) dest.value = saved.dests[typ.id];
  });
}

function pct(value) {
  return Math.round(Number(value || 0) * 100);
}

function originalRefName(name) {
  return String(name || "").replace(/_[0-9a-f]{6}(\.[^.]+)$/i, "$1");
}

function uniqueRefs(refs) {
  const seen = new Set();
  const out = [];
  (refs || []).forEach((ref) => {
    const key = originalRefName(ref.name);
    if (!key || seen.has(key)) return;
    seen.add(key);
    out.push(ref);
  });
  return out;
}

function refAnalysisText(typ, ref) {
  const analysis = ref.analysis || {};
  if (!analysis.why && !analysis.reason) {
    return "尚未用答题分析这张参考图。点「重新分析参考图」后会列出依据。";
  }
  const x = analysis.subject_x != null ? `（x=${Number(analysis.subject_x).toFixed(2)}）` : "";
  const qaKind = analysis.qa_names && analysis.qa_names.length ? analysis.qa_names.join("、") : "未归入当前构图类型";
  return [
    `答题：${analysis.has_subject ? "有主体" : "无主体"} · 按包围盒中心 ${analysis.subject_position_name || "位置不明"}${x} · ${analysis.repeating ? "有重复主体" : "无重复主体"}`,
    `答题归类：${qaKind}`,
    `为何算${typ.name}：${analysis.why || analysis.reason || ""}`,
  ].join("\n");
}

function renderCompTypes() {
  const host = $("compRefTypes");
  if (!host) return;
  const saved = snapshotTypeFields();
  host.innerHTML = (comp.types || [])
    .map((typ) => {
      const refs = uniqueRefs(typ.refs || []);
      const how = typ.how || "读入参考图后，会在这里列出每张的答题分析和归类理由。";
      const th = pct(typ.threshold == null ? 0.58 : typ.threshold);
      const rows = refs
        .map((ref) => {
          const analysis = ref.analysis || {};
          const warn = analysis.why && analysis.agrees === false ? " is-warn" : "";
          const current = comp.refCurrent && ref.name === comp.refCurrent ? " is-current" : "";
          return `<div class="ref-row${warn}${current}">
            <div class="ref-thumb" style="background-image:url('/api/composition/ref/${typ.id}/${ref.id}?w=200')"></div>
            <p><strong>${escapeHtml(ref.name)}</strong>${escapeHtml(refAnalysisText(typ, ref))}</p>
            <button type="button" class="ref-del" data-del-type="${typ.id}" data-del-ref="${ref.id}" aria-label="删除">×</button>
          </div>`;
        })
        .join("");
      const delBtn = BUILTIN_IDS.has(typ.id)
        ? ""
        : `<button type="button" data-del-type-card="${typ.id}">删除类型</button>`;
      return `<article class="type-card type-card-static">
        <strong>${escapeHtml(typ.name)}</strong>
        <small>${escapeHtml(how)}</small>
        <label class="field"><span>构图定义</span>
          <textarea id="desc_${typ.id}" placeholder="用文字解释什么是这种构图">${escapeHtml(typ.description || "")}</textarea>
        </label>
        <div class="type-actions">
          <button type="button" data-save-desc="${typ.id}">保存定义</button>
        </div>
        <div class="field">
          <span class="row-between">匹配度阈值 <strong id="thLabel_${typ.id}">${th}%</strong></span>
          <div class="slider-row">
            <input id="thRange_${typ.id}" type="range" min="30" max="90" step="1" value="${th}" data-th="${typ.id}" />
            <input id="thNum_${typ.id}" type="number" min="30" max="90" step="1" value="${th}" data-th-num="${typ.id}" />
          </div>
        </div>
        <label class="field"><span>追加参考图文件夹</span>
          <input id="refFolder_${typ.id}" type="text" spellcheck="false" placeholder="只读这一层里的图片" />
        </label>
        <div class="type-actions">
          <button type="button" data-add-refs="${typ.id}">读入并分析参考图</button>
          <button type="button" data-reanalyze="${typ.id}" ${refs.length ? "" : "disabled"}>重新分析参考图</button>
          ${delBtn}
        </div>
        <div class="ref-analysis">${rows || '<p class="hint">还没有参考图。</p>'}</div>
      </article>`;
    })
    .join("");
  restoreTypeFields(saved);
  host.querySelectorAll("[data-add-refs]").forEach((btn) => {
    btn.addEventListener("click", () => addCompRefs(btn.dataset.addRefs));
  });
  host.querySelectorAll("[data-reanalyze]").forEach((btn) => {
    btn.addEventListener("click", () => reanalyzeCompRefs(btn.dataset.reanalyze));
  });
  host.querySelectorAll("[data-del-ref]").forEach((btn) => {
    btn.addEventListener("click", () => deleteCompRef(btn.dataset.delType, btn.dataset.delRef));
  });
  host.querySelectorAll("[data-del-type-card]").forEach((btn) => {
    btn.addEventListener("click", () => deleteCompType(btn.dataset.delTypeCard));
  });
  host.querySelectorAll("[data-save-desc]").forEach((btn) => {
    btn.addEventListener("click", () => saveCompDesc(btn.dataset.saveDesc));
  });
  host.querySelectorAll("[data-th]").forEach((el) => {
    el.addEventListener("input", () => setTypeThreshold(el.dataset.th, Number(el.value)));
  });
  host.querySelectorAll("[data-th-num]").forEach((el) => {
    el.addEventListener("input", () => setTypeThreshold(el.dataset.thNum, Number(el.value)));
  });
}

function renderCompExportFields() {
  const host = $("compExportFields");
  if (!host) return;
  const saved = snapshotTypeFields();
  host.innerHTML = (comp.types || [])
    .map((typ) => {
      return `<label class="field">
        <span>${escapeHtml(typ.name)}目标目录</span>
        <input id="destComp_${typ.id}" type="text" spellcheck="false" placeholder="默认：原目录旁 *_comp_${escapeHtml(typ.name)}" />
      </label>
      <button id="exportComp_${typ.id}" class="ghost" type="button" disabled>移出${escapeHtml(typ.name)}</button>`;
    })
    .join("");
  restoreTypeFields(saved);
  (comp.types || []).forEach((typ) => {
    const btn = $(`exportComp_${typ.id}`);
    if (btn) btn.addEventListener("click", () => exportComp(typ.id));
  });
}

function renderCompTabs() {
  const host = $("compTabs");
  if (!host) return;
  const tabs = [
    ["all", "全部"],
    ...(comp.types || []).map((typ) => [typ.id, typ.name]),
    ["none", "未归类"],
  ];
  if (!tabs.some((tab) => tab[0] === comp.filter)) comp.filter = "all";
  host.innerHTML = tabs
    .map(
      ([id, name]) =>
        `<button class="tab ${comp.filter === id ? "active" : ""}" data-compfilter="${id}" type="button">${escapeHtml(name)}</button>`
    )
    .join("");
  host.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      comp.filter = tab.dataset.compfilter;
      renderCompTabs();
      renderCompGrid();
    });
  });
}

function renderCompStats() {
  const host = $("compStats");
  if (!host) return;
  const items = comp.items;
  const cells = [
    ["全部图片", items.length || "—"],
    ...(comp.types || []).map((typ) => [typ.name, items.length ? items.filter((it) => hasKind(it, typ.id)).length : "—"]),
    ["未归类", items.length ? items.filter((it) => hasKind(it, "none")).length : "—"],
  ];
  host.innerHTML = cells.map(([name, count]) => `<article><small>${escapeHtml(name)}</small><b>${count}</b></article>`).join("");
}

function renderCompChart() {
  const canvas = $("compHist");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const rows = [
    ...(comp.types || []).map((typ) => [typ.name, comp.items.filter((it) => hasKind(it, typ.id)).length]),
    ["未归类", comp.items.filter((it) => hasKind(it, "none")).length],
  ];
  const height = Math.max(140, Math.max(rows.length, 1) * 36);
  canvas.height = height * dpr;
  canvas.width = width * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const max = Math.max(1, ...rows.map((r) => r[1]));
  const rowH = height / Math.max(rows.length, 1);
  rows.forEach(([name, count], i) => {
    const bar = ((width - 150) * count) / max;
    const y = i * rowH + 8;
    ctx.fillStyle = name === "未归类" ? "#c9b8a4" : "#b5431f";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(140, y, Math.max(bar, 0), rowH - 16, 5);
    else ctx.rect(140, y, Math.max(bar, 0), rowH - 16);
    ctx.fill();
    ctx.fillStyle = "#6f665d";
    ctx.font = "12px sans-serif";
    ctx.fillText(name, 4, y + rowH * 0.55);
    ctx.fillText(String(count), 148 + bar, y + rowH * 0.55);
  });
}

function itemAnalysisText(item) {
  const sideName = { left: "左侧", right: "右侧", center: "中部" }[item.subject_side] || "";
  const x = item.subject_x != null ? `（x=${Number(item.subject_x).toFixed(2)}）` : "";
  const side = sideName ? ` · 主体在${sideName}` : "";
  const qaKind = item.qa_names && item.qa_names.length
    ? item.qa_names.join("、")
    : kindNames(item).join("、") || "未归入当前构图类型";
  const lines = [
    `答题：${item.has_subject ? "有主体" : "无主体"} · ${item.subject_position_name || "位置不明"}${x}${side} · ${item.repeating ? "有重复主体" : "无重复主体"}`,
    `答题归类：${qaKind}`,
    `依据：${item.why || item.reason || "无"}`,
  ];
  if (item.retries) lines.push(`已重问 ${item.retries} 次，不接受套模板坐标。`);
  return lines.join("\n");
}

function renderCompGrid() {
  const items = visibleCompItems();
  $("compGalleryCount").textContent = comp.items.length
    ? `当前视图 ${items.length} 张`
    : "还没有构图分析结果";
  $("compGrid").innerHTML = items
    .map((item) => {
      const kinds = itemKinds(item);
      const hit = kinds.length > 0;
      const label = hit
        ? `${kindNames(item).join(" · ")} · ${item.subject_position_name || ""}`
        : item.has_subject
          ? `未归类 · ${item.subject_position_name || "其他位置"}`
          : "未归类 · 无主体";
      return `
        <article class="card ${hit ? "is-text" : ""}" data-id="${item.id}">
          <div class="thumb" style="background-image:url('/api/thumb/${item.id}?job_id=${comp.jobId}&w=420')"></div>
          <div class="meta">
            <strong title="${item.rel}">${item.name}</strong>
            <span class="badge ${hit ? "hot" : ""}">${label}</span>
            <p class="card-why">${escapeHtml(itemAnalysisText(item))}</p>
          </div>
        </article>`;
    })
    .join("");
  $("compGrid").querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => openCompLightbox(card.dataset.id));
  });
}

function restoreCompExportButtons() {
  (comp.types || []).forEach((typ) => {
    const btn = $(`exportComp_${typ.id}`);
    if (btn) btn.disabled = !comp.items.some((it) => hasKind(it, typ.id));
  });
  $("exportCompUnassignedBtn").disabled = !comp.items.some((it) => hasKind(it, "none"));
}

function setCompScanControls() {
  $("compScanBtn").disabled = false;
  $("compScanBtn").textContent = comp.items.length || comp.running ? "重新分析" : "开始分析";
  $("compStopBtn").disabled = !comp.running;
}

function renderComp() {
  renderCompStats();
  renderCompChart();
  renderCompTabs();
  renderCompGrid();
  restoreCompExportButtons();
  setCompScanControls();
}

function applyCompStore(composition) {
  const saved = snapshotTypeFields();
  comp.types = (composition && composition.types) || [];
  comp.llm = (composition && composition.llm) || {};
  renderCompTypes();
  renderCompExportFields();
  restoreTypeFields(saved);
  renderComp();
  if (!comp.running && comp.llm.detail) $("compScanHint").textContent = comp.llm.detail;
}

function currentTypeThresholds() {
  const out = {};
  (comp.types || []).forEach((typ) => {
    out[typ.id] = typeThreshold(typ.id);
  });
  return out;
}

function setTypeThreshold(typeId, raw) {
  const value = Math.max(30, Math.min(90, Number(raw) || 58));
  const typ = typeById(typeId);
  typ.threshold = value / 100;
  const range = $(`thRange_${typeId}`);
  const num = $(`thNum_${typeId}`);
  const label = $(`thLabel_${typeId}`);
  if (range && Number(range.value) !== value) range.value = String(value);
  if (num && Number(num.value) !== value) num.value = String(value);
  if (label) label.textContent = `${value}%`;
  renderComp();
  clearTimeout(comp.thTimers[typeId]);
  comp.thTimers[typeId] = setTimeout(() => {
    fetch(`/api/composition/types/${typeId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold: value / 100 }),
    }).catch(() => {});
  }, 280);
}

async function saveCompDesc(typeId) {
  const desc = $(`desc_${typeId}`);
  $("compRefHint").textContent = "正在保存定义…";
  const res = await fetch(`/api/composition/types/${typeId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description: desc ? desc.value : "" }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("compRefHint").textContent = errorDetail(data, "保存定义失败");
    return;
  }
  applyCompStore(data.store);
  $("compRefHint").textContent = `已保存「${typeName(typeId)}」的定义。`;
}

async function addCompType() {
  const name = $("compTypeName").value.trim();
  const description = $("compTypeDesc").value.trim();
  const folder = $("compRefFolder").value.trim();
  if (!name) {
    $("compRefHint").textContent = "请填写构图类型名称。";
    return;
  }
  if (!description && !folder) {
    $("compRefHint").textContent = "请填写构图定义，或提供参考图文件夹。";
    return;
  }
  if (comp.busy) return;
  comp.busy = true;
  $("compRefHint").textContent = folder ? "正在添加类型，随后逐张分析参考图…" : "正在添加构图类型…";
  try {
    const res = await fetch("/api/composition/types", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, folder, description }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("compRefHint").textContent = errorDetail(data, "添加失败");
      return;
    }
    $("compTypeName").value = "";
    $("compTypeDesc").value = "";
    $("compRefFolder").value = "";
    applyCompStore(data.store);
    if (data.job_id) {
      $("compRefHint").textContent = `已添加「${name}」，开始分析参考图并刷新描述。`;
      startRefJob(data.job_id);
      return;
    }
    $("compRefHint").textContent = `已添加「${name}」。`;
  } catch (err) {
    $("compRefHint").textContent = `添加失败：${err.message || err}`;
  } finally {
    if (!comp.refJobId) comp.busy = false;
  }
}

async function deleteCompType(typeId) {
  if (comp.busy) return;
  comp.busy = true;
  try {
    const res = await fetch(`/api/composition/types/${typeId}`, { method: "DELETE" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("compRefHint").textContent = errorDetail(data, "删除类型失败");
      return;
    }
    applyCompStore(data);
    $("compRefHint").textContent = "已删除该构图类型。";
  } catch (err) {
    $("compRefHint").textContent = `删除失败：${err.message || err}`;
  } finally {
    comp.busy = false;
  }
}

function openCompLightbox(id) {
  const item = comp.items.find((it) => it.id === id);
  if (!item) return;
  const img = new Image();
  img.onload = () => {
    const canvas = $("preview");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
  };
  img.src = `/api/image/${item.id}?job_id=${comp.jobId}`;
  $("lbName").textContent = item.name;
  const lines = [
    `${item.width}×${item.height}`,
    `判定：${kindNames(item).join("、") || "未归类"}`,
    itemAnalysisText(item),
  ];
  if (item.sidecar) lines.push(`已写入 ${item.sidecar}`);
  $("lbMeta").textContent = lines.join("\n");
  $("lightbox").showModal();
}

function applyCompJob(job) {
  comp.jobId = job.id;
  comp.items = job.items || [];
  comp.running = job.status === "running" || job.status === "queued" || job.status === "listing";
  if (job.status === "error") {
    $("compProgressText").textContent = job.error || "分析失败";
    $("compScanHint").textContent = job.error || "分析失败";
  } else if (job.status === "listing") {
    $("compProgressText").textContent = job.total
      ? `正在列出文件，已找到 ${job.total} 张（还没开始分析）`
      : "正在列出文件，只收集路径，尚未分析…";
  } else if (job.status === "running" || job.status === "queued") {
    let text = `正在看图 ${job.processed || 0} / ${job.total || "?"}`;
    if (job.processed && job.started_at) {
      const sec = (Date.now() / 1000 - job.started_at) / job.processed;
      text += ` · ${sec.toFixed(1)} 秒/张`;
    }
    if (job.current) text += ` · ${job.current}`;
    $("compProgressText").textContent = text;
  } else if (job.status === "done") {
    $("compProgressText").textContent = `完成 ${job.processed} 张`;
  } else if (job.status === "stopped") {
    $("compProgressText").textContent = `已停止 ${job.processed} / ${job.total || job.processed}`;
  }
  renderComp();
}

function stopCompPolling() {
  if (comp.pollTimer) {
    clearInterval(comp.pollTimer);
    comp.pollTimer = null;
  }
}

async function pollCompJob() {
  if (!comp.jobId) return;
  const res = await fetch(`/api/jobs/${comp.jobId}`);
  const job = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("compProgressText").textContent =
      res.status === 404 ? "任务已丢失，请重新分析" : "读取进度失败，请重新分析";
    stopCompPolling();
    comp.running = false;
    setCompScanControls();
    return;
  }
  applyCompJob(job);
  if (job.status === "done" || job.status === "stopped" || job.status === "error") {
    stopCompPolling();
    comp.running = false;
    setCompScanControls();
  }
}

async function startCompScan() {
  const folder = $("folder").value.trim();
  if (!folder) {
    $("compScanHint").textContent = "请填写本地文件夹路径。";
    return;
  }
  if (comp.running) await stopCompScan();
  stopCompPolling();
  comp.running = true;
  $("compStopBtn").disabled = false;
  $("compExportHint").textContent = "";
  $("compScanHint").textContent = "正在逐张答题（与参考图同一套方法），并对照参考图的主体位置…";
  const res = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      folder,
      recursive: $("recursive").checked,
      kind: "composition",
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "扫描失败" }));
    $("compScanHint").textContent = err.detail || "扫描失败";
    comp.running = false;
    setCompScanControls();
    return;
  }
  const data = await res.json();
  comp.jobId = data.job_id;
  comp.items = [];
  comp.running = true;
  $("compScanHint").textContent = "构图分析进行中，可随时停止。";
  renderComp();
  comp.pollTimer = setInterval(pollCompJob, 400);
  pollCompJob();
}

async function stopCompScan() {
  if (!comp.jobId || !comp.running) return;
  $("compStopBtn").disabled = true;
  await fetch(`/api/jobs/${comp.jobId}/stop`, { method: "POST" });
  $("compScanHint").textContent = "已请求停止。";
  await pollCompJob();
}

async function exportComp(typeId) {
  if (!comp.jobId) {
    $("compExportHint").textContent = "请先分析完再移动。";
    return;
  }
  const unassigned = typeId === "none";
  const label = unassigned ? "未归类" : typeName(typeId);
  const destId = unassigned ? "destCompUnassigned" : `destComp_${typeId}`;
  $("compExportHint").textContent = `正在挪走${label}…`;
  document.querySelectorAll("#compControls button.ghost").forEach((btn) => {
    btn.disabled = true;
  });
  try {
    const destInput = $(destId);
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: comp.jobId,
        threshold: 0,
        type_id: unassigned ? null : typeId,
        dest: destInput && destInput.value.trim() ? destInput.value.trim() : null,
        kind: unassigned ? "composition_rest" : "composition_match",
        type_thresholds: currentTypeThresholds(),
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("compExportHint").textContent = errorDetail(data, `移动失败（${res.status}）`);
      return;
    }
    const extra = data.skipped ? `，另有 ${data.skipped} 张已在目标目录` : "";
    $("compExportHint").textContent = `已挪走 ${data.count} 张${label}到 ${data.dest}${extra}`;
    const jobRes = await fetch(`/api/jobs/${comp.jobId}`);
    if (jobRes.ok) applyCompJob(await jobRes.json());
  } catch (err) {
    $("compExportHint").textContent = `移动失败：${err.message || err}`;
  } finally {
    restoreCompExportButtons();
  }
}

function showRefProgress(job) {
  const box = $("compRefProgress");
  const text = $("compRefProgressText");
  const fill = $("compRefProgressFill");
  const stop = $("compRefStopBtn");
  if (!box || !text || !fill) return;
  box.hidden = false;
  const total = job.total || 0;
  const processed = job.processed || 0;
  const pct = total ? Math.min(100, Math.round((processed / total) * 100)) : 0;
  fill.style.width = `${pct}%`;
  const action = job.action === "reanalyze" ? "重新分析参考图" : "分析参考图";
  const label = job.type_name || "该构图";
  let line = `${action}「${label}」 ${processed} / ${total || "?"}`;
  if (job.current) line += ` · 正在看 ${job.current}`;
  if (job.last_name && job.last_name !== job.current) line += ` · 已刷新 ${job.last_name} 的描述`;
  if (job.refreshed) line += ` · 已刷新 ${job.refreshed} 张`;
  if (job.status === "done") {
    line = `已完成「${label}」${processed} 张，描述已刷新。`;
  } else if (job.status === "stopped") {
    line = `已停止「${label}」${processed} / ${total}，已刷新 ${job.refreshed || 0} 张描述。`;
  } else if (job.status === "error") {
    line = job.error || "参考图分析失败";
  }
  text.textContent = line;
  if (stop) stop.disabled = job.status !== "running" && job.status !== "queued";
}

function hideRefProgressSoon() {
  setTimeout(() => {
    if (comp.refJobId) return;
    const box = $("compRefProgress");
    if (box) box.hidden = true;
  }, 4000);
}

function stopRefPolling() {
  if (comp.refPollTimer) {
    clearInterval(comp.refPollTimer);
    comp.refPollTimer = null;
  }
}

function startRefJob(jobId) {
  stopRefPolling();
  comp.refJobId = jobId;
  comp.busy = true;
  $("compRefProgress").hidden = false;
  $("compRefProgressText").textContent = "开始分析参考图，会逐张刷新描述…";
  $("compRefProgressFill").style.width = "0%";
  $("compRefStopBtn").disabled = false;
  comp.refPollTimer = setInterval(pollRefJob, 400);
  pollRefJob();
}

async function pollRefJob() {
  if (!comp.refJobId) return;
  const res = await fetch(`/api/jobs/${comp.refJobId}`);
  const job = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("compRefProgressText").textContent = "读取参考图分析进度失败。";
    stopRefPolling();
    comp.refJobId = null;
    comp.busy = false;
    hideRefProgressSoon();
    return;
  }
  comp.refCurrent = job.current || "";
  if (job.store) applyCompStore(job.store);
  showRefProgress(job);
  if (job.status === "done" || job.status === "stopped" || job.status === "error") {
    stopRefPolling();
    comp.refJobId = null;
    comp.refCurrent = "";
    comp.busy = false;
    if (job.store) applyCompStore(job.store);
    $("compRefHint").textContent =
      job.status === "done"
        ? `参考图分析完成，已刷新 ${job.refreshed || job.processed || 0} 张描述。`
        : job.status === "stopped"
          ? `已停止参考图分析，已刷新 ${job.refreshed || 0} 张描述。`
          : job.error || "参考图分析失败";
    hideRefProgressSoon();
  }
}

async function stopRefJob() {
  if (!comp.refJobId) return;
  $("compRefStopBtn").disabled = true;
  await fetch(`/api/jobs/${comp.refJobId}/stop`, { method: "POST" });
  $("compRefProgressText").textContent = "已请求停止，当前这张看完就会停。";
  await pollRefJob();
}

async function addCompRefs(typeId) {
  const input = $(`refFolder_${typeId}`);
  const folder = input && input.value.trim();
  const label = typeName(typeId);
  if (!folder) {
    $("compRefHint").textContent = `请填写「${label}」参考图文件夹路径。`;
    return;
  }
  if (comp.busy) {
    $("compRefHint").textContent = "请等当前参考图分析结束。";
    return;
  }
  comp.busy = true;
  $("compRefHint").textContent = `正在读入「${label}」参考图…`;
  try {
    const res = await fetch(`/api/composition/types/${typeId}/refs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("compRefHint").textContent = errorDetail(data, "读入参考图失败");
      return;
    }
    applyCompStore(data.store);
    const added = data.added || 0;
    const skipped = data.skipped || 0;
    const skipTxt = skipped ? `，跳过重复 ${skipped} 张` : "";
    if (data.job_id) {
      $("compRefHint").textContent = `「${label}」新读入 ${added} 张${skipTxt}，正在逐张分析并刷新描述。`;
      startRefJob(data.job_id);
      return;
    }
    $("compRefHint").textContent = skipped
      ? `「${label}」没有新图，${skipped} 张都已有了，未重复添加。`
      : `「${label}」参考图已读入。`;
  } catch (err) {
    $("compRefHint").textContent = `读入失败：${err.message || err}`;
  } finally {
    if (!comp.refJobId) comp.busy = false;
  }
}

async function reanalyzeCompRefs(typeId) {
  const label = typeName(typeId);
  if (comp.busy) {
    $("compRefHint").textContent = "请等当前参考图分析结束。";
    return;
  }
  comp.busy = true;
  $("compRefHint").textContent = `开始重新分析「${label}」参考图…`;
  try {
    const res = await fetch(`/api/composition/types/${typeId}/analyze`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("compRefHint").textContent = errorDetail(data, "重新分析失败");
      return;
    }
    if (data.store) applyCompStore(data.store);
    if (data.job_id) {
      startRefJob(data.job_id);
      return;
    }
    $("compRefHint").textContent = `「${label}」没有需要分析的参考图。`;
  } catch (err) {
    $("compRefHint").textContent = `重新分析失败：${err.message || err}`;
  } finally {
    if (!comp.refJobId) comp.busy = false;
  }
}

async function deleteCompRef(typeId, refId) {
  if (comp.busy) return;
  comp.busy = true;
  try {
    const res = await fetch(`/api/composition/types/${typeId}/refs/${refId}`, { method: "DELETE" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("compRefHint").textContent = errorDetail(data, "删除参考图失败");
      return;
    }
    applyCompStore(data.store || data);
    $("compRefHint").textContent = "已删除该参考图。";
  } catch (err) {
    $("compRefHint").textContent = `删除失败：${err.message || err}`;
  } finally {
    comp.busy = false;
  }
}

async function loadCompStatus() {
  const res = await fetch("/api/defaults");
  if (!res.ok) return;
  const data = await res.json();
  applyCompStore(data.composition || {});
}

window.renderComp = renderComp;

$("compAddTypeBtn").addEventListener("click", addCompType);
$("compRefStopBtn").addEventListener("click", stopRefJob);
$("compScanBtn").addEventListener("click", startCompScan);
$("compStopBtn").addEventListener("click", stopCompScan);
$("exportCompUnassignedBtn").addEventListener("click", () => exportComp("none"));
window.addEventListener("resize", () => {
  if (!$("compStage").hidden) renderCompChart();
});
loadCompStatus();
renderComp();
