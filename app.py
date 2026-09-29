"""Local web app: scan a folder, score text-area ratio, preview and export."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from lut_store import (
    apply_layers,
    apply_to_image,
    create_or_update_lut,
    cube_path,
    resolve_grade,
    delete_lut,
    hald_path,
    list_luts,
    load_preview_source,
    parse_stack_query,
    rename_lut,
    set_active,
)

from color_analysis import analyze_color
from crop_borders import analyze_borders, save_cropped
from composition import COMP_LABELS, JUDGED_KINDS, analyze_composition, sidecar_path
from composition_llm import llm_status
from composition_store import (
    add_refs,
    add_type,
    analyze_and_save_ref,
    delete_ref,
    delete_type,
    get_ref_record,
    library_for_match,
    list_ref_ids,
    public_store,
    update_type,
    resolve_ref,
    save_settings,
)
from detector import analyze_image, is_image_name, warmup

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
CACHE = ROOT / ".cache" / "thumbs"
CACHE.mkdir(parents=True, exist_ok=True)

DEFAULT_FOLDER = str(ROOT / "samples")

app = FastAPI(title="Image Text Sorter")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()
_ocr_status = {"ready": False, "engine": "", "error": ""}


class ScanRequest(BaseModel):
    folder: str
    recursive: bool = False
    kind: str = "text"  # text | color | composition | crop


class ExportRequest(BaseModel):
    job_id: str
    threshold: float = Field(ge=0.0, le=1.0)
    extreme_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    over_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    under_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    neon_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.58, ge=0.0, le=1.0)
    type_thresholds: Optional[Dict[str, float]] = None
    type_id: Optional[str] = None
    dest: Optional[str] = None
    dest_uncropped: Optional[str] = None
    kind: str = "text"  # text | rest | color_fail | color_pass | composition_match | composition_rest | crop
    crop_boxes: Optional[Dict[str, List[int]]] = None


class CropBoxRequest(BaseModel):
    box: Optional[List[int]] = None
    reset: bool = False


class CompositionTypeRequest(BaseModel):
    name: str
    folder: str
    description: str = ""


class CompositionUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CompositionRefsRequest(BaseModel):
    folder: str


class CompositionSettingsRequest(BaseModel):
    active_type_id: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class LutGenerateRequest(BaseModel):
    name: str = ""
    description: str = ""
    folder: str = ""
    lut_id: Optional[str] = None
    regenerate: bool = False


class LutRenameRequest(BaseModel):
    name: str
    description: Optional[str] = None


class LutStrengthRequest(BaseModel):
    job_id: str
    item_id: Optional[str] = None
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    strengths: Optional[Dict[str, float]] = None
    all_strength: Optional[float] = Field(default=None, ge=0.0, le=1.5)


class LutLayerIn(BaseModel):
    lut_id: str
    strength: float = Field(default=1.0, ge=0.0, le=1.5)
    enabled: bool = True
    apply: str = "auto"
    hue: Optional[float] = None
    sat: Optional[float] = None
    gamma: Optional[float] = None
    contrast: Optional[float] = None
    hue_red: Optional[float] = None
    hue_yellow: Optional[float] = None
    hue_green: Optional[float] = None
    hue_cyan: Optional[float] = None
    hue_blue: Optional[float] = None
    hue_magenta: Optional[float] = None


class LutGradeIn(BaseModel):
    mode: str = "auto"
    hue: Optional[float] = None
    sat: Optional[float] = None
    gamma: Optional[float] = None
    contrast: Optional[float] = None


class LutExportRequest(BaseModel):
    job_id: str
    lut_id: Optional[str] = None
    dest: Optional[str] = None
    item_ids: Optional[List[str]] = None
    layers: Optional[List[LutLayerIn]] = None
    item_layers: Optional[Dict[str, List[LutLayerIn]]] = None
    item_grades: Optional[Dict[str, LutGradeIn]] = None


class LutExportOneBody(BaseModel):
    job_id: str
    item_id: str
    lut_id: Optional[str] = None
    stack: Optional[str] = None
    layers: Optional[List[LutLayerIn]] = None
    grade: Optional[LutGradeIn] = None


def _public_item(item: dict, kind: str) -> dict:
    base = {
        "id": item["id"],
        "name": item["name"],
        "rel": item["rel"],
        "width": item["width"],
        "height": item["height"],
    }
    if kind == "color":
        base.update(
            {
                "extreme_ratio": item.get("extreme_ratio", 0.0),
                "bright_ratio": item.get("bright_ratio", 0.0),
                "dark_ratio": item.get("dark_ratio", 0.0),
                "white_ratio": item.get("white_ratio", 0.0),
                "black_ratio": item.get("black_ratio", 0.0),
                "over_ratio": item.get("over_ratio", 0.0),
                "under_ratio": item.get("under_ratio", 0.0),
                "neon_ratio": item.get("neon_ratio", 0.0),
                "harmony": item.get("harmony", "none"),
                "harmony_name": item.get("harmony_name", ""),
                "harmony_detail": item.get("harmony_detail", ""),
                "colors": item.get("colors", []),
            }
        )
        return base
    if kind == "lut":
        base.update(
            {
                "strength": float(item.get("strength") or 1.0),
                "layer_strengths": item.get("layer_strengths") or {},
            }
        )
        return base
    if kind == "crop":
        width = item.get("width") or 0
        height = item.get("height") or 0
        box = item.get("crop_box") or [0, 0, width, height]
        base.update(
            {
                "crop_box": box,
                "auto_crop_box": item.get("auto_crop_box") or box,
                "crop_width": item.get("crop_width") or width,
                "crop_height": item.get("crop_height") or height,
                "cropped": bool(item.get("cropped")),
                "edited": bool(item.get("edited")),
                "border": item.get("border") or "none",
                "trim_top": item.get("trim_top") or 0,
                "trim_bottom": item.get("trim_bottom") or 0,
                "trim_left": item.get("trim_left") or 0,
                "trim_right": item.get("trim_right") or 0,
                "saved": item.get("saved") or "",
            }
        )
        return base
    if kind == "composition":
        base.update(
            {
                "composition": item.get("composition") or "none",
                "composition_name": item.get("composition_name") or "未归类",
                "compositions": _item_kinds(item),
                "composition_names": item.get("composition_names") or [],
                "has_subject": bool(item.get("has_subject")),
                "subject_position": item.get("subject_position") or "none",
                "subject_position_name": item.get("subject_position_name") or "无主体",
                "subject_x": item.get("subject_x"),
                "subject_y": item.get("subject_y"),
                "subject_box": item.get("subject_box"),
                "subject_side": item.get("subject_side") or "",
                "retries": item.get("retries") or 0,
                "lazy_corrected": bool(item.get("lazy_corrected")),
                "repeating": bool(item.get("repeating")),
                "reason": item.get("reason") or "",
                "why": item.get("why") or "",
                "ref_notes": item.get("ref_notes") or [],
                "qa_kinds": item.get("qa_kinds") or [],
                "qa_names": item.get("qa_names") or [],
                "scores": item.get("scores") or {},
                "sidecar": item.get("sidecar") or "",
            }
        )
        return base
    base.update(
        {
            "text_ratio": item.get("text_ratio", 0.0),
            "box_count": item.get("box_count", 0),
            "texts": item.get("texts", []),
            "boxes": item.get("boxes", []),
            "engine": item.get("engine", ""),
        }
    )
    return base


def _skip_dir_name(name: str) -> bool:
    return (
        name.startswith(".")
        or name == "__text_sorted__"
        or name.endswith("_with_text")
        or name.endswith("_no_text")
        or name.endswith("_composed")
        or name.endswith("_color_fail")
        or name.endswith("_color_pass")
        or "_comp_" in name
        or name.endswith("_cropped")
        or name.endswith("_uncropped")
        or name.endswith("_lut")
        or name.endswith("_with_lut")
    )


def _list_images(folder: Path, recursive: bool, job_id: Optional[str] = None) -> List[Path]:
    found: List[Path] = []

    def note() -> None:
        if not job_id:
            return
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job["total"] = len(found)

    if recursive:
        for root, dirs, names in os.walk(folder, followlinks=False):
            dirs[:] = [d for d in dirs if not _skip_dir_name(d)]
            for name in names:
                if is_image_name(name):
                    found.append(Path(root) / name)
            note()
            with _jobs_lock:
                if job_id and _jobs.get(job_id, {}).get("cancel"):
                    break
    else:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if is_image_name(entry.name) and entry.is_file(follow_symlinks=False):
                    found.append(Path(entry.path))
        note()
    found.sort(key=lambda p: p.as_posix().lower())
    return found


def _run_scan(job_id: str, folder: Path, recursive: bool, kind: str) -> None:
    if kind == "composition":
        status = llm_status()
        if not status.get("ready") or status.get("mode") != "vision":
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = status.get("detail") or "构图分析需要能看图的视觉模型"
            return
    with _jobs_lock:
        _jobs[job_id]["status"] = "listing"
    files = _list_images(folder, recursive, job_id)
    with _jobs_lock:
        _jobs[job_id]["total"] = len(files)
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()

    workers = 1 if kind == "composition" else 4 if kind in ("color", "crop", "lut") else 2

    def work(path: Path) -> dict:
        try:
            return _analyze_one(path, folder, kind)
        except Exception as exc:  # noqa: BLE001
            return {
                "id": uuid.uuid4().hex,
                "path": str(path.resolve()),
                "name": path.name,
                "rel": str(path.relative_to(folder)),
                "width": 0,
                "height": 0,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, path) for path in files]
        for fut in as_completed(futures):
            with _jobs_lock:
                if _jobs[job_id].get("cancel"):
                    _jobs[job_id]["status"] = "stopped"
                    return
            item = fut.result()
            with _jobs_lock:
                job = _jobs[job_id]
                if job.get("cancel"):
                    job["status"] = "stopped"
                    return
                job["items"].append(item)
                job["processed"] = len(job["items"])
                job["current"] = item.get("name") or ""

    with _jobs_lock:
        job = _jobs[job_id]
        job["status"] = "stopped" if job.get("cancel") else "done"


def _analyze_one(path: Path, folder: Path, kind: str) -> dict:
    base = {
        "id": uuid.uuid4().hex,
        "path": str(path.resolve()),
        "name": path.name,
        "rel": str(path.relative_to(folder)),
    }
    if kind == "color":
        det = analyze_color(str(path))
        base.update(det.as_public())
        return base
    if kind == "composition":
        base.update(analyze_composition(str(path), library_for_match()))
        return base
    if kind == "crop":
        plan = analyze_borders(str(path)).as_public()
        plan["auto_crop_box"] = list(plan.get("crop_box") or [0, 0, 0, 0])
        plan["edited"] = False
        base.update(plan)
        return base
    if kind == "lut":
        with Image.open(path) as im:
            width, height = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):
                width, height = height, width
        base.update({"width": width, "height": height, "strength": 1.0})
        return base
    det = analyze_image(str(path))
    base.update(
        {
            "width": det.width,
            "height": det.height,
            "text_ratio": round(float(det.ratio), 6),
            "box_count": det.box_count,
            "texts": det.texts,
            "boxes": det.boxes,
            "engine": det.engine,
        }
    )
    return base


def _maybe_start_ref_job(type_id: str, ref_ids: List[str], action: str) -> str:
    ref_ids = [rid for rid in ref_ids if rid]
    if not ref_ids:
        return ""
    type_name = ""
    for typ in public_store().get("types") or []:
        if typ.get("id") == type_id:
            type_name = typ.get("name") or type_id
            break
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "folder": "",
            "recursive": False,
            "kind": "ref_analyze",
            "status": "running",
            "processed": 0,
            "total": len(ref_ids),
            "items": [],
            "cancel": False,
            "started_at": time.time(),
            "current": "",
            "error": "",
            "type_id": type_id,
            "type_name": type_name,
            "action": action,
            "refreshed": 0,
            "last_name": "",
            "store": public_store(),
        }
    threading.Thread(target=_run_ref_job, args=(job_id, type_id, ref_ids), daemon=True).start()
    return job_id


def _run_ref_job(job_id: str, type_id: str, ref_ids: List[str]) -> None:
    for ref_id in ref_ids:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job or job.get("cancel"):
                if job:
                    job["status"] = "stopped"
                    job["store"] = public_store()
                return
            try:
                ref = get_ref_record(type_id, ref_id)
                job["current"] = ref.get("name") or ref_id
            except KeyError:
                job["current"] = ref_id
        try:
            analyze_and_save_ref(type_id, ref_id)
            refreshed = True
        except Exception as exc:  # noqa: BLE001
            refreshed = False
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job:
                    job["error"] = str(exc)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job["processed"] = int(job.get("processed") or 0) + 1
            if refreshed:
                job["refreshed"] = int(job.get("refreshed") or 0) + 1
                job["last_name"] = job.get("current") or ""
            job["store"] = public_store()
            if job.get("cancel"):
                job["status"] = "stopped"
                return
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "error" if job.get("error") and job.get("processed") == 0 else "done"
        job["current"] = ""
        job["store"] = public_store()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/defaults")
def defaults() -> dict:
    return {"folder": DEFAULT_FOLDER, "ocr": _ocr_status, "composition": public_store(), "luts": list_luts()}


@app.get("/api/composition/types")
def composition_types() -> dict:
    return public_store()


@app.post("/api/composition/types")
def composition_add_type(req: CompositionTypeRequest) -> dict:
    try:
        folder = Path(req.folder).expanduser() if (req.folder or "").strip() else None
        typ = add_type(req.name, folder, req.description)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = _maybe_start_ref_job(typ["id"], list_ref_ids(typ["id"]), "add")
    return {"type": typ, "store": public_store(), "job_id": job_id}


@app.patch("/api/composition/types/{type_id}")
def composition_rename(type_id: str, req: CompositionUpdateRequest) -> dict:
    try:
        typ = update_type(type_id, req.name, req.description, req.threshold)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"type": typ, "store": public_store()}


@app.delete("/api/composition/types/{type_id}")
def composition_delete(type_id: str) -> dict:
    try:
        store = delete_type(type_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return store


@app.post("/api/composition/types/{type_id}/refs")
def composition_add_refs(type_id: str, req: CompositionRefsRequest) -> dict:
    try:
        typ, added_ids, skipped = add_refs(type_id, Path(req.folder))
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = _maybe_start_ref_job(type_id, added_ids, "add")
    return {"type": typ, "store": public_store(), "job_id": job_id, "added": len(added_ids), "skipped": skipped}


@app.post("/api/composition/types/{type_id}/analyze")
def composition_reanalyze(type_id: str) -> dict:
    try:
        ref_ids = list_ref_ids(type_id)
        if not ref_ids:
            raise ValueError("这个类型还没有参考图")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    job_id = _maybe_start_ref_job(type_id, ref_ids, "reanalyze")
    return {"type": next((t for t in public_store()["types"] if t["id"] == type_id), None), "store": public_store(), "job_id": job_id}


@app.delete("/api/composition/types/{type_id}/refs/{ref_id}")
def composition_delete_ref(type_id: str, ref_id: str) -> dict:
    try:
        typ = delete_ref(type_id, ref_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"type": typ, "store": public_store()}


@app.put("/api/composition/settings")
def composition_settings(req: CompositionSettingsRequest) -> dict:
    save_settings(req.active_type_id, req.threshold)
    return public_store()


@app.get("/api/composition/ref/{type_id}/{ref_id}")
def composition_ref_image(type_id: str, ref_id: str, w: int = 200) -> FileResponse:
    try:
        path = resolve_ref(type_id, ref_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    w = max(80, min(w, 720))
    cache_file = CACHE / f"ref_{type_id}_{ref_id}_{w}.jpg"
    if not cache_file.exists():
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((w, int(w * 1.6)))
            im.save(cache_file, "JPEG", quality=82)
    return FileResponse(cache_file, media_type="image/jpeg")


@app.post("/api/warmup")
def api_warmup() -> dict:
    try:
        engine = warmup()
        _ocr_status.update({"ready": True, "engine": engine, "error": ""})
    except Exception as exc:  # noqa: BLE001
        _ocr_status.update({"ready": False, "engine": "opencv", "error": str(exc)})
    return _ocr_status


@app.post("/api/scan")
def start_scan(req: ScanRequest) -> dict:
    folder = Path(req.folder).expanduser()
    if not folder.is_dir():
        raise HTTPException(400, f"文件夹不存在：{folder}")
    kind = req.kind if req.kind in ("text", "color", "composition", "crop", "lut") else "text"
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "folder": str(folder.resolve()),
            "recursive": req.recursive,
            "kind": kind,
            "status": "queued",
            "processed": 0,
            "total": 0,
            "items": [],
            "cancel": False,
            "started_at": None,
            "current": "",
            "error": "",
        }
    threading.Thread(target=_run_scan, args=(job_id, folder.resolve(), req.recursive, kind), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        kind = job.get("kind", "text")
        raw_items = list(job.get("items") or [])
        snapshot = {
            "id": job["id"],
            "folder": job.get("folder") or "",
            "recursive": job.get("recursive"),
            "kind": kind,
            "status": job["status"],
            "processed": job["processed"],
            "total": job["total"],
            "started_at": job.get("started_at"),
            "current": job.get("current") or "",
            "error": job.get("error") or "",
            "type_id": job.get("type_id") or "",
            "type_name": job.get("type_name") or "",
            "action": job.get("action") or "",
            "refreshed": job.get("refreshed") or 0,
            "last_name": job.get("last_name") or "",
            "store": job.get("store"),
        }
    if kind == "ref_analyze":
        snapshot["items"] = []
        if snapshot.get("store") is None:
            snapshot["store"] = public_store()
        return snapshot
    snapshot["items"] = [_public_item(it, kind) for it in raw_items]
    return snapshot


def _normalize_crop_box(width: int, height: int, box: List[int]) -> List[int]:
    if width <= 0 or height <= 0:
        raise HTTPException(400, "图片尺寸未知")
    if not isinstance(box, list) or len(box) < 4:
        raise HTTPException(400, "裁切框无效")
    left, top, right, bottom = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    left = min(max(0, left), max(0, width - 1))
    top = min(max(0, top), max(0, height - 1))
    right = min(width, max(left + 1, right))
    bottom = min(height, max(top + 1, bottom))
    return [left, top, right, bottom]


def _set_item_crop_box(item: dict, box: Optional[List[int]], edited: bool) -> None:
    width = int(item.get("width") or 0)
    height = int(item.get("height") or 0)
    if not item.get("auto_crop_box"):
        item["auto_crop_box"] = list(item.get("crop_box") or [0, 0, width, height])
    if box is None:
        box = item.get("auto_crop_box") or [0, 0, width, height]
    left, top, right, bottom = _normalize_crop_box(width, height, box)
    item["crop_box"] = [left, top, right, bottom]
    item["crop_width"] = right - left
    item["crop_height"] = bottom - top
    item["trim_left"] = left
    item["trim_top"] = top
    item["trim_right"] = width - right
    item["trim_bottom"] = height - bottom
    item["cropped"] = [left, top, right, bottom] != [0, 0, width, height]
    item["edited"] = edited
    if item["cropped"] and item.get("border") in (None, "", "none"):
        item["border"] = "manual"
    if not item["cropped"]:
        item["border"] = "none"


@app.patch("/api/jobs/{job_id}/crop/{item_id}")
def update_crop_box(job_id: str, item_id: str, req: CropBoxRequest) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        if job.get("kind") != "crop":
            raise HTTPException(400, "当前任务不是去边裁剪")
        item = next((it for it in job.get("items") or [] if it.get("id") == item_id), None)
        if not item:
            raise HTTPException(404, "图片不在此任务中")
        if req.reset:
            _set_item_crop_box(item, item.get("auto_crop_box"), False)
        else:
            _set_item_crop_box(item, req.box, True)
        return _public_item(item, "crop")


@app.post("/api/jobs/{job_id}/stop")
def stop_scan(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        if job["status"] in ("queued", "running"):
            job["cancel"] = True
            if job["status"] == "queued":
                job["status"] = "stopped"
        return {
            "id": job["id"],
            "status": job["status"],
            "processed": job["processed"],
            "total": job["total"],
        }


@app.get("/api/image/{item_id}")
def get_image(item_id: str, job_id: str = Query(...)) -> FileResponse:
    path = _resolve_item_path(job_id, item_id)
    return FileResponse(path)


def _oriented_jpeg(path: Path, cache_file: Path, max_side: int, quality: int) -> None:
    if cache_file.exists():
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as im:
        im.draft(None, (max_side, max_side))
        im = ImageOps.exif_transpose(im) or im
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(cache_file, "JPEG", quality=quality)


@app.get("/api/preview/{item_id}")
def get_preview(item_id: str, job_id: str = Query(...), w: int = 1600) -> FileResponse:
    path = _resolve_item_path(job_id, item_id)
    w = max(400, min(int(w), 2000))
    cache_file = CACHE / f"{item_id}_prev_{w}.jpg"
    _oriented_jpeg(path, cache_file, w, 86)
    return FileResponse(cache_file, media_type="image/jpeg")


@app.get("/api/thumb/{item_id}")
def get_thumb(item_id: str, job_id: str = Query(...), w: int = 360) -> FileResponse:
    path = _resolve_item_path(job_id, item_id)
    w = max(80, min(w, 720))
    cache_file = CACHE / f"{item_id}_{w}_exif.jpg"
    _oriented_jpeg(path, cache_file, w, 82)
    return FileResponse(cache_file, media_type="image/jpeg")


@app.post("/api/export")
def export_images(req: ExportRequest) -> dict:
    with _jobs_lock:
        job = _jobs.get(req.job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        items = list(job["items"])
        src_folder = Path(job["folder"])

    allowed = ("text", "rest", "color_fail", "color_pass", "composition_match", "composition_rest", "crop")
    if req.kind not in allowed:
        raise HTTPException(400, "不支持的导出类型")
    suffix = {
        "text": "with_text",
        "rest": "no_text",
        "color_fail": "color_fail",
        "color_pass": "color_pass",
        "composition_match": f"comp_{_comp_slug(req.type_id)}",
        "composition_rest": "comp_unassigned",
        "crop": "cropped",
    }[req.kind]
    dest = Path(req.dest).expanduser() if req.dest else src_folder.parent / f"{src_folder.name}_{suffix}"
    dest = dest.resolve()
    if dest == src_folder.resolve():
        raise HTTPException(400, "目标目录不能和原文件夹相同")
    if req.kind == "crop":
        dest_uncropped = (
            Path(req.dest_uncropped).expanduser()
            if req.dest_uncropped
            else src_folder.parent / f"{src_folder.name}_uncropped"
        )
        dest_uncropped = dest_uncropped.resolve()
        if dest_uncropped == src_folder.resolve() or dest == dest_uncropped:
            raise HTTPException(400, "无需裁剪的目标目录不能和原文件夹或裁切文件夹相同")
        return _export_cropped(req.job_id, items, src_folder, dest, dest_uncropped, req.crop_boxes)

    moved = []
    skipped = []
    updates = []
    json_records = []
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for item in items:
            if not _export_match(
                item,
                req.kind,
                req.threshold,
                req.over_threshold,
                req.under_threshold,
                req.neon_threshold,
                req.similarity_threshold,
                req.type_id,
                req.type_thresholds,
            ):
                continue
            src = Path(item["path"])
            if not src.is_file():
                continue
            sidecar = sidecar_path(str(src))
            target = _move_file(src, dest)
            if sidecar.is_file():
                _move_file(sidecar, dest)
            updates.append((item["id"], str(target)))
            if src.resolve() == target:
                skipped.append(target.name)
            else:
                moved.append(target.name)
            if req.kind in ("color_fail", "color_pass"):
                record = _color_record(
                    item,
                    target.name,
                    req.over_threshold,
                    req.under_threshold,
                    req.neon_threshold,
                    passed=req.kind == "color_pass",
                )
                sidecar = dest / f"{target.stem}.json"
                sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                json_records.append(record)
            if req.kind in ("composition_match", "composition_rest"):
                json_records.append(_comp_record(item, target.name))
        if req.kind in ("color_fail", "color_pass"):
            index_name = "color_pass.json" if req.kind == "color_pass" else "color_fail.json"
            _write_color_index(dest, json_records, index_name)
        if req.kind in ("composition_match", "composition_rest"):
            index_name = (
                f"composition_{_comp_slug(req.type_id)}.json"
                if req.kind == "composition_match"
                else "composition_unassigned.json"
            )
            _write_color_index(dest, json_records, index_name)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise HTTPException(507, "磁盘空间不足，无法继续移动。请换到空间更大的位置。") from exc
        raise HTTPException(500, f"移动失败：{exc}") from exc

    if updates:
        with _jobs_lock:
            job = _jobs.get(req.job_id)
            if job:
                by_id = {it["id"]: it for it in job["items"]}
                for item_id, new_path in updates:
                    if item_id in by_id:
                        by_id[item_id]["path"] = new_path

    return {
        "dest": str(dest),
        "count": len(moved),
        "skipped": len(skipped),
        "files": moved,
    }


def _export_match(
    item: dict,
    kind: str,
    threshold: float,
    over_threshold: float = 0.10,
    under_threshold: float = 0.20,
    neon_threshold: float = 0.20,
    similarity_threshold: float = 0.58,
    type_id: Optional[str] = None,
    type_thresholds: Optional[Dict[str, float]] = None,
) -> bool:
    if kind == "color_fail":
        return _color_fails(item, over_threshold, under_threshold, neon_threshold)
    if kind == "color_pass":
        return not _color_fails(item, over_threshold, under_threshold, neon_threshold)
    if kind == "composition_match":
        return type_id in _item_kinds(item, type_thresholds)
    if kind == "composition_rest":
        return not _item_kinds(item, type_thresholds)
    if kind == "crop":
        return True
    is_text = float(item.get("text_ratio") or 0) + 1e-12 >= threshold
    if kind == "text":
        return is_text
    return not is_text


def _comp_thresholds(type_thresholds: Optional[Dict[str, float]] = None, fallback: float = 0.58) -> Dict[str, float]:
    known: Dict[str, float] = {}
    for typ in public_store().get("types") or []:
        type_id = typ.get("id")
        if not type_id:
            continue
        if type_thresholds and type_id in type_thresholds:
            value = type_thresholds[type_id]
        elif typ.get("threshold") is not None:
            value = typ["threshold"]
        else:
            value = fallback
        known[type_id] = max(0.0, min(1.0, float(value)))
    return known


def _comp_hits(
    item: dict,
    similarity_threshold: float,
    type_thresholds: Optional[Dict[str, float]] = None,
) -> List[dict]:
    known = type_thresholds if type_thresholds is not None else _comp_thresholds(None, similarity_threshold)
    if not known:
        return []
    names = {
        typ.get("id"): typ.get("name")
        for typ in (public_store().get("types") or [])
        if typ.get("id")
    }
    hits = []
    for type_id, row in (item.get("scores") or {}).items():
        if type_id not in known:
            continue
        score = float(row.get("score") or 0) if isinstance(row, dict) else float(row or 0)
        if score + 1e-12 < known[type_id]:
            continue
        hits.append(
            {
                "type_id": type_id,
                "name": (row.get("name") if isinstance(row, dict) else "") or names.get(type_id) or type_id,
                "score": score,
            }
        )
    hits.sort(key=lambda row: row["score"], reverse=True)
    return hits


def _comp_score(item: dict, type_id: Optional[str]) -> float:
    scores = item.get("scores") or {}
    if type_id and type_id in scores:
        row = scores[type_id]
        if isinstance(row, dict):
            return float(row.get("score") or 0)
        return float(row or 0)
    return float(item.get("similarity") or 0)


def _comp_slug(type_id: Optional[str]) -> str:
    if type_id in COMP_LABELS:
        return COMP_LABELS[type_id]
    for typ in public_store().get("types") or []:
        if typ.get("id") == type_id:
            return typ.get("name") or type_id or "comp"
    return type_id or "comp"


def _item_kinds(item: dict, type_thresholds: Optional[Dict[str, float]] = None) -> List[str]:
    known: List[str] = []
    for kind in item.get("qa_kinds") or []:
        if kind and kind not in known:
            known.append(kind)
    kinds = item.get("compositions")
    if isinstance(kinds, list):
        for kind in kinds:
            if kind and kind != "none" and kind not in known:
                known.append(kind)
    one = item.get("composition")
    if one and one != "none" and one not in known:
        known.append(one)
    scores = item.get("scores") or {}
    if scores:
        thresholds = type_thresholds if type_thresholds is not None else _comp_thresholds()
        for type_id, row in scores.items():
            if type_id in JUDGED_KINDS:
                continue
            score = row.get("score") if isinstance(row, dict) else row
            threshold = thresholds.get(type_id, 0.58)
            if float(score or 0) + 1e-12 >= float(threshold) and type_id not in known:
                known.append(type_id)
    return known


def _comp_record(item: dict, filename: str) -> dict:
    kinds = _item_kinds(item)
    names = item.get("composition_names") or [COMP_LABELS[k] for k in kinds]
    return {
        "file": filename,
        "构图": names,
        "compositions": kinds,
        "has_subject": bool(item.get("has_subject")),
        "subject_position": item.get("subject_position") or "none",
        "subject_position_name": item.get("subject_position_name") or "",
        "repeating": bool(item.get("repeating")),
        "reason": item.get("reason") or "",
        "size": {"width": item.get("width") or 0, "height": item.get("height") or 0},
    }


def _bright_ratio(item: dict) -> float:
    if item.get("bright_ratio") is not None:
        return float(item.get("bright_ratio") or 0)
    return max(float(item.get("white_ratio") or 0), float(item.get("over_ratio") or 0))


def _dark_ratio(item: dict) -> float:
    if item.get("dark_ratio") is not None:
        return float(item.get("dark_ratio") or 0)
    return max(float(item.get("black_ratio") or 0), float(item.get("under_ratio") or 0))


MAX_PASS_COLORS = 3


def _main_color_count(item: dict) -> int:
    return len(item.get("colors") or [])


def _harmony_ok(item: dict) -> bool:
    if (item.get("harmony_detail") or "") == "无法读取图片":
        return False
    if (item.get("harmony") or "none") != "none":
        return True
    return _main_color_count(item) <= MAX_PASS_COLORS


def _color_fails(
    item: dict,
    over_threshold: float,
    under_threshold: float,
    neon_threshold: float,
) -> bool:
    if _bright_ratio(item) + 1e-12 >= over_threshold:
        return True
    if _dark_ratio(item) + 1e-12 >= under_threshold:
        return True
    if float(item.get("neon_ratio") or 0) + 1e-12 >= neon_threshold:
        return True
    return not _harmony_ok(item)


def _color_record(
    item: dict,
    filename: str,
    over_threshold: float,
    under_threshold: float,
    neon_threshold: float,
    passed: bool = False,
) -> dict:
    reasons = []
    notes = []
    bright = _bright_ratio(item)
    dark = _dark_ratio(item)
    neon_over = float(item.get("neon_ratio") or 0) + 1e-12 >= neon_threshold
    harmony = item.get("harmony") or "none"
    if bright + 1e-12 >= over_threshold:
        reasons.append("过曝面积超标（已忽略白衬衫等固有色）")
    else:
        notes.append("过曝未超标（固有白未计入）")
    if dark + 1e-12 >= under_threshold:
        reasons.append("欠曝面积超标（已忽略黑发等固有色）")
    else:
        notes.append("欠曝未超标（固有黑未计入）")
    if neon_over:
        reasons.append("高饱和高明度纯色面积超标")
    else:
        notes.append("高饱和纯色未超标")
    color_count = _main_color_count(item)
    if not _harmony_ok(item):
        reasons.append("主色超过3块，且不是对比色、互补色、邻近色、相邻色或同色系")
    elif harmony == "none":
        notes.append(f"主色 {color_count} 块（不超过3块），配色通过")
    else:
        notes.append(item.get("harmony_detail") or item.get("harmony_name") or "配色合格")
    return {
        "file": filename,
        "fail": not passed,
        "pass": passed,
        "reasons": [] if passed else reasons,
        "notes": notes if passed else [],
        "bright_ratio": bright,
        "dark_ratio": dark,
        "white_ratio": item.get("white_ratio"),
        "black_ratio": item.get("black_ratio"),
        "over_ratio": item.get("over_ratio"),
        "under_ratio": item.get("under_ratio"),
        "neon_ratio": item.get("neon_ratio"),
        "harmony": item.get("harmony"),
        "harmony_name": item.get("harmony_name"),
        "harmony_detail": item.get("harmony_detail"),
        "colors": item.get("colors") or [],
        "color_count": color_count,
        "thresholds": {"over": over_threshold, "under": under_threshold, "neon": neon_threshold, "max_colors": MAX_PASS_COLORS},
        "size": {"width": item.get("width") or 0, "height": item.get("height") or 0},
    }


def _write_color_index(dest: Path, new_records: List[dict], index_name: str = "color_fail.json") -> None:
    index_path = dest / index_name
    existing = []
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            existing = payload.get("items") or []
        except json.JSONDecodeError:
            existing = []
    by_file = {row.get("file"): row for row in existing if row.get("file")}
    for row in new_records:
        by_file[row["file"]] = row
    items = list(by_file.values())
    index_path.write_text(
        json.dumps({"count": len(items), "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _export_cropped(
    job_id: str,
    items: List[dict],
    src_folder: Path,
    dest: Path,
    dest_uncropped: Path,
    crop_boxes: Optional[Dict[str, List[int]]] = None,
) -> dict:
    saved = []
    copied = []
    failed = []
    updates = []
    overrides = crop_boxes or {}
    try:
        for item in items:
            src = Path(item.get("path") or "")
            if not src.is_file():
                continue
            rel = Path(item.get("rel") or src.name)
            if rel.is_absolute() or ".." in rel.parts:
                rel = Path(src.name)
            override = overrides.get(item.get("id") or "")
            if override is not None:
                _set_item_crop_box(item, override, True)
            box = item.get("crop_box")
            will_crop = bool(item.get("cropped") and isinstance(box, list) and len(box) >= 4)
            target_root = dest if will_crop else dest_uncropped
            target = target_root / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if will_crop:
                    path = save_cropped(str(src), str(target), (int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                    saved.append(Path(path).name)
                else:
                    if src.resolve() != target.resolve():
                        shutil.copy2(src, target)
                    copied.append(target.name)
                updates.append((item["id"], str(target.resolve())))
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{src.name}：{exc}")
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise HTTPException(507, "磁盘空间不足，无法继续保存。请换到空间更大的位置。") from exc
        raise HTTPException(500, f"保存失败：{exc}") from exc
    if updates:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                by_id = {it["id"]: it for it in job["items"]}
                for item_id, new_path in updates:
                    if item_id in by_id:
                        by_id[item_id]["saved"] = new_path
    if failed and not saved and not copied:
        raise HTTPException(500, f"保存失败：{failed[0]}")
    return {
        "dest": str(dest),
        "dest_uncropped": str(dest_uncropped),
        "count": len(saved) + len(copied),
        "cropped": len(saved),
        "copied": len(copied),
        "skipped": 0,
        "failed": len(failed),
        "files": saved + copied,
    }


def _move_file(src: Path, dest_dir: Path) -> Path:
    src = src.resolve()
    dest_dir = dest_dir.resolve()
    target = dest_dir / src.name
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == src:
                return target
        except OSError:
            pass
        target.unlink()
    try:
        src.rename(target)
    except OSError:
        shutil.move(str(src), str(target))
    return target.resolve()


def _resolve_item_path(job_id: str, item_id: str) -> Path:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        for item in job["items"]:
            if item["id"] == item_id:
                path = Path(item["path"])
                if not path.is_file():
                    raise HTTPException(404, "文件已不存在")
                return path
    raise HTTPException(404, "图片不存在")


@app.get("/api/lut/library")
def lut_library() -> dict:
    return list_luts()


@app.post("/api/lut/generate")
def lut_generate(req: LutGenerateRequest) -> dict:
    try:
        return create_or_update_lut(
            name=req.name,
            description=req.description,
            ref_folder=req.folder,
            lut_id=req.lut_id,
            use_saved_base=bool(req.regenerate),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.patch("/api/lut/{lut_id}")
def lut_rename(lut_id: str, req: LutRenameRequest) -> dict:
    try:
        lut = rename_lut(lut_id, req.name, req.description)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"lut": lut, "library": list_luts()}


@app.post("/api/lut/{lut_id}/activate")
def lut_activate(lut_id: str) -> dict:
    try:
        return set_active(lut_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/lut/{lut_id}")
def lut_delete(lut_id: str) -> dict:
    try:
        return delete_lut(lut_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/lut/{lut_id}/hald")
def lut_hald(lut_id: str) -> FileResponse:
    try:
        path = hald_path(lut_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="image/png")


@app.get("/api/lut/{lut_id}/cube")
def lut_cube(lut_id: str) -> FileResponse:
    try:
        path = cube_path(lut_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, filename=f"{lut_id}.cube", media_type="text/plain")


@app.post("/api/lut/strengths")
def lut_strengths(req: LutStrengthRequest) -> dict:
    with _jobs_lock:
        job = _jobs.get(req.job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        by_id = {it["id"]: it for it in job.get("items") or []}
        updated = 0
        if req.all_strength is not None:
            for item in by_id.values():
                item["strength"] = float(req.all_strength)
                updated += 1
        if req.strengths:
            for item_id, value in req.strengths.items():
                if item_id in by_id:
                    by_id[item_id]["strength"] = float(max(0.0, min(1.5, value)))
                    updated += 1
        if req.item_id and req.strength is not None and req.item_id in by_id:
            by_id[req.item_id]["strength"] = float(req.strength)
            updated += 1
        items = [_public_item(it, "lut") for it in job.get("items") or []]
    return {"updated": updated, "items": items}


def _norm_layers(layers) -> list:
    out = []
    for layer in layers or []:
        if hasattr(layer, "model_dump"):
            layer = layer.model_dump()
        lut_id = (layer.get("lut_id") or "").strip()
        if not lut_id:
            continue
        item = {
            "lut_id": lut_id,
            "strength": float(max(0.0, min(1.5, layer.get("strength") if layer.get("strength") is not None else 1.0))),
            "enabled": layer.get("enabled") is not False,
            "apply": (layer.get("apply") or "auto").lower(),
        }
        for key in ("hue", "sat", "gamma", "contrast", "hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue", "hue_magenta"):
            if layer.get(key) is not None:
                item[key] = float(layer[key])
        out.append(item)
    return out


def _layers_for_item(item: dict, fallback: list) -> list:
    overrides = item.get("layer_strengths") or {}
    if not overrides:
        return fallback
    layers = []
    for layer in fallback:
        copy = dict(layer)
        if copy["lut_id"] in overrides:
            copy["strength"] = float(overrides[copy["lut_id"]])
        layers.append(copy)
    return layers


def _stack_cache_key(layers: list, grade: Optional[dict] = None) -> str:
    raw = ",".join(
        f"{x['lut_id']}:{int(round(x['strength'] * 100))}:{1 if x.get('enabled', True) else 0}:"
        f"{x.get('apply') or 'auto'}:{x.get('hue')}:{x.get('sat')}:{x.get('gamma')}:{x.get('contrast')}:"
        f"{x.get('hue_red')}:{x.get('hue_yellow')}:{x.get('hue_green')}:{x.get('hue_cyan')}:{x.get('hue_blue')}:{x.get('hue_magenta')}"
        for x in layers
    )
    if grade:
        raw += (
            f"|{grade.get('mode') or 'auto'}:"
            f"{grade.get('hue')}:"
            f"{grade.get('sat')}:"
            f"{grade.get('gamma')}:"
            f"{grade.get('contrast')}"
        )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _grade_dict(grade) -> Optional[dict]:
    if grade is None:
        return None
    if hasattr(grade, "model_dump"):
        grade = grade.model_dump()
    return {
        "mode": (grade.get("mode") or "auto").lower(),
        "hue": grade.get("hue"),
        "sat": grade.get("sat"),
        "gamma": grade.get("gamma"),
        "contrast": grade.get("contrast"),
    }


@app.get("/api/lut/preview/{item_id}")
def lut_preview(
    item_id: str,
    job_id: str = Query(...),
    lut_id: str = "",
    strength: float = 1.0,
    stack: str = "",
    w: int = 720,
    grade: str = "auto",
    hue: Optional[float] = None,
    sat: Optional[float] = None,
    gamma: Optional[float] = None,
    contrast: Optional[float] = None,
) -> FileResponse:
    path = _resolve_item_path(job_id, item_id)
    layers = parse_stack_query(stack)
    if not layers and lut_id:
        layers = [{"lut_id": lut_id, "strength": max(0.0, min(1.5, float(strength))), "enabled": True}]
    if not layers:
        raise HTTPException(400, "还没有叠加层")
    w = max(80, min(int(w), 4000))
    grade_spec = {"mode": grade, "hue": hue, "sat": sat, "gamma": gamma, "contrast": contrast}
    cache_file = CACHE / f"lut_{item_id}_{w}_{_stack_cache_key(layers, grade_spec)}_c11.jpg"
    if not cache_file.exists():
        try:
            apply_layers(path, cache_file, layers, max_side=w, grade=grade_spec)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"套 LUT 失败：{exc}") from exc
    return FileResponse(cache_file, media_type="image/jpeg")


@app.get("/api/lut/suggest/{item_id}")
def lut_suggest(item_id: str, job_id: str = Query(...), stack: str = "") -> dict:
    path = _resolve_item_path(job_id, item_id)
    layers = parse_stack_query(stack)
    if not layers:
        raise HTTPException(400, "还没有叠加层")
    try:
        rgb, _meta, _room, space = load_preview_source(path, 480)
        plan = resolve_grade(rgb, layers, {"mode": "auto"}, space=space)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"分析失败：{exc}") from exc
    space = plan["space"]
    return {
        "mode": space["mode"],
        "used": plan["used"],
        "reason": space["reason"],
        "score": space["score"],
        "room_mean": space["room_mean"],
        "sky_frac": space["sky_frac"],
        "layers": plan["layers"],
    }


def _icc_is_srgb(icc: Optional[bytes]) -> bool:
    if not icc:
        return True
    try:
        from io import BytesIO

        from PIL import ImageCms

        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(icc)))
    except Exception:  # noqa: BLE001
        return False
    return "srgb" in (desc or "").lower().replace(" ", "")


def _png_response(arr) -> Response:
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(arr).save(buf, "PNG", compress_level=1)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/lut/live/{item_id}")
def lut_live_info(item_id: str, job_id: str = Query(...), w: int = 1200) -> dict:
    path = _resolve_item_path(job_id, item_id)
    w = max(80, min(int(w), 4000))
    rgb, meta, _room, space = load_preview_source(path, w)
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "mode": space["mode"],
        "live": _icc_is_srgb(meta.get("icc")),
    }


@app.get("/api/lut/live/{item_id}/src")
def lut_live_src(item_id: str, job_id: str = Query(...), w: int = 1200) -> Response:
    path = _resolve_item_path(job_id, item_id)
    rgb, meta, _room, _space = load_preview_source(path, max(80, min(int(w), 4000)))
    u8 = meta.get("u8")
    if u8 is None or u8.shape != rgb.shape:
        u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return _png_response(u8)


@app.get("/api/lut/live/{item_id}/room")
def lut_live_room(item_id: str, job_id: str = Query(...), w: int = 1200) -> Response:
    path = _resolve_item_path(job_id, item_id)
    _rgb, _meta, room, _space = load_preview_source(path, max(80, min(int(w), 4000)))
    return _png_response(np.clip(room * 255.0 + 0.5, 0, 255).astype(np.uint8))


_LUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _lut_target(dest: Path, item: dict) -> Optional[Path]:
    src = Path(item.get("path") or "")
    if not src.is_file():
        return None
    rel = Path(item.get("rel") or src.name)
    if rel.is_absolute() or ".." in rel.parts:
        rel = Path(src.name)
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() not in _LUT_SUFFIXES:
        target = target.with_suffix(".jpg")
    return target


def _export_one_file(item: dict, dest: Path, fallback: list, per_item: dict, grades: Optional[dict] = None) -> Path:
    src = Path(item.get("path") or "")
    target = _lut_target(dest, item)
    if target is None:
        raise FileNotFoundError(src.name or "文件不存在")
    layers = _norm_layers(per_item.get(item.get("id") or "")) or _layers_for_item(item, fallback)
    if not layers:
        raise ValueError(f"{src.name}：没有叠加层")
    grade = None
    if grades:
        grade = _grade_dict(grades.get(item.get("id") or ""))
    return apply_layers(src, target, layers, max_side=None, grade=grade)


def _export_lut_items(
    job_id: str,
    item_ids: Optional[List[str]],
    dest: Path,
    fallback: list,
    item_layers: Optional[dict] = None,
    item_grades: Optional[dict] = None,
) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        items = list(job.get("items") or [])
        folder = Path(job.get("folder") or "")
    if item_ids:
        want = set(item_ids)
        items = [it for it in items if it.get("id") in want]
    if not items:
        raise HTTPException(400, "没有可导出的图片")
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    failed = []
    per_item = item_layers or {}
    workers = min(4, max(1, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        grades = item_grades or {}
        futs = {pool.submit(_export_one_file, item, dest, fallback, per_item, grades): item for item in items}
        for fut in as_completed(futs):
            item = futs[fut]
            try:
                saved.append(fut.result().name)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{item.get('name') or '图'}：{exc}")
    if not saved:
        raise HTTPException(500, failed[0] if failed else "导出失败")
    return {"dest": str(dest), "count": len(saved), "failed": len(failed), "files": saved, "folder": str(folder)}


@app.post("/api/lut/export")
def lut_export_folder(req: LutExportRequest) -> dict:
    with _jobs_lock:
        job = _jobs.get(req.job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        src_folder = Path(job["folder"])
    dest = Path(req.dest).expanduser() if (req.dest or "").strip() else src_folder.parent / f"{src_folder.name}_lut"
    if dest.resolve() == src_folder.resolve():
        raise HTTPException(400, "导出目录不能和原文件夹相同")
    fallback = _norm_layers(req.layers)
    if not fallback and req.lut_id:
        fallback = [{"lut_id": req.lut_id, "strength": 1.0, "enabled": True}]
    try:
        return _export_lut_items(req.job_id, req.item_ids, dest, fallback, req.item_layers, req.item_grades)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc


def _lut_one_response(path: Path, layers: list, grade: Optional[dict] = None) -> FileResponse:
    suffix = path.suffix.lower() if path.suffix.lower() in _LUT_SUFFIXES else ".jpg"
    tmp = CACHE / f"lut_one_{path.stem}_{uuid.uuid4().hex[:8]}{suffix}"
    try:
        out = apply_layers(path, tmp, layers, max_side=None, grade=grade)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"导出失败：{exc}") from exc
    filename = f"{path.stem}_lut{out.suffix}"
    media = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
    }.get(out.suffix.lower(), "image/jpeg")
    return FileResponse(out, filename=filename, media_type=media)


@app.post("/api/lut/export-one")
def lut_export_one_post(req: LutExportOneBody) -> FileResponse:
    path = _resolve_item_path(req.job_id, req.item_id)
    layers = _norm_layers(req.layers) or parse_stack_query(req.stack or "")
    if not layers and req.lut_id:
        layers = [{"lut_id": req.lut_id, "strength": 1.0, "enabled": True}]
    if not layers:
        raise HTTPException(400, "还没有叠加层")
    return _lut_one_response(path, layers, _grade_dict(req.grade))


@app.get("/api/lut/export-one/{item_id}")
def lut_export_one(
    item_id: str,
    job_id: str = Query(...),
    lut_id: str = "",
    stack: str = "",
    strength: float = 1.0,
) -> FileResponse:
    path = _resolve_item_path(job_id, item_id)
    layers = parse_stack_query(stack)
    if not layers and lut_id:
        layers = [{"lut_id": lut_id, "strength": max(0.0, min(1.5, float(strength))), "enabled": True}]
    if not layers:
        raise HTTPException(400, "还没有叠加层")
    return _lut_one_response(path, layers)


@app.post("/api/lut/export-zip")
def lut_export_zip(req: LutExportRequest) -> FileResponse:
    with _jobs_lock:
        job = _jobs.get(req.job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        items = list(job.get("items") or [])
        folder_name = Path(job.get("folder") or "lut").name
    if req.item_ids:
        want = set(req.item_ids)
        items = [it for it in items if it.get("id") in want]
    if not items:
        raise HTTPException(400, "没有可导出的图片")
    fallback = _norm_layers(req.layers)
    if not fallback and req.lut_id:
        fallback = [{"lut_id": req.lut_id, "strength": 1.0, "enabled": True}]
    if not fallback:
        raise HTTPException(400, "还没有叠加层")
    zip_path = CACHE / f"lut_zip_{req.job_id}_{uuid.uuid4().hex[:8]}.zip"
    CACHE.mkdir(parents=True, exist_ok=True)
    wrote = 0
    per_item = req.item_layers or {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            workers = min(4, max(1, len(items)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                grades = req.item_grades or {}
                futs = {pool.submit(_export_one_file, item, tmp_dir, fallback, per_item, grades): item for item in items}
                for fut in as_completed(futs):
                    try:
                        out = fut.result()
                    except Exception:
                        continue
                    arc = str(out.relative_to(tmp_dir))
                    zf.write(out, arcname=arc)
                    wrote += 1
    if wrote == 0:
        raise HTTPException(500, "打包失败，没有成功套上 LUT 的图片")
    return FileResponse(zip_path, filename=f"{folder_name}_lut.zip", media_type="application/zip")


def main() -> None:
    global DEFAULT_FOLDER
    parser = argparse.ArgumentParser(description="图片文字占比分类与预览")
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="默认扫描目录")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    DEFAULT_FOLDER = str(Path(args.folder).expanduser())

    import uvicorn

    print(f"打开 http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
