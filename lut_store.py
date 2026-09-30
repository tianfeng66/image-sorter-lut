"""Persist generated LUTs so they can be reused later."""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lut_engine import (
    CUBE_SIZE,
    analyze_grade_space,
    apply_adjust_room,
    apply_cube,
    apply_look,
    apply_tune,
    build_cube_rgb,
    build_hald,
    default_tune,
    desaturate,
    estimate_adjust_room,
    generate_look,
    list_images,
    load_cube_file,
    load_rgb,
    save_cube_file,
    save_hald_png,
    tune_from_params,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / ".data" / "luts"
INDEX = DATA / "library.json"
SEED_DIR = ROOT / "luts"
_lock = threading.Lock()
_cube_cache: Dict[str, np.ndarray] = {}
_params_cache: Dict[str, Optional[dict]] = {}
_SOURCE_CACHE_BYTES = 384 * 1024 * 1024
_source_cache: "OrderedDict[tuple, Tuple[np.ndarray, dict, np.ndarray, dict, int]]" = OrderedDict()
_source_cache_size = 0
_source_lock = threading.Lock()


def _empty() -> dict:
    return {"luts": [], "active_id": ""}


def _read() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    if not INDEX.exists():
        return _empty()
    try:
        data = json.loads(INDEX.read_text(encoding="utf-8"))
    except Exception:
        return _empty()
    data.setdefault("luts", [])
    data.setdefault("active_id", "")
    return data


def _write(data: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _lut_dir(lut_id: str) -> Path:
    return DATA / lut_id


def _public(entry: dict) -> dict:
    lut_id = entry["id"]
    folder = _lut_dir(lut_id)
    return {
        "id": lut_id,
        "name": entry.get("name") or "未命名 LUT",
        "description": entry.get("description") or "",
        "summary": entry.get("summary") or "",
        "notes": entry.get("notes") or [],
        "from_refs": bool(entry.get("from_refs")),
        "from_desc": bool(entry.get("from_desc")),
        "swatches": entry.get("swatches") or [],
        "ref_folder": entry.get("ref_folder") or "",
        "ref_count": int(entry.get("ref_count") or 0),
        "created_at": entry.get("created_at") or "",
        "updated_at": entry.get("updated_at") or "",
        "has_hald": (folder / "hald.png").is_file(),
        "has_cube": (folder / "look.cube").is_file(),
        "tune": tune_from_params(entry.get("params") or {}),
        "params": entry.get("params") or None,
    }


def ensure_seed() -> dict:
    with _lock:
        data = _read()
        if data["luts"]:
            return data
        seeds = [
            ("cyan_lake", "青湖 Cyan Lake", "青绿水色、冷白高光"),
            ("spring_green", "春绿 Spring Green", "黄绿叶子、偏蓝天空"),
        ]
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for key, name, desc in seeds:
            src_cube = SEED_DIR / f"lut_{key}.cube"
            src_hald = SEED_DIR / f"lut_{key}_hald.png"
            if not src_cube.is_file():
                continue
            lut_id = uuid.uuid4().hex[:12]
            dest = _lut_dir(lut_id)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_cube, dest / "look.cube")
            if src_hald.is_file():
                shutil.copy2(src_hald, dest / "hald.png")
            data["luts"].append(
                {
                    "id": lut_id,
                    "name": name,
                    "description": desc,
                    "summary": desc,
                    "notes": ["从上一组蓝绿参考图生成"],
                    "swatches": [],
                    "ref_folder": "",
                    "ref_count": 0,
                    "created_at": now,
                    "updated_at": now,
                    "params": {},
                    "base_params": {},
                }
            )
            if not data["active_id"]:
                data["active_id"] = lut_id
        _write(data)
        return data


def list_luts() -> dict:
    data = ensure_seed()
    return {
        "luts": [_public(x) for x in data["luts"]],
        "active_id": data.get("active_id") or "",
        "ref_folder": str(SEED_DIR / "refs_teal") if (SEED_DIR / "refs_teal").is_dir() else "",
    }


def get_entry(lut_id: str) -> dict:
    data = ensure_seed()
    for entry in data["luts"]:
        if entry["id"] == lut_id:
            return entry
    raise KeyError("找不到这张 LUT")


def hald_path(lut_id: str) -> Path:
    path = _lut_dir(lut_id) / "hald.png"
    if not path.is_file():
        raise FileNotFoundError("还没有 LUT 图")
    return path


def cube_path(lut_id: str) -> Path:
    path = _lut_dir(lut_id) / "look.cube"
    if not path.is_file():
        raise FileNotFoundError("还没有 .cube 文件")
    return path


def load_cube(lut_id: str) -> np.ndarray:
    cached = _cube_cache.get(lut_id)
    if cached is not None:
        return cached
    volume = load_cube_file(cube_path(lut_id))
    _cube_cache[lut_id] = volume
    return volume


def load_params(lut_id: str) -> Optional[dict]:
    if lut_id in _params_cache:
        return _params_cache[lut_id]
    try:
        params = get_entry(lut_id).get("params") or None
    except KeyError:
        params = None
    if not params:
        params = None
    _params_cache[lut_id] = params
    return params


def set_active(lut_id: str) -> dict:
    with _lock:
        data = _read()
        if lut_id and not any(x["id"] == lut_id for x in data["luts"]):
            raise KeyError("找不到这张 LUT")
        data["active_id"] = lut_id
        _write(data)
    return list_luts()


def rename_lut(lut_id: str, name: str, description: Optional[str] = None) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("请填写 LUT 名称")
    with _lock:
        data = _read()
        for entry in data["luts"]:
            if entry["id"] != lut_id:
                continue
            entry["name"] = name
            if description is not None:
                entry["description"] = description
            entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write(data)
            return _public(entry)
    raise KeyError("找不到这张 LUT")


def delete_lut(lut_id: str) -> dict:
    with _lock:
        data = _read()
        data["luts"] = [x for x in data["luts"] if x["id"] != lut_id]
        if data.get("active_id") == lut_id:
            data["active_id"] = data["luts"][0]["id"] if data["luts"] else ""
        _write(data)
    shutil.rmtree(_lut_dir(lut_id), ignore_errors=True)
    _cube_cache.pop(lut_id, None)
    _params_cache.pop(lut_id, None)
    return list_luts()


def _write_assets(lut_id: str, name: str, params: dict) -> None:
    dest = _lut_dir(lut_id)
    dest.mkdir(parents=True, exist_ok=True)
    hald = build_hald(params)
    volume = build_cube_rgb(params, CUBE_SIZE)
    save_hald_png(dest / "hald.png", hald)
    save_cube_file(dest / "look.cube", volume, name)
    _cube_cache[lut_id] = volume
    _params_cache.pop(lut_id, None)


def create_or_update_lut(
    name: str,
    description: str,
    ref_folder: str = "",
    lut_id: Optional[str] = None,
    use_saved_base: bool = False,
) -> dict:
    name = (name or "").strip() or "未命名 LUT"
    description = (description or "").strip()
    refs: List[Path] = []
    folder = (ref_folder or "").strip()
    if folder:
        path = Path(folder).expanduser()
        if not path.is_dir():
            raise ValueError(f"参考文件夹不存在：{path}")
        refs = list_images(path, recursive=False)
        if not refs:
            raise ValueError("参考文件夹里没有图片")

    base_params = None
    if lut_id and use_saved_base:
        try:
            existing = get_entry(lut_id)
            base_params = existing.get("base_params") or existing.get("params")
        except KeyError:
            lut_id = None

    if not refs and not description and not base_params:
        raise ValueError("请提供参考图文件夹，或填写 LUT 特性描述")

    look = generate_look(refs, description, base_params=base_params)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        data = _read()
        entry = None
        if lut_id:
            for item in data["luts"]:
                if item["id"] == lut_id:
                    entry = item
                    break
        if entry is None:
            lut_id = uuid.uuid4().hex[:12]
            entry = {"id": lut_id, "created_at": now}
            data["luts"].insert(0, entry)
        entry.update(
            {
                "name": name,
                "description": description,
                "summary": look["summary"],
                "notes": look["notes"],
                "from_refs": look.get("from_refs"),
                "from_desc": look.get("from_desc"),
                "swatches": look["swatches"],
                "ref_folder": str(Path(folder).expanduser()) if folder else entry.get("ref_folder") or "",
                "ref_count": len(refs) or int(entry.get("ref_count") or 0),
                "updated_at": now,
                "params": look["params"],
                "base_params": look["base_params"],
            }
        )
        data["active_id"] = lut_id
        _write(data)
    _write_assets(lut_id, name, look["params"])
    return {"lut": _public(get_entry(lut_id)), "library": list_luts()}


def apply_to_image(lut_id: str, source: Path, dest: Path, strength: float, max_side: Optional[int] = None) -> Path:
    return apply_layers(source, dest, [{"lut_id": lut_id, "strength": strength, "enabled": True}], max_side)


def _lut_name(lut_id: str) -> str:
    try:
        return get_entry(lut_id).get("name") or ""
    except KeyError:
        return ""


def _stack_skin_only(layers: list) -> bool:
    return any("绿调" in _lut_name((layer.get("lut_id") or "").strip()) for layer in (layers or []) if layer.get("enabled") is not False)


def _layer_tune(layer: dict) -> dict:
    params = load_params((layer.get("lut_id") or "").strip())
    tune = tune_from_params(params) if params else default_tune()
    for key in ("hue", "sat", "gamma", "contrast", "black", "hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue", "hue_magenta"):
        if layer.get(key) is not None:
            tune[key] = float(layer[key])
    return tune


def _layer_apply(layer: dict, image_used: str) -> str:
    apply = (layer.get("apply") or "auto").lower()
    if apply not in {"auto", "lut", "tune"}:
        apply = "auto"
    if apply == "auto":
        return "tune" if image_used == "tune" else "lut"
    return apply


def _apply_one_layer(rgb: np.ndarray, layer: dict, image_used: str) -> np.ndarray:
    strength = float(layer.get("strength") or 0.0)
    lut_id = (layer.get("lut_id") or "").strip()
    if not lut_id or strength <= 1e-6:
        return rgb
    if _layer_apply(layer, image_used) == "tune":
        return apply_tune(rgb, _layer_tune(layer), strength)
    params = load_params(lut_id)
    if params:
        return apply_look(rgb, params, strength, protect_wash="绿调" not in _lut_name(lut_id))
    try:
        return apply_cube(rgb, load_cube(lut_id), strength)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到 LUT 文件：{lut_id}") from exc


def _apply_stack(rgb: np.ndarray, layers: list, image_used: str = "lut") -> np.ndarray:
    out = rgb
    for layer in layers or []:
        if layer.get("enabled") is False:
            continue
        out = _apply_one_layer(out, layer, image_used)
    return out


def _probe_look_cast(layers: list, image_used: str = "lut") -> np.ndarray:
    probe = np.zeros((12, 12, 3), dtype=np.float32)
    probe[:4] = 0.32
    probe[4:8] = 0.50
    probe[8:] = 0.68
    graded = _apply_stack(probe, layers, image_used)
    return (graded - probe).mean(axis=(0, 1)).astype(np.float32)


def _analyze_source(rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
    room = estimate_adjust_room(rgb)
    return room, analyze_grade_space(rgb, room)


def load_preview_source(source: Path, max_side: int) -> Tuple[np.ndarray, dict, np.ndarray, dict]:
    """Downscaled source plus its room/space analysis, reused while sliders move."""
    global _source_cache_size
    stat = Path(source).stat()
    key = (str(source), stat.st_mtime_ns, stat.st_size, int(max_side))
    with _source_lock:
        hit = _source_cache.get(key)
        if hit is not None:
            _source_cache.move_to_end(key)
            return hit[0], dict(hit[1]), hit[2], hit[3]
    meta: dict = {}
    rgb = load_rgb(Path(source), max_side=max_side, meta=meta)
    room, space = _analyze_source(rgb)
    for arr in (rgb, room, meta.get("u8"), meta.get("u16")):
        if isinstance(arr, np.ndarray):
            arr.setflags(write=False)
    size = sum(a.nbytes for a in (rgb, room, meta.get("u8"), meta.get("u16")) if isinstance(a, np.ndarray))
    with _source_lock:
        if key not in _source_cache:
            _source_cache[key] = (rgb, meta, room, space, size)
            _source_cache_size += size
            while _source_cache_size > _SOURCE_CACHE_BYTES and len(_source_cache) > 1:
                _old_key, old = _source_cache.popitem(last=False)
                _source_cache_size -= old[4]
    return rgb, dict(meta), room, space


def resolve_grade(rgb, layers: list, grade: Optional[dict] = None, space: Optional[dict] = None) -> dict:
    if space is None:
        space = analyze_grade_space(rgb)
    spec = grade or {}
    mode = (spec.get("mode") or "auto").lower()
    if mode not in {"auto", "lut", "tune"}:
        mode = "auto"
    used = space["mode"] if mode == "auto" else mode
    layer_tunes = []
    for layer in layers or []:
        if layer.get("enabled") is False:
            continue
        lut_id = (layer.get("lut_id") or "").strip()
        if not lut_id:
            continue
        layer_tunes.append(
            {
                "lut_id": lut_id,
                "apply": _layer_apply(layer, used),
                **_layer_tune(layer),
            }
        )
    return {"space": space, "mode": mode, "used": used, "layers": layer_tunes}


def apply_layers(
    source: Path,
    dest: Path,
    layers: list,
    max_side: Optional[int] = None,
    grade: Optional[dict] = None,
) -> Path:
    from lut_engine import save_rgb

    if max_side:
        source_rgb, meta, room, space = load_preview_source(source, max_side)
    else:
        meta = {}
        source_rgb = load_rgb(source, max_side=None, meta=meta)
        room, space = _analyze_source(source_rgb)
    if (grade or {}).get("desat"):
        # Room/space stay on the color original: skin and sky are only detectable there.
        source_rgb = desaturate(source_rgb)
        meta.pop("u8", None)
        meta.pop("u16", None)
    keep = None
    plan = resolve_grade(source_rgb, layers, grade, space=space)
    rgb = _apply_stack(source_rgb, layers, plan["used"])
    if source_rgb.shape == rgb.shape:
        all_tune = all(_layer_apply(layer, plan["used"]) == "tune" for layer in (layers or []) if layer.get("enabled") is not False)
        rgb, keep = apply_adjust_room(
            source_rgb,
            rgb,
            look_cast=_probe_look_cast(layers, plan["used"]),
            skin_only=all_tune or _stack_skin_only(layers),
            room=room,
        )
    if source_rgb.shape == rgb.shape and keep is not None:
        meta["keep"] = keep
    return save_rgb(dest, rgb, source.suffix, meta=meta, preview=max_side is not None)


def parse_stack_query(raw: str) -> list:
    layers = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        bits = part.split(":")
        lut_id = bits[0].strip()
        if not lut_id:
            continue
        try:
            strength = max(0.0, min(1.5, float(bits[1]) / 100.0))
        except (ValueError, IndexError):
            strength = 1.0
        layer = {"lut_id": lut_id, "strength": strength, "enabled": True, "apply": "auto"}
        if len(bits) >= 3 and bits[2] in {"auto", "lut", "tune"}:
            layer["apply"] = bits[2]
        if len(bits) >= 7:
            try:
                layer["hue"] = float(bits[3])
                layer["sat"] = float(bits[4])
                layer["gamma"] = float(bits[5])
                layer["contrast"] = float(bits[6])
            except ValueError:
                pass
        if len(bits) >= 13:
            try:
                for key, raw in zip(
                    ("hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue", "hue_magenta"),
                    bits[7:13],
                ):
                    layer[key] = float(raw)
            except ValueError:
                pass
        layers.append(layer)
    return layers
