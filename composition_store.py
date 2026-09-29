"""Persist composition types, reference copies, and extracted features."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import List, Optional, Set, Tuple

from composition import COMP_LABELS, extract_features, pack_features
from composition_llm import llm_status
from detector import is_image_name

ROOT = Path(__file__).resolve().parent
DATA = ROOT / ".data"
STORE = DATA / "composition.json"
REFS = DATA / "composition_refs"
_lock = threading.Lock()
BUILTIN_TYPES = (("center", "中心构图"),)
BUILTIN_IDS = {tid for tid, _ in BUILTIN_TYPES}


def _empty() -> dict:
    return {"active_type_id": "", "threshold": 0.58, "types": []}


def load_store() -> dict:
    with _lock:
        return _read()


def save_settings(active_type_id: Optional[str] = None, threshold: Optional[float] = None) -> dict:
    with _lock:
        data = _read()
        if active_type_id is not None:
            data["active_type_id"] = active_type_id
        if threshold is not None:
            data["threshold"] = max(0.0, min(1.0, float(threshold)))
        _write(data)
        return data


def ensure_builtin_types() -> dict:
    with _lock:
        data = _read()
        name_to_id = {name: type_id for type_id, name in BUILTIN_TYPES}
        by_id = {}
        leftovers = []
        changed = False
        for typ in data["types"]:
            type_id = typ.get("id")
            if type_id in BUILTIN_IDS:
                by_id[type_id] = typ
            else:
                leftovers.append(typ)
        for type_id, name in BUILTIN_TYPES:
            if type_id in by_id:
                continue
            by_id[type_id] = {
                "id": type_id,
                "name": name,
                "description": "",
                "threshold": 0.58,
                "refs": [],
            }
            changed = True
        kept = []
        for typ in leftovers:
            target_id = name_to_id.get(typ.get("name") or "")
            if not target_id:
                kept.append(typ)
                continue
            target = by_id[target_id]
            if not (target.get("description") or "").strip() and typ.get("description"):
                target["description"] = typ.get("description") or ""
            if typ.get("threshold") is not None:
                target["threshold"] = typ["threshold"]
            target.setdefault("refs", []).extend(typ.get("refs") or [])
            changed = True
        data["types"] = [by_id[type_id] for type_id, _ in BUILTIN_TYPES] + kept
        removed_dirs: List[Path] = []
        if not data.get("center_only_cleanup"):
            stay = []
            for typ in data["types"]:
                if typ.get("id") == "center":
                    stay.append(typ)
                else:
                    removed_dirs.append(REFS / typ["id"])
                    changed = True
            data["types"] = stay
            data["active_type_id"] = "center"
            data["center_only_cleanup"] = True
        removed_files: List[Path] = []
        for typ in data["types"]:
            did_change, paths = _dedupe_type_refs(typ)
            if did_change:
                changed = True
                removed_files.extend(paths)
        if changed:
            _write(data)
    for path in removed_files:
        if path.is_file():
            path.unlink(missing_ok=True)
    for folder in removed_dirs:
        shutil.rmtree(folder, ignore_errors=True)
    return data


def public_store() -> dict:
    data = ensure_builtin_types()
    types = [_public_type(typ) for typ in data.get("types") or []]
    return {
        "active_type_id": data.get("active_type_id") or "",
        "threshold": float(data.get("threshold") or 0.58),
        "types": types,
        "llm": llm_status(),
    }


def library_for_match() -> List[dict]:
    data = ensure_builtin_types()
    cleaned = []
    for typ in data.get("types") or []:
        feats = []
        for ref in typ.get("refs") or []:
            feat = ref.get("features")
            if not feat:
                continue
            packed = dict(feat)
            packed["name"] = ref.get("name") or packed.get("name") or ""
            feats.append(packed)
        analyses = []
        for ref in typ.get("refs") or []:
            analysis = ref.get("analysis") or {}
            if not analysis:
                continue
            analyses.append(
                {
                    "name": ref.get("name") or "",
                    "has_subject": analysis.get("has_subject"),
                    "subject_x": analysis.get("subject_x"),
                    "subject_y": analysis.get("subject_y"),
                    "subject_box": analysis.get("subject_box"),
                    "subject_position": analysis.get("subject_position"),
                    "repeating": analysis.get("repeating"),
                    "qa_kinds": analysis.get("qa_kinds") or [],
                }
            )
        cleaned.append(
            {
                "id": typ["id"],
                "name": typ.get("name") or "",
                "description": typ.get("description") or "",
                "threshold": _type_threshold(typ, data),
                "features": feats,
                "analyses": analyses,
            }
        )
    return cleaned


def add_type(name: str, folder: Optional[Path] = None, description: str = "") -> dict:
    name = (name or "").strip()
    description = (description or "").strip()
    if not name:
        raise ValueError("请填写构图类型名称")
    ensure_builtin_types()
    files = _list_images(folder) if folder is not None else []
    if folder is not None and not files:
        raise ValueError("参考文件夹里没有图片")
    if not files and not description:
        raise ValueError("请填写构图定义，或提供参考图文件夹")
    mapped = _kind_from_name(name)
    with _lock:
        data = _read()
        existing_id = ""
        for typ in data["types"]:
            if typ.get("name") == name or typ.get("id") == mapped:
                existing_id = typ["id"]
                break
    if existing_id:
        if description:
            update_type(existing_id, description=description)
        if files:
            public, _added, _skipped = add_refs(existing_id, folder)
            return public
        return update_type(existing_id)
    type_id = mapped or uuid.uuid4().hex[:12]
    dest = REFS / type_id
    dest.mkdir(parents=True, exist_ok=True)
    unique_files, _skipped = _unique_images(files)
    refs = []
    for path, digest, orig in unique_files:
        ref = _ingest_ref(path, dest, mapped or type_id, analyze=False)
        if not ref:
            continue
        ref["hash"] = digest
        ref["name"] = orig
        refs.append(ref)
    if files and not refs:
        shutil.rmtree(dest, ignore_errors=True)
        raise ValueError("参考图都读不出来")
    typ = {
        "id": type_id,
        "name": name,
        "description": description,
        "threshold": 0.58,
        "refs": refs,
    }
    with _lock:
        data = _read()
        if any(item.get("id") == type_id for item in data["types"]):
            data["types"] = [item for item in data["types"] if item.get("id") != type_id]
        data["types"].append(typ)
        data["active_type_id"] = type_id
        _write(data)
    return _public_type(typ)


def update_type(
    type_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    threshold: Optional[float] = None,
) -> dict:
    with _lock:
        data = _read()
        typ = _find(data, type_id)
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("名称不能为空")
            typ["name"] = name
        if description is not None:
            typ["description"] = description.strip()
        if threshold is not None:
            typ["threshold"] = max(0.0, min(1.0, float(threshold)))
        _write(data)
        return _public_type(typ)


def delete_type(type_id: str) -> dict:
    with _lock:
        data = _read()
        data["types"] = [t for t in data["types"] if t.get("id") != type_id]
        if data.get("active_type_id") == type_id:
            data["active_type_id"] = data["types"][0]["id"] if data["types"] else ""
        _write(data)
    shutil.rmtree(REFS / type_id, ignore_errors=True)
    return public_store()


def add_refs(type_id: str, folder: Path):
    ensure_builtin_types()
    files = _list_images(folder)
    if not files:
        raise ValueError("参考文件夹里没有图片")
    dest = REFS / type_id
    dest.mkdir(parents=True, exist_ok=True)
    with _lock:
        data = _read()
        typ = _find(data, type_id)
        seen_hash, seen_name = _ref_keys(typ.get("refs") or [])
    unique_files, skipped = _unique_images(files, seen_hash, seen_name)
    added = []
    for path, digest, orig in unique_files:
        ref = _ingest_ref(path, dest, type_id, analyze=False)
        if not ref:
            continue
        ref["hash"] = digest
        ref["name"] = orig
        added.append(ref)
    if not added:
        if skipped:
            with _lock:
                return _public_type(_find(_read(), type_id)), [], skipped
        raise ValueError("追加的参考图都读不出来")
    added_ids = []
    with _lock:
        data = _read()
        typ = _find(data, type_id)
        seen_hash, seen_name = _ref_keys(typ.get("refs") or [])
        for ref in added:
            digest = ref.get("hash") or ""
            orig = _original_name(ref.get("name") or "")
            if (digest and digest in seen_hash) or (orig and orig in seen_name):
                skipped += 1
                if ref.get("path"):
                    Path(ref["path"]).unlink(missing_ok=True)
                continue
            if digest:
                seen_hash.add(digest)
            if orig:
                seen_name.add(orig)
            typ.setdefault("refs", []).append(ref)
            added_ids.append(ref["id"])
        _write(data)
        return _public_type(typ), added_ids, skipped


def delete_ref(type_id: str, ref_id: str) -> dict:
    with _lock:
        data = _read()
        typ = _find(data, type_id)
        kept = []
        removed = None
        for ref in typ.get("refs") or []:
            if ref.get("id") == ref_id:
                removed = ref
            else:
                kept.append(ref)
        if removed is None:
            raise KeyError("参考图不存在")
        typ["refs"] = kept
        _write(data)
        public = _public_type(typ)
    if removed and removed.get("path"):
        Path(removed["path"]).unlink(missing_ok=True)
    return public


def list_ref_ids(type_id: str) -> List[str]:
    ensure_builtin_types()
    data = load_store()
    typ = _find(data, type_id)
    return [ref.get("id") for ref in typ.get("refs") or [] if ref.get("id")]


def get_ref_record(type_id: str, ref_id: str) -> dict:
    data = load_store()
    typ = _find(data, type_id)
    for ref in typ.get("refs") or []:
        if ref.get("id") == ref_id:
            return dict(ref)
    raise KeyError("参考图不存在")


def write_ref_analysis(type_id: str, ref_id: str, analysis: dict) -> dict:
    with _lock:
        data = _read()
        typ = _find(data, type_id)
        found = False
        for ref in typ.get("refs") or []:
            if ref.get("id") == ref_id:
                ref["analysis"] = analysis
                found = True
                break
        if not found:
            raise KeyError("参考图不存在")
        _write(data)
        return _public_type(typ)


def reanalyze_refs(type_id: str) -> dict:
    from composition import analyze_reference

    ids = list_ref_ids(type_id)
    if not ids:
        raise ValueError("这个类型还没有参考图")
    public = None
    for ref_id in ids:
        public = analyze_and_save_ref(type_id, ref_id)
    return public or update_type(type_id)


def analyze_and_save_ref(type_id: str, ref_id: str) -> dict:
    from composition import analyze_reference

    ref = get_ref_record(type_id, ref_id)
    path = Path(ref.get("path") or "")
    if not path.is_file():
        analysis = {
            "why": "参考图文件已不存在。",
            "reason": "参考图文件已不存在。",
        }
    else:
        analysis = analyze_reference(str(path), type_id)
    return write_ref_analysis(type_id, ref_id, analysis)


def resolve_ref(type_id: str, ref_id: str) -> Path:
    data = load_store()
    typ = _find(data, type_id)
    for ref in typ.get("refs") or []:
        if ref.get("id") == ref_id:
            path = Path(ref["path"])
            if not path.is_file():
                raise FileNotFoundError("参考图文件已不存在")
            return path
    raise KeyError("参考图不存在")


def _read() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    REFS.mkdir(parents=True, exist_ok=True)
    if not STORE.exists():
        return _empty()
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty()
    data.setdefault("types", [])
    data.setdefault("active_type_id", "")
    data.setdefault("threshold", 0.58)
    fallback = float(data.get("threshold") or 0.58)
    for typ in data["types"]:
        if typ.get("threshold") is None:
            typ["threshold"] = fallback
    return data


def _write(data: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _find(data: dict, type_id: str) -> dict:
    for typ in data.get("types") or []:
        if typ.get("id") == type_id:
            return typ
    raise KeyError("构图类型不存在")


def _list_images(folder: Path) -> List[Path]:
    folder = folder.expanduser()
    if not folder.is_dir():
        raise ValueError(f"文件夹不存在：{folder}")
    files = [p for p in folder.iterdir() if p.is_file() and is_image_name(p.name) and not p.name.startswith(".")]
    files.sort(key=lambda p: p.name.lower())
    return files


def _file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _original_name(name: str) -> str:
    filename = Path(name or "").name
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    if re.search(r"_[0-9a-f]{6}$", stem):
        stem = stem[:-7]
    return f"{stem}{suffix}"


def _ref_keys(refs: List[dict]) -> Tuple[Set[str], Set[str]]:
    hashes: Set[str] = set()
    names: Set[str] = set()
    for ref in refs:
        digest = ref.get("hash") or ""
        if digest:
            hashes.add(digest)
        orig = _original_name(ref.get("name") or "")
        if orig:
            names.add(orig)
    return hashes, names


def _unique_images(
    files: List[Path],
    extra_hashes: Optional[Set[str]] = None,
    extra_names: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[Path, str, str]], int]:
    seen_hash = set(extra_hashes or [])
    seen_name = set(extra_names or [])
    unique: List[Tuple[Path, str, str]] = []
    skipped = 0
    for path in files:
        digest = _file_hash(path)
        orig = _original_name(path.name)
        if (digest and digest in seen_hash) or (orig and orig in seen_name):
            skipped += 1
            continue
        if digest:
            seen_hash.add(digest)
        if orig:
            seen_name.add(orig)
        unique.append((path, digest, orig))
    return unique, skipped


def _dedupe_type_refs(typ: dict) -> Tuple[bool, List[Path]]:
    refs = list(typ.get("refs") or [])
    if not refs:
        return False, []
    ranked = sorted(
        refs,
        key=lambda ref: (
            0 if (ref.get("analysis") or {}).get("why") else 1,
            0 if Path(ref.get("path") or "").is_file() else 1,
            0 if ref.get("name") == _original_name(ref.get("name") or "") else 1,
        ),
    )
    kept = []
    seen_hash: Set[str] = set()
    seen_name: Set[str] = set()
    removed: List[Path] = []
    changed = False
    for ref in ranked:
        path = Path(ref.get("path") or "")
        digest = ref.get("hash") or ""
        if not digest and path.is_file():
            digest = _file_hash(path)
            ref["hash"] = digest
            changed = True
        orig = _original_name(ref.get("name") or "")
        if orig and orig != ref.get("name"):
            ref["name"] = orig
            changed = True
        duplicate = (digest and digest in seen_hash) or (orig and orig in seen_name)
        if duplicate:
            if path.is_file():
                removed.append(path)
            changed = True
            continue
        if digest:
            seen_hash.add(digest)
        if orig:
            seen_name.add(orig)
        kept.append(ref)
    if [ref.get("id") for ref in kept] != [ref.get("id") for ref in refs]:
        changed = True
    typ["refs"] = kept
    return changed, removed


def _ingest_ref(src: Path, dest_dir: Path, kind_id: str = "", analyze: bool = False) -> Optional[dict]:
    from composition import analyze_reference

    feat = extract_features(str(src))
    if feat is None:
        return None
    ref_id = uuid.uuid4().hex[:10]
    target = dest_dir / f"{ref_id}_{src.name}"
    shutil.copy2(src, target)
    analysis = analyze_reference(str(target), kind_id) if analyze else {}
    return {
        "id": ref_id,
        "name": _original_name(src.name),
        "path": str(target.resolve()),
        "width": feat.width,
        "height": feat.height,
        "features": pack_features(src.name, feat),
        "analysis": analysis,
        "hash": _file_hash(src),
    }


def _kind_from_name(name: str) -> str:
    for type_id, label in BUILTIN_TYPES:
        if name == label or name == type_id:
            return type_id
    return ""


def _public_analysis(analysis: Optional[dict]) -> dict:
    if not analysis:
        return {}
    return {
        "has_subject": bool(analysis.get("has_subject")),
        "subject_x": analysis.get("subject_x"),
        "subject_y": analysis.get("subject_y"),
        "subject_box": analysis.get("subject_box"),
        "subject_position": analysis.get("subject_position") or "none",
        "subject_position_name": analysis.get("subject_position_name") or "",
        "repeating": bool(analysis.get("repeating")),
        "qa_kinds": analysis.get("qa_kinds") or [],
        "qa_names": analysis.get("qa_names") or [],
        "reason": analysis.get("reason") or "",
        "why": analysis.get("why") or "",
        "agrees": bool(analysis.get("agrees")),
    }


def _type_how(typ: dict) -> str:
    name = typ.get("name") or COMP_LABELS.get(typ.get("id") or "", "该构图")
    refs = typ.get("refs") or []
    analyses = [ref.get("analysis") for ref in refs if ref.get("analysis")]
    if not refs:
        return (
            f"还没有参考图。读入后会对每张提问：有没有主体、主体在水平方向哪、有没有重复主体，"
            f"并说明为什么算「{name}」。"
        )
    if not analyses:
        return f"已有 {len(refs)} 张参考图，但还没有答题分析。点「重新分析参考图」后会列出每张的依据。"
    agrees = sum(1 for row in analyses if row.get("agrees"))
    pos_counts: dict = {}
    xs = []
    for row in analyses:
        label = row.get("subject_position_name") or "位置不明"
        pos_counts[label] = pos_counts.get(label, 0) + 1
        if row.get("subject_x") is not None:
            xs.append(float(row["subject_x"]))
    pos_txt = "、".join(f"{label} {count} 张" for label, count in pos_counts.items())
    x_txt = f"主体水平位置约 {min(xs):.2f}–{max(xs):.2f}。" if xs else ""
    pending = len(refs) - len(analyses)
    pending_txt = f"另有 {pending} 张尚未答题分析。" if pending else ""
    return (
        f"读入时对每张参考图答题：有没有主体、主体在水平方向哪（只看 x 轴）、有没有重复主体。"
        f"已分析 {len(analyses)} 张，其中 {agrees} 张答题也认为是「{name}」。{pos_txt}。{x_txt}{pending_txt}"
    )


def _public_type(typ: dict) -> dict:
    refs = []
    for ref in typ.get("refs") or []:
        refs.append(
            {
                "id": ref.get("id"),
                "name": ref.get("name"),
                "width": ref.get("width") or 0,
                "height": ref.get("height") or 0,
                "analysis": _public_analysis(ref.get("analysis")),
            }
        )
    return {
        "id": typ.get("id"),
        "name": typ.get("name"),
        "description": typ.get("description") or "",
        "threshold": _type_threshold(typ),
        "refs": refs,
        "ref_count": len(refs),
        "how": _type_how(typ),
    }


def _type_threshold(typ: dict, data: Optional[dict] = None) -> float:
    if typ.get("threshold") is not None:
        return max(0.0, min(1.0, float(typ["threshold"])))
    if data is not None and data.get("threshold") is not None:
        return max(0.0, min(1.0, float(data["threshold"])))
    return 0.58
