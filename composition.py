"""Classify photos by subject placement: center, thirds, or repeating."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from image_io import load_preview_bgr

COMP_LABELS = {
    "center": "中心构图",
    "thirds": "三分构图",
    "repeat": "重复构图",
    "none": "未归类",
}
JUDGED_KINDS = ("center", "thirds", "repeat")
JUDGED_NAMES = {COMP_LABELS[k] for k in JUDGED_KINDS}
SIDECAR_COMP_KEYS = (
    "构图",
    "构图方式",
    "composition",
    "compositions",
    "composition_name",
    "composition_names",
    "composition_reason",
    "type_id",
    "type_name",
    "best_type_id",
    "best_type_name",
    "assigned",
    "scores",
    "match",
    "similarity",
    "中心构图",
    "三分构图",
    "重复构图",
)
POSITION_LABELS = {
    "center": "水平居中",
    "thirds": "三分线",
    "other": "其他位置",
    "none": "无主体",
}

ENERGY_GRID = 12
EDGE_GRID = 8
MAX_PEAKS = 5


def _num(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


@dataclass
class CompFeatures:
    width: int
    height: int
    energy: np.ndarray
    edges: np.ndarray
    stats: np.ndarray
    peaks: List[Tuple[float, float, float]]

    def as_public(self) -> dict:
        cx, cy, spread, center, border = (float(x) for x in self.stats[:5])
        return {
            "width": int(self.width),
            "height": int(self.height),
            "centroid": [_num(cx), _num(cy)],
            "spread": _num(spread),
            "center_mass": _num(center),
            "border_mass": _num(border),
            "peaks": [[_num(x), _num(y), _num(w)] for x, y, w in self.peaks],
        }


@dataclass
class TypeMatch:
    type_id: str
    name: str
    score: float
    ref_name: str
    flipped: bool = False
    visual: float = 0.0
    llm: Optional[float] = None
    reason: str = ""


@dataclass
class CompResult:
    width: int
    height: int
    matches: List[TypeMatch] = field(default_factory=list)
    notes: str = ""

    def score_for(self, type_id: str) -> float:
        for row in self.matches:
            if row.type_id == type_id:
                return row.score
        return 0.0

    def as_public(self) -> dict:
        best = self.matches[0] if self.matches else None
        return {
            "width": int(self.width),
            "height": int(self.height),
            "similarity": _num(best.score, 4) if best else 0.0,
            "best_type_id": best.type_id if best else "",
            "best_type_name": best.name if best else "",
            "best_ref": best.ref_name if best else "",
            "scores": {
                row.type_id: {
                    "name": row.name,
                    "score": _num(row.score, 4),
                    "visual": _num(row.visual, 4),
                    "llm": None if row.llm is None else _num(row.llm, 4),
                    "reason": row.reason,
                    "ref_name": row.ref_name,
                    "flipped": row.flipped,
                }
                for row in self.matches
            },
            "notes": self.notes,
        }


def extract_features(path: str, max_side: int = 480) -> Optional[CompFeatures]:
    image, width, height = load_preview_bgr(path, max_side)
    if image is None:
        return None
    return features_from_bgr(image, width, height)


def features_from_bgr(image: np.ndarray, width: int, height: int) -> CompFeatures:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (float(mag.max()) + 1e-6)
    contrast = cv2.Laplacian(gray, cv2.CV_32F)
    contrast = np.abs(contrast)
    contrast = contrast / (float(contrast.max()) + 1e-6)
    energy = np.clip(0.72 * mag + 0.28 * contrast, 0, 1)
    energy = cv2.GaussianBlur(energy, (9, 9), 0)
    if float(energy.max()) > 0:
        energy = energy / float(energy.max())

    energy_grid = _grid_mean(energy, ENERGY_GRID)
    edges = cv2.Canny(gray, 60, 160)
    edge_grid = _grid_mean(edges.astype(np.float32) / 255.0, EDGE_GRID)
    stats = _spatial_stats(energy)
    peaks = _peaks(energy)
    return CompFeatures(
        width=width,
        height=height,
        energy=energy_grid,
        edges=edge_grid,
        stats=stats,
        peaks=peaks,
    )


def summarize_refs(feats: Sequence[CompFeatures]) -> dict:
    if not feats:
        return {"detail": "还没有可用的参考图。", "centroid": [0.5, 0.5], "center_mass": 0, "border_mass": 0, "peak_count": 0}
    cxs = [float(f.stats[0]) for f in feats]
    cys = [float(f.stats[1]) for f in feats]
    centers = [float(f.stats[3]) for f in feats]
    borders = [float(f.stats[4]) for f in feats]
    peaks = [len(f.peaks) for f in feats]
    cx, cy = float(np.mean(cxs)), float(np.mean(cys))
    center, border = float(np.mean(centers)), float(np.mean(borders))
    peak_count = int(round(float(np.mean(peaks)))) if peaks else 0
    detail = _describe(cx, cy, center, border, peak_count, len(feats))
    return {
        "detail": detail,
        "centroid": [_num(cx), _num(cy)],
        "center_mass": _num(center),
        "border_mass": _num(border),
        "peak_count": peak_count,
        "ref_count": len(feats),
    }


def match_features(query: CompFeatures, library: Sequence[dict]) -> CompResult:
    matches: List[TypeMatch] = []
    flipped = _flip_features(query)
    for typ in library:
        refs = typ.get("features") or []
        best = TypeMatch(typ["id"], typ.get("name") or "", 0.0, "", False)
        for ref in refs:
            score, used_flip = _best_ref_score(query, flipped, ref)
            if score > best.score:
                best = TypeMatch(typ["id"], typ.get("name") or "", score, ref.get("name") or "", used_flip)
        best.visual = best.score
        matches.append(best)
    matches.sort(key=lambda m: m.score, reverse=True)
    notes = ""
    if matches:
        top = matches[0]
        notes = f"最接近「{top.name}」，相似度 {top.score * 100:.1f}%"
        if top.ref_name:
            notes += f"，参考 {top.ref_name}"
    return CompResult(width=query.width, height=query.height, matches=matches, notes=notes)


def apply_language_scores(result: CompResult, path: str, library: Sequence[dict], feat: CompFeatures) -> CompResult:
    from composition_llm import judge_type, llm_call

    by_id = {row.type_id: row for row in result.matches}
    for typ in library:
        row = by_id.get(typ.get("id") or "")
        if row is None:
            continue
        judged = llm_call(judge_type, path, typ, feat, row.visual)
        if not judged.get("used") or judged.get("score") is None:
            if judged.get("reason"):
                row.reason = judged["reason"]
            continue
        row.llm = float(judged["score"])
        row.reason = judged.get("reason") or ""
        row.score = float(0.4 * row.visual + 0.6 * row.llm)
    result.matches.sort(key=lambda m: m.score, reverse=True)
    if result.matches:
        top = result.matches[0]
        bits = [f"最接近「{top.name}」，综合 {top.score * 100:.1f}%（参考图 {top.visual * 100:.1f}%"]
        if top.llm is not None:
            bits.append(f"定义 {top.llm * 100:.1f}%")
        notes = "，".join(bits) + "）"
        if top.ref_name:
            notes += f"。最像参考图 {top.ref_name}"
        if top.reason:
            notes += f"。{top.reason}"
        result.notes = notes
    return result


def analyze_reference(path: str, kind_id: str = "") -> dict:
    from composition_llm import judge_subject, llm_call

    judged = llm_call(judge_subject, path)
    has_subject = bool(judged.get("has_subject"))
    position = judged.get("subject_position") or "none"
    repeating = bool(judged.get("repeating"))
    qa_kinds = _decide_kinds(has_subject, position, repeating)
    return {
        "has_subject": has_subject,
        "subject_x": judged.get("subject_x"),
        "subject_y": judged.get("subject_y"),
        "subject_box": judged.get("subject_box"),
        "subject_side": judged.get("subject_side") or "",
        "subject_position": position,
        "subject_position_name": POSITION_LABELS.get(position, "无主体"),
        "repeating": repeating,
        "qa_kinds": qa_kinds,
        "qa_names": [COMP_LABELS[k] for k in qa_kinds],
        "reason": judged.get("reason") or judged.get("error") or "",
        "why": explain_why_ref(kind_id, judged, qa_kinds),
        "agrees": bool(kind_id and kind_id in qa_kinds),
    }


def explain_why_ref(kind_id: str, judged: dict, qa_kinds: Sequence[str]) -> str:
    label = COMP_LABELS.get(kind_id) or kind_id or "该构图"
    x = judged.get("subject_x")
    x_txt = f"x={float(x):.2f}" if x is not None else "水平位置不明"
    pos_name = POSITION_LABELS.get(judged.get("subject_position") or "none", "无主体")
    parts = []
    if judged.get("error"):
        return f"参考图未能完成答题：{judged['error']}"
    if not judged.get("has_subject"):
        parts.append("答题认为没有明确主体。")
    else:
        repeat_txt = "并看到多个形态类似的重复主体。" if judged.get("repeating") else "未见重复主体。"
        side = judged.get("subject_side") or ""
        side_txt = {"left": "主体在画面左侧，", "right": "主体在画面右侧，", "center": "主体在画面中部，"}.get(side, "")
        parts.append(f"按主体焦点，{side_txt}水平位置在{pos_name}（{x_txt}），{repeat_txt}")
    if kind_id in qa_kinds:
        parts.append(f"因此这张参考图符合「{label}」。")
    else:
        qa = "、".join(COMP_LABELS[k] for k in qa_kinds) or "未归入当前构图类型"
        parts.append(f"你把它作为「{label}」参考图；答题更倾向{qa}。待判图仍会对照它的主体位置。")
    if judged.get("reason"):
        parts.append(str(judged["reason"]))
    return "".join(parts)


def explain_why_scan(judged: dict, qa_kinds: Sequence[str], ref_notes: Sequence[str], library) -> str:
    if judged.get("error"):
        return f"未能完成答题：{judged['error']}"
    parts = []
    x = judged.get("subject_x")
    x_txt = f"x={float(x):.2f}" if x is not None else "水平位置不明"
    pos_name = POSITION_LABELS.get(judged.get("subject_position") or "none", "无主体")
    if not judged.get("has_subject"):
        parts.append("答题认为没有明确主体。")
    else:
        repeat_txt = "并看到多个形态类似的重复主体。" if judged.get("repeating") else "未见重复主体。"
        side = judged.get("subject_side") or ""
        side_txt = {"left": "主体在画面左侧，", "right": "主体在画面右侧，", "center": "主体在画面中部，"}.get(side, "")
        parts.append(f"和分析参考图同一套方法：{side_txt}按主体焦点，水平位置在{pos_name}（{x_txt}），{repeat_txt}")
    if "center" in qa_kinds:
        parts.append("因此判为中心构图。")
    else:
        parts.append("因此不判为中心构图。")
    if ref_notes:
        parts.append("".join(ref_notes))
    elif library:
        analyzed = sum(len(typ.get("analyses") or []) for typ in library)
        if analyzed == 0:
            parts.append("参考图还没有答题分析，本次只按主体位置判定。请先点「重新分析参考图」。")
    if judged.get("retries"):
        parts.append(f"模型初答不合格，已重问 {judged['retries']} 次。")
    if judged.get("reason"):
        parts.append(str(judged["reason"]))
    return "".join(parts)


def analyze_composition(path: str, library: Optional[Sequence[dict]] = None) -> dict:
    from PIL import Image

    from composition_llm import judge_subject, llm_call

    width = height = 0
    try:
        with Image.open(path) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001
        return _comp_public(0, 0, False, "none", False, [], "无法读取图片")
    judged = llm_call(judge_subject, path)
    has_subject = bool(judged.get("has_subject"))
    position = judged.get("subject_position") or "none"
    repeating = bool(judged.get("repeating"))
    labels = dict(COMP_LABELS)
    for typ in library or []:
        if typ.get("id"):
            labels[typ["id"]] = typ.get("name") or typ["id"]
    allowed = {typ.get("id") for typ in (library or []) if typ.get("id")}
    qa_kinds = _decide_kinds(has_subject, position, repeating)
    if allowed:
        qa_kinds = [k for k in qa_kinds if k in allowed]
    reason = judged.get("reason") or ""
    if judged.get("error"):
        reason = judged["error"]
    ref_notes: List[str] = []
    scores: dict = {}
    ref_kinds: List[str] = []
    if library:
        ref_kinds, ref_notes, scores = _kinds_from_refs(judged, library)
        if allowed:
            ref_kinds = [k for k in ref_kinds if k in allowed]
            scores = {key: val for key, val in scores.items() if key in allowed}
    kinds = list(dict.fromkeys([*qa_kinds, *ref_kinds]))
    why = explain_why_scan(judged, qa_kinds, ref_notes, library)
    if ref_notes:
        reason = f"{reason} {' '.join(ref_notes)}".strip()
    result = _comp_public(
        width,
        height,
        has_subject,
        position,
        repeating,
        kinds,
        reason,
        judged.get("subject_x"),
        judged.get("subject_y"),
        ref_notes,
        library,
        judged.get("subject_box"),
    )
    result["qa_kinds"] = qa_kinds
    result["qa_names"] = [labels[k] for k in qa_kinds if k in labels]
    result["subject_side"] = judged.get("subject_side") or ""
    result["retries"] = judged.get("retries") or 0
    result["lazy_corrected"] = bool(judged.get("lazy_corrected"))
    result["why"] = why
    result["scores"] = scores
    result["library_names"] = [typ.get("name") for typ in (library or []) if typ.get("name")]
    if not judged.get("error"):
        try:
            result["sidecar"] = write_sidecar(path, result)
        except OSError as exc:
            extra = f"JSON 未写入：{exc}"
            result["reason"] = f"{result.get('reason') or ''}（{extra}）".strip("（）")
    return result


def _decide_kinds(has_subject: bool, position: str, repeating: bool) -> List[str]:
    kinds: List[str] = []
    if repeating:
        kinds.append("repeat")
    if has_subject and position == "center":
        kinds.append("center")
    elif has_subject and position == "thirds":
        kinds.append("thirds")
    return kinds


def _kinds_from_refs(judged: dict, library: Sequence[dict]) -> Tuple[List[str], List[str], dict]:
    kinds = []
    notes = []
    scores = {}
    for typ in library:
        type_id = typ.get("id") or ""
        refs = typ.get("analyses") or []
        best_name = ""
        best_score = 0.0
        best_x = None
        for ref in refs:
            score = _qa_layout_score(judged, ref)
            if score > best_score:
                best_score = score
                best_name = ref.get("name") or ""
                best_x = ref.get("subject_x")
        x_txt = f"，参考图主体 x={float(best_x):.2f}" if best_x is not None else ""
        scores[type_id] = {
            "name": typ.get("name") or "",
            "score": _num(best_score, 4),
            "visual": _num(best_score, 4),
            "llm": None,
            "reason": f"主体位置对照{x_txt}" if best_name else "参考图尚未答题分析",
            "ref_name": best_name,
        }
        if best_name:
            notes.append(
                f"对照参考图：最像「{typ.get('name')}」的 {best_name}（主体位置相似度 {best_score * 100:.0f}%{x_txt}）。"
            )
        elif typ.get("features"):
            notes.append(f"「{typ.get('name')}」的参考图还没有答题分析，无法用同一套方法对照。")
        if type_id in JUDGED_KINDS:
            continue
        threshold = 0.58 if typ.get("threshold") is None else float(typ["threshold"])
        if best_score + 1e-12 >= threshold:
            kinds.append(type_id)
    return kinds, notes, scores


def _qa_layout_score(query: dict, ref: dict) -> float:
    if not query.get("has_subject") or not ref.get("has_subject"):
        return 0.15 if (not query.get("has_subject") and not ref.get("has_subject")) else 0.0
    qx, rx = query.get("subject_x"), ref.get("subject_x")
    if qx is None or rx is None:
        return 0.12
    s_x = 1.0 - min(1.0, abs(float(qx) - float(rx)) / 0.28)
    qy, ry = query.get("subject_y"), ref.get("subject_y")
    s_y = 0.5
    if qy is not None and ry is not None:
        s_y = 1.0 - min(1.0, abs(float(qy) - float(ry)) / 0.40)
    s_pos = 1.0 if query.get("subject_position") == ref.get("subject_position") else 0.4
    iou = _box_iou(query.get("subject_box"), ref.get("subject_box"))
    if iou is None:
        return float(np.clip(0.62 * s_x + 0.13 * s_y + 0.25 * s_pos, 0, 1))
    return float(np.clip(0.48 * s_x + 0.10 * s_y + 0.18 * s_pos + 0.24 * iou, 0, 1))


def _box_iou(a, b):
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return None
    if len(a) < 4 or len(b) < 4:
        return None
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    except (TypeError, ValueError):
        return None
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _comp_public(
    width: int,
    height: int,
    has_subject: bool,
    position: str,
    repeating: bool,
    kinds: Sequence[str],
    reason: str,
    subject_x=None,
    subject_y=None,
    ref_notes=None,
    library=None,
    subject_box=None,
) -> dict:
    labels = dict(COMP_LABELS)
    for typ in library or []:
        type_id = typ.get("id")
        if type_id:
            labels[type_id] = typ.get("name") or type_id
    kinds = [k for k in kinds if k in labels and k != "none"]
    names = [labels[k] for k in kinds]
    primary = kinds[0] if kinds else "none"
    return {
        "width": int(width or 0),
        "height": int(height or 0),
        "composition": primary,
        "composition_name": "、".join(names) if names else "未归类",
        "compositions": kinds,
        "composition_names": names,
        "has_subject": bool(has_subject),
        "subject_position": position if position in POSITION_LABELS else "none",
        "subject_position_name": POSITION_LABELS.get(position, "无主体"),
        "subject_x": subject_x,
        "subject_y": subject_y,
        "subject_box": subject_box,
        "repeating": bool(repeating),
        "reason": reason or "",
        "ref_notes": list(ref_notes or []),
    }


def sidecar_path(image_path: str) -> Path:
    return Path(image_path).with_suffix(".json")


def write_sidecar(image_path: str, result: dict) -> str:
    path = Path(image_path)
    sidecar = sidecar_path(str(path))
    existing = _read_sidecar(sidecar)
    kept = _kept_composition_names(existing)
    merged = dict(existing)
    for key in SIDECAR_COMP_KEYS:
        merged.pop(key, None)
    scanned = set(result.get("library_names") or []) | JUDGED_NAMES
    kept = [name for name in kept if name not in scanned]
    names = list(dict.fromkeys([*kept, *(result.get("composition_names") or [])]))
    merged["file"] = existing.get("file") or path.name
    merged["构图"] = names
    merged["has_subject"] = bool(result.get("has_subject"))
    merged["subject_position"] = result.get("subject_position") or "none"
    merged["subject_position_name"] = result.get("subject_position_name") or ""
    merged["subject_x"] = result.get("subject_x")
    merged["subject_y"] = result.get("subject_y")
    merged["subject_box"] = result.get("subject_box")
    merged["repeating"] = bool(result.get("repeating"))
    merged["composition_reason"] = result.get("why") or result.get("reason") or ""
    for kind in JUDGED_KINDS:
        zh = COMP_LABELS[kind]
        if kind in (result.get("compositions") or []):
            merged[zh] = True
    sidecar.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(sidecar.resolve())


def _read_sidecar(sidecar) -> dict:
    path = Path(sidecar)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _kept_composition_names(data: dict) -> List[str]:
    names: List[str] = []
    for key in ("构图", "构图方式", "composition_names", "assigned"):
        names.extend(_as_name_list(data.get(key)))
    for key in ("composition_name", "type_name", "best_type_name"):
        names.extend(_as_name_list(data.get(key)))
    for key in ("composition", "type_id", "best_type_id"):
        cid = data.get(key)
        if cid in COMP_LABELS:
            names.append(COMP_LABELS[cid])
    for zh in JUDGED_NAMES:
        value = data.get(zh)
        if value is True or value in (1, "1", "true", "True", "是"):
            names.append(zh)
        elif isinstance(value, dict):
            names.append(zh)
    kept = []
    for name in names:
        name = str(name).strip()
        if not name or name in ("未归类", "none"):
            continue
        if name in JUDGED_NAMES:
            continue
        if name not in kept:
            kept.append(name)
    return kept


def _as_name_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.extend(_split_names(item))
            elif isinstance(item, dict):
                label = item.get("name") or item.get("composition_name") or ""
                if label:
                    out.append(str(label))
        return out
    if isinstance(value, str):
        return _split_names(value)
    return []


def _split_names(text: str) -> List[str]:
    cleaned = str(text).replace("、", " ").replace("，", " ").replace(",", " ").replace("/", " ")
    return [part.strip() for part in cleaned.split() if part.strip()]


def match_path(path: str, library: Sequence[dict]) -> CompResult:
    feat = extract_features(path)
    if feat is None:
        return CompResult(width=0, height=0, notes="无法读取图片")
    if not library:
        return CompResult(width=feat.width, height=feat.height, notes="还没有构图类型")
    result = match_features(feat, library)
    if any((typ.get("description") or "").strip() for typ in library):
        result = apply_language_scores(result, path, library, feat)
    return result


def pack_features(name: str, feat: CompFeatures) -> dict:
    return {
        "name": name,
        "energy": [float(x) for x in feat.energy.ravel()],
        "edges": [float(x) for x in feat.edges.ravel()],
        "stats": [float(x) for x in feat.stats.ravel()],
        "peaks": [[float(a) for a in p] for p in feat.peaks],
        **feat.as_public(),
    }


def _best_ref_score(query: CompFeatures, flipped: CompFeatures, ref: dict) -> Tuple[float, bool]:
    s0 = _score(query, ref)
    s1 = _score(flipped, ref)
    if s1 > s0:
        return s1, True
    return s0, False


def _score(query: CompFeatures, ref: dict) -> float:
    energy = np.asarray(ref.get("energy") or [], dtype=np.float32)
    edges = np.asarray(ref.get("edges") or [], dtype=np.float32)
    stats = np.asarray(ref.get("stats") or [], dtype=np.float32)
    if energy.size != query.energy.size or edges.size != query.edges.size:
        return 0.0
    s_e = _cosine(query.energy.ravel(), energy)
    s_g = _cosine(query.edges.ravel(), edges)
    if stats.size == query.stats.size and stats.size:
        dist = float(np.linalg.norm(query.stats - stats))
        s_s = 1.0 - min(1.0, dist / (np.sqrt(stats.size) * 0.55 + 1e-6))
    else:
        s_s = 0.0
    s_p = _peak_score(query.peaks, ref.get("peaks") or [])
    return float(np.clip(0.42 * s_e + 0.28 * s_g + 0.15 * s_s + 0.15 * s_p, 0, 1))


def _flip_features(feat: CompFeatures) -> CompFeatures:
    energy = feat.energy.reshape(ENERGY_GRID, ENERGY_GRID)[:, ::-1].copy()
    edges = feat.edges.reshape(EDGE_GRID, EDGE_GRID)[:, ::-1].copy()
    stats = feat.stats.copy()
    stats[0] = 1.0 - stats[0]
    peaks = [(1.0 - x, y, w) for x, y, w in feat.peaks]
    return CompFeatures(feat.width, feat.height, energy, edges, stats, peaks)


def _grid_mean(image: np.ndarray, n: int) -> np.ndarray:
    h, w = image.shape[:2]
    out = np.zeros((n, n), dtype=np.float32)
    for y in range(n):
        y1, y2 = int(y * h / n), int((y + 1) * h / n)
        for x in range(n):
            x1, x2 = int(x * w / n), int((x + 1) * w / n)
            cell = image[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
            out[y, x] = float(cell.mean()) if cell.size else 0.0
    total = float(out.sum())
    if total > 0:
        out /= total
    return out


def _spatial_stats(energy: np.ndarray) -> np.ndarray:
    h, w = energy.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    mass = float(energy.sum()) + 1e-6
    cx = float((energy * xs).sum() / mass / max(w - 1, 1))
    cy = float((energy * ys).sum() / mass / max(h - 1, 1))
    sx = float(np.sqrt(((xs / max(w - 1, 1) - cx) ** 2 * energy).sum() / mass))
    sy = float(np.sqrt(((ys / max(h - 1, 1) - cy) ** 2 * energy).sum() / mass))
    spread = float(np.sqrt(sx * sx + sy * sy))
    y1, y2 = int(h * 0.28), int(h * 0.72)
    x1, x2 = int(w * 0.28), int(w * 0.72)
    center = float(energy[y1:y2, x1:x2].sum() / mass)
    band = max(2, int(min(h, w) * 0.12))
    border = energy[:band, :].sum() + energy[-band:, :].sum() + energy[:, :band].sum() + energy[:, -band:].sum()
    border = float(border / mass)
    thirds = []
    for iy in range(3):
        for ix in range(3):
            yy1, yy2 = int(iy * h / 3), int((iy + 1) * h / 3)
            xx1, xx2 = int(ix * w / 3), int((ix + 1) * w / 3)
            thirds.append(float(energy[yy1:yy2, xx1:xx2].sum() / mass))
    d1 = float(np.mean(np.diag(cv2.resize(energy, (24, 24)))))
    d2 = float(np.mean(np.diag(np.fliplr(cv2.resize(energy, (24, 24))))))
    return np.array([cx, cy, spread, center, border, d1, d2, *thirds], dtype=np.float32)


def _peaks(energy: np.ndarray) -> List[Tuple[float, float, float]]:
    blur = cv2.GaussianBlur(energy, (11, 11), 0)
    h, w = blur.shape[:2]
    kernel = np.ones((9, 9), np.uint8)
    mx = cv2.dilate(blur, kernel)
    mask = (blur >= mx) & (blur > float(blur.max()) * 0.22)
    ys, xs = np.where(mask)
    pts = []
    for y, x in zip(ys, xs):
        pts.append((x / max(w - 1, 1), y / max(h - 1, 1), float(blur[y, x])))
    pts.sort(key=lambda p: p[2], reverse=True)
    kept: List[Tuple[float, float, float]] = []
    for x, y, weight in pts:
        if any(abs(x - ox) < 0.14 and abs(y - oy) < 0.14 for ox, oy, _ in kept):
            continue
        kept.append((x, y, weight))
        if len(kept) >= MAX_PEAKS:
            break
    return kept


def _peak_score(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    if not a or not b:
        return 0.35 if not a and not b else 0.12
    pa = np.array([[p[0], p[1]] for p in a], dtype=np.float32)
    pb = np.array([[p[0], p[1]] for p in b], dtype=np.float32)
    d_ab = _chamfer(pa, pb)
    d_ba = _chamfer(pb, pa)
    dist = 0.5 * (d_ab + d_ba)
    count = 1.0 - min(1.0, abs(len(a) - len(b)) / 4.0)
    return float(np.clip((1.0 - min(1.0, dist / 0.45)) * 0.75 + 0.25 * count, 0, 1))


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d = 0.0
    for p in a:
        d += float(np.min(np.linalg.norm(b - p, axis=1)))
    return d / max(len(a), 1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0, 1))


def _describe(cx: float, cy: float, center: float, border: float, peak_count: int, n: int) -> str:
    x = "左侧" if cx < 0.38 else "右侧" if cx > 0.62 else "中轴"
    y = "上方" if cy < 0.38 else "下方" if cy > 0.62 else "中部"
    bits = [f"{n} 张参考图的主体质量中心多在画面{x}{y}"]
    if border >= 0.34:
        bits.append("四周能量偏高，接近框架/包围")
    if center >= 0.42:
        bits.append("中心区域更重，接近居中")
    elif peak_count >= 3:
        bits.append(f"常见 {peak_count} 个显著点，利于三角或多点构图")
    elif peak_count == 2:
        bits.append("常见两点对位，利于对称或对角")
    return "；".join(bits) + "。"
