"""Score an image against a composition type description via a local or remote LLM."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

from composition import CompFeatures

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".data" / "comp_llm_cache"
_lock = threading.Lock()
_backend_cache = {"at": 0.0, "info": None}

VISION_MARKERS = (
    "llava",
    "vision",
    "vl",
    "vlm",
    "minicpm",
    "moondream",
    "bakllava",
    "gemma3",
    "qwen2.5vl",
    "qwen2-vl",
    "qwen3-vl",
)


def llm_status() -> dict:
    info = _detect()
    if not info:
        return {
            "ready": False,
            "backend": "",
            "model": "",
            "mode": "",
            "detail": "未检测到可用模型。本机可开 Ollama，或设置 OPENAI_API_KEY / COMPOSITION_LLM_BASE。",
        }
    return {
        "ready": True,
        "backend": info["backend"],
        "model": info["model"],
        "mode": info["mode"],
        "detail": (
            f"将用 {info['backend']} / {info['model']} 逐张量主体左右位置；坐标偷懒或自相矛盾会重问，不对照参考图蒙混。"
            if info["mode"] == "vision"
            else f"本机 {info['backend']} / {info['model']} 看不见图片，构图分析需要视觉模型。"
        ),
    }


SUBJECT_PROMPT = (
    "请看附件这张照片，只根据当前画面认真估量，禁止套模板。"
    "主体是第一眼看到、面积最大的那个对象，不是它内部的小细节。"
    "动物正面像：主体是整只动物，必须包住头和角，不要只框身体伸出画面的一侧。"
    "车辆：主体是整辆车，不要只框车里的人、车灯或旁边的车。"
    "人像：包住可见的头到身体；构图位置看头/脸的中心。"
    "路牌、标志牌：只框牌子本身，不要把树丛、天空、路面包进框。"
    "先在脑子里把画面左右分成三等份，再量这些数："
    "left_gap=主体焦点到左边缘的距离，right_gap=主体焦点到右边缘的距离，两者相加应接近 1。"
    "focus 是主体最醒目部分的中心。人/动物用头或脸，车辆用车身，路牌用牌子中心。"
    "side：主体大部分在左/中/右。nearer_edge：焦点离哪条竖边更近，差不多写 equal。"
    "主体很大、形状对称、很醒目，都不等于在画面正中，也不等于 focus_x=0.5。"
    "没有亲眼量过，禁止写 0.5；左右明显不等时更禁止写 0.5。"
    "坐标必须估到两位小数，不要 0.3/0.5/0.7 这种整十模板。"
    "框必须紧贴主体外轮廓。"
    "坐标原点在左上角：x=0 最左，x=1 最右；y=0 最上，y=1 最下。"
    "不要自己判断居中还是三分。"
    "再判断是否有多个形态类似、重复出现的主体。"
    "只返回 JSON，不要其他文字："
    '{"has_subject":true或false,'
    '"side":"left或center或right",'
    '"nearer_edge":"left或right或equal",'
    '"left_gap":0到1,'
    '"right_gap":0到1,'
    '"focus_x":0到1,'
    '"focus_y":0到1,'
    '"x1":0到1,'
    '"y1":0到1,'
    '"x2":0到1,'
    '"y2":0到1,'
    '"repeating":true或false,'
    '"reason":"主体是什么，左右空隙哪边更大"}。'
)


def judge_subject(path: str) -> dict:
    info = _detect()
    if not info:
        return {
            "has_subject": False,
            "subject_position": "none",
            "repeating": False,
            "reason": "",
            "error": "没有可用的语言模型",
        }
    if info.get("mode") != "vision":
        return {
            "has_subject": False,
            "subject_position": "none",
            "repeating": False,
            "reason": "",
            "error": "当前模型看不见图片，请安装视觉模型后再分析",
        }
    key = _subject_cache_key(path, info)
    cached = _read_cache(key)
    if cached is not None:
        return cached
    try:
        parsed = _judge_subject_careful(info, path)
    except Exception as exc:  # noqa: BLE001
        parsed = {
            "has_subject": False,
            "subject_position": "none",
            "repeating": False,
            "reason": "",
            "error": f"模型调用失败：{exc}",
        }
    if not parsed.get("error"):
        _write_cache(key, parsed)
    return parsed


def _judge_subject_careful(info: dict, path: str) -> dict:
    raw = _ask_vision(info, SUBJECT_PROMPT, path)
    data = _load_json_obj(raw)
    if data is None:
        return {
            "has_subject": False,
            "subject_position": "none",
            "repeating": False,
            "reason": "",
            "error": "模型没有返回可解析结果",
        }
    flags = _lazy_flags(data)
    retries = 0
    while flags and retries < 2:
        retries += 1
        raw = _ask_vision(info, _retry_prompt(data, flags), path, temperature=0.2)
        nxt = _load_json_obj(raw)
        if nxt is None:
            continue
        data = nxt
        flags = _lazy_flags(data)
    parsed = _parse_subject_data(data)
    parsed["retries"] = retries
    parsed["lazy_flags"] = flags
    parsed["lazy_corrected"] = bool(retries or flags or parsed.get("corrected"))
    if flags:
        extra = "；".join(flags)
        parsed["reason"] = f"{parsed.get('reason') or ''}（初答坐标不合格：{extra}，已按左右关系纠正）".strip()
    elif retries:
        parsed["reason"] = f"{parsed.get('reason') or ''}（已重问并按画面重新估量）".strip()
    return parsed


def _parse_subject(raw: str) -> dict:
    data = _load_json_obj(raw)
    if data is None:
        return {
            "has_subject": False,
            "subject_position": "none",
            "repeating": False,
            "reason": "",
            "error": "模型没有返回可解析结果",
        }
    return _parse_subject_data(data)


def _parse_subject_data(data: dict) -> dict:
    has_subject = _as_bool(data.get("has_subject"))
    reason = str(data.get("reason") or "").strip()
    side = _resolve_side(data, reason)
    box = _subject_box(data)
    focus_x = _as_ratio(data.get("focus_x"))
    focus_y = _as_ratio(data.get("focus_y"))
    gap_x = _x_from_gaps(data)
    if gap_x is not None and (focus_x is None or _is_half(focus_x) or abs(gap_x - focus_x) > 0.08):
        focus_x = gap_x
    box_x = (box[0] + box[2]) / 2 if box else _as_ratio(data.get("subject_x"))
    box_y = (box[1] + box[3]) / 2 if box else _as_ratio(data.get("subject_y"))
    subject_x = focus_x if focus_x is not None else box_x
    subject_y = focus_y if focus_y is not None else box_y
    before = subject_x
    subject_x, subject_y = _apply_side(side, subject_x, subject_y, box_x, box_y)
    return {
        "has_subject": has_subject,
        "subject_x": subject_x,
        "subject_y": subject_y,
        "subject_box": box,
        "subject_side": side or "",
        "subject_position": _position_from_xy(has_subject, subject_x, subject_y),
        "repeating": _as_bool(data.get("repeating")),
        "reason": reason,
        "corrected": before is not None and subject_x is not None and abs(float(before) - float(subject_x)) > 0.04,
        "nearer_edge": _parse_nearer(data.get("nearer_edge")),
    }


def _load_json_obj(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _is_half(value) -> bool:
    return value is not None and abs(float(value) - 0.5) <= 0.012


def _is_tenth_template(value) -> bool:
    if value is None:
        return True
    stepped = round(float(value) * 10) / 10
    return abs(float(value) - stepped) <= 0.012


def _parse_nearer(value) -> str:
    text = str(value or "").strip().lower()
    if text in {"left", "l", "左", "左侧", "左边"}:
        return "left"
    if text in {"right", "r", "右", "右侧", "右边"}:
        return "right"
    if text in {"equal", "same", "center", "centre", "middle", "均", "一样", "差不多"}:
        return "equal"
    return ""


def _x_from_gaps(data: dict):
    left = _as_ratio(data.get("left_gap"))
    right = _as_ratio(data.get("right_gap"))
    if left is None or right is None:
        return None
    total = left + right
    if total < 0.7 or total > 1.3:
        return None
    return max(0.0, min(1.0, left / total))


def _resolve_side(data: dict, reason: str) -> str:
    side = _parse_side(data.get("side"))
    nearer = _parse_nearer(data.get("nearer_edge"))
    from_reason = _side_from_reason(reason)
    gap_x = _x_from_gaps(data)
    gap_side = ""
    if gap_x is not None:
        if gap_x <= 0.40:
            gap_side = "left"
        elif gap_x >= 0.60:
            gap_side = "right"
        else:
            gap_side = "center"
    if side in {"left", "right"}:
        return side
    if from_reason in {"left", "right"}:
        return from_reason
    if side == "center":
        if nearer in {"left", "right"} and gap_side == nearer:
            return nearer
        return "center"
    if nearer in {"left", "right"}:
        return nearer
    if gap_side in {"left", "right"}:
        return gap_side
    return side or gap_side or from_reason


def _lazy_flags(data: dict) -> list:
    if not data or not _as_bool(data.get("has_subject")):
        return []
    reason = str(data.get("reason") or "")
    side = _parse_side(data.get("side")) or _side_from_reason(reason)
    nearer = _parse_nearer(data.get("nearer_edge"))
    focus_x = _as_ratio(data.get("focus_x"))
    focus_y = _as_ratio(data.get("focus_y"))
    box = _subject_box(data)
    gap_x = _x_from_gaps(data)
    flags = []
    if not side:
        flags.append("没回答主体在左中右哪侧")
    if _is_half(focus_x) and _is_half(focus_y) and side in {"left", "right"}:
        flags.append("焦点写成0.50，但已说明主体偏一侧")
    if _is_half(focus_x) and nearer in {"left", "right"}:
        flags.append("焦点写成0.50，但离一边更近")
    if side == "left" and focus_x is not None and focus_x >= 0.45:
        flags.append("说在左侧，focus_x却靠中或靠右")
    if side == "right" and focus_x is not None and focus_x <= 0.55:
        flags.append("说在右侧，focus_x却靠中或靠左")
    if side == "center" and nearer in {"left", "right"}:
        flags.append("side写center，但nearer_edge偏向一侧")
    if box and side == "left" and box[0] <= 0.22 and box[2] >= 0.75:
        flags.append("声称在左，包围盒却横跨整幅画面")
    if box and side == "right" and box[0] <= 0.25 and box[2] >= 0.78:
        flags.append("声称在右，包围盒却横跨整幅画面")
    if (
        _is_half(focus_x)
        and _is_half(focus_y)
        and box
        and _is_tenth_template(box[0])
        and _is_tenth_template(box[2])
        and (box[2] - box[0]) >= 0.55
        and side == "center"
    ):
        flags.append("中心坐标和宽框都像套模板，需要重新量")
    if gap_x is not None and focus_x is not None and abs(gap_x - focus_x) > 0.12:
        flags.append("left_gap/right_gap和focus_x对不上")
    return flags


def _retry_prompt(data: dict, flags: list) -> str:
    focus_x = _as_ratio(data.get("focus_x"))
    box = _subject_box(data)
    side = _parse_side(data.get("side")) or "?"
    box_txt = f"{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}" if box else "无"
    focus_txt = f"{focus_x:.2f}" if focus_x is not None else "无"
    why = "；".join(flags) or "坐标不合格"
    return (
        "上一轮没有认真量这张图，答案作废。"
        f"问题：{why}。当时 side={side}，focus_x={focus_txt}，框={box_txt}。"
        "请重新看图，只量当前主体，禁止再用 0.5 或 0.3/0.7 这种整十模板。"
        "框必须紧贴主体，不要包树、天空、路面。"
        "left_gap 是焦点到左边缘，right_gap 是焦点到右边缘，二者相加应接近 1。"
        "左右空隙差得明显时，focus_x 绝不能是 0.5。"
        "只返回 JSON："
        '{"has_subject":true或false,"side":"left或center或right","nearer_edge":"left或right或equal",'
        '"left_gap":0到1,"right_gap":0到1,"focus_x":0到1,"focus_y":0到1,'
        '"x1":0到1,"y1":0到1,"x2":0到1,"y2":0到1,"repeating":true或false,'
        '"reason":"主体是什么，左右空隙哪边更大"}。'
    )


def _subject_box(data: dict):
    raw = data.get("subject_box") or data.get("box")
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        values = raw[:4]
    else:
        values = [data.get("x1"), data.get("y1"), data.get("x2"), data.get("y2")]
    nums = [_as_ratio(item) for item in values]
    if any(item is None for item in nums):
        return None
    x1, y1, x2, y2 = nums
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    if right - left < 0.02 or bottom - top < 0.02:
        return None
    return [left, top, right, bottom]


def _box_unreliable(box) -> bool:
    left, top, right, bottom = box
    area = max(0.0, right - left) * max(0.0, bottom - top)
    if area < 0.12:
        return True
    if right < 0.45 or left > 0.55:
        return True
    return False


def _as_ratio(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.5:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _parse_side(value) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "left": "left",
        "l": "left",
        "right": "right",
        "r": "right",
        "center": "center",
        "centre": "center",
        "middle": "center",
        "central": "center",
        "左": "left",
        "左侧": "left",
        "左边": "left",
        "偏左": "left",
        "靠左": "left",
        "右": "right",
        "右侧": "right",
        "右边": "right",
        "偏右": "right",
        "靠右": "right",
        "中": "center",
        "中间": "center",
        "居中": "center",
    }
    if text in aliases:
        return aliases[text]
    return ""


def _side_from_reason(reason: str) -> str:
    text = reason or ""
    left = any(mark in text for mark in ("偏左", "左侧", "左边", "靠左", "左偏", "位于画面左"))
    right = any(mark in text for mark in ("偏右", "右侧", "右边", "靠右", "右偏", "位于画面右"))
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return ""


def _apply_side(side: str, subject_x, subject_y, box_x, box_y):
    if side == "left":
        if box_x is not None and box_x <= 0.42:
            subject_x = box_x
        elif subject_x is None or abs(float(subject_x) - 0.5) <= 0.12:
            subject_x = 0.33
        else:
            subject_x = min(float(subject_x), 0.38)
    elif side == "right":
        if box_x is not None and box_x >= 0.58:
            subject_x = box_x
        elif subject_x is None or abs(float(subject_x) - 0.5) <= 0.12:
            subject_x = 0.67
        else:
            subject_x = max(float(subject_x), 0.62)
    if subject_y is None:
        subject_y = box_y
    return subject_x, subject_y


def _position_from_xy(has_subject: bool, subject_x, subject_y, box=None) -> str:
    if not has_subject:
        return "none"
    if subject_x is None:
        return "other"
    d_center = abs(float(subject_x) - 0.5)
    d_third = min(abs(float(subject_x) - 1 / 3), abs(float(subject_x) - 2 / 3))
    if d_center <= 0.10 and d_center <= d_third + 0.02:
        return "center"
    if d_third <= 0.08:
        return "thirds"
    return "other"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "是", "有"}


def _subject_cache_key(path: str, info: dict) -> str:
    st = Path(path).stat()
    blob = "|".join(
        [
            "subject-v8-careful",
            str(Path(path).resolve()),
            str(int(st.st_mtime)),
            str(st.st_size),
            info.get("backend") or "",
            info.get("model") or "",
            info.get("mode") or "",
        ]
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def judge_type(
    path: str,
    typ: dict,
    feat: CompFeatures,
    visual: float,
) -> dict:
    description = (typ.get("description") or "").strip()
    if not description:
        return {"score": None, "reason": "", "used": False}
    info = _detect()
    if not info:
        return {"score": None, "reason": "没有可用的语言模型，仅按参考图相似度", "used": False}
    if info.get("mode") != "vision":
        return {
            "score": None,
            "reason": "当前模型看不见图片，文字定义未参与这张图的判断，仅按参考图布局相似度",
            "used": False,
        }

    key = _cache_key(path, typ, visual, info)
    cached = _read_cache(key)
    if cached is not None:
        return cached

    prompt = _prompt(typ, feat, visual)
    try:
        if info["mode"] == "vision":
            raw = _ask_vision(info, prompt, path)
        else:
            raw = _ask_text(info, prompt)
        parsed = _parse_score(raw)
    except Exception as exc:  # noqa: BLE001
        parsed = {"score": None, "reason": f"模型调用失败：{exc}", "used": False}
    if parsed.get("used"):
        _write_cache(key, parsed)
    return parsed


def _detect() -> Optional[dict]:
    now = time.time()
    if _backend_cache["info"] is not None and now - _backend_cache["at"] < 20:
        return _backend_cache["info"]
    info = _detect_now()
    _backend_cache["at"] = now
    _backend_cache["info"] = info
    return info


def _detect_now() -> Optional[dict]:
    forced_model = os.environ.get("COMPOSITION_LLM_MODEL", "").strip()
    forced_base = os.environ.get("COMPOSITION_LLM_BASE", "").strip()
    api_key = os.environ.get("COMPOSITION_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if forced_base or api_key:
        model = forced_model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        base = forced_base or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return {"backend": "openai", "model": model, "mode": "vision", "base": base.rstrip("/"), "key": api_key}

    ollama = _ollama_models()
    if not ollama:
        return None
    names = [m.get("name") or "" for m in ollama]
    vision = next((n for n in names if _is_vision_name(n)), "")
    if vision:
        return {"backend": "ollama", "model": vision, "mode": "vision", "base": "http://127.0.0.1:11434"}
    text = forced_model if forced_model in names else next((n for n in names if n.startswith("llama3.2")), "") or names[0]
    return {"backend": "ollama", "model": text, "mode": "layout", "base": "http://127.0.0.1:11434"}


def _ollama_models() -> list:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("models") or []
    except Exception:
        return []


def _is_vision_name(name: str) -> bool:
    low = name.lower()
    return any(mark in low for mark in VISION_MARKERS)


def _prompt(typ: dict, feat: CompFeatures, visual: float) -> str:
    return (
        "你是照片构图评审。请看附件里的这张照片，只按用户写下的文字定义判断它是否属于该构图类型。"
        "不要用自动提取的规律，不要评审定义文本本身写得好不好。\n"
        f"类型名称：{typ.get('name') or ''}\n"
        f"用户定义：{typ.get('description') or ''}\n"
        "只返回 JSON，不要其他文字："
        '{"score":0到1的小数,"reason":"一两句中文，必须针对这张照片的画面"}。'
        "1=照片非常符合定义，0=照片完全不符。"
    )


def _layout_text(feat: CompFeatures) -> str:
    cx, cy, spread, center, border = (float(x) for x in feat.stats[:5])
    peaks = "、".join(f"({p[0]:.2f},{p[1]:.2f})" for p in feat.peaks) or "无"
    thirds = [float(x) for x in feat.stats[7:16]] if feat.stats.size >= 16 else []
    grid = ""
    if thirds:
        grid = "九宫格质量[" + ",".join(f"{v:.2f}" for v in thirds) + "]"
    return (
        f"主体中心=({cx:.2f},{cy:.2f})，分散度={spread:.2f}，"
        f"中心能量={center:.2f}，边框能量={border:.2f}，"
        f"显著点{len(feat.peaks)}个：{peaks}。{grid}"
        "坐标原点在左上，1为右/下。"
    )


def _ask_text(info: dict, prompt: str) -> str:
    if info["backend"] == "ollama":
        return _ollama_chat(info, prompt, images=None)
    return _openai_chat(info, prompt, image_b64=None)


def _ask_vision(info: dict, prompt: str, path: str, temperature: float = 0.1) -> str:
    image_b64 = _jpeg_b64(path)
    if info["backend"] == "ollama":
        return _ollama_chat(info, prompt, images=[image_b64], temperature=temperature)
    return _openai_chat(info, prompt, image_b64=image_b64, temperature=temperature)


def _ollama_chat(info: dict, prompt: str, images: Optional[list], temperature: float = 0.1) -> str:
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    payload = {
        "model": info["model"],
        "stream": False,
        "format": "json",
        "options": {"temperature": float(temperature), "num_predict": 360},
        "messages": [message],
    }
    req = urllib.request.Request(
        f"{info['base']}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("message") or {}).get("content") or ""


def _openai_chat(info: dict, prompt: str, image_b64: Optional[str], temperature: float = 0.1) -> str:
    content: list = [{"type": "text", "text": prompt}]
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    payload = {
        "model": info["model"],
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    headers = {"Content-Type": "application/json"}
    if info.get("key"):
        headers["Authorization"] = f"Bearer {info['key']}"
    req = urllib.request.Request(
        f"{info['base']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _jpeg_b64(path: str) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((768, 768))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=86)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_score(raw: str) -> dict:
    text = (raw or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {"score": None, "reason": "模型没有返回可解析分数", "used": False}
        data = json.loads(text[start : end + 1])
    score = data.get("score")
    try:
        score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return {"score": None, "reason": "模型分数无效", "used": False}
    reason = str(data.get("reason") or "").strip()
    return {"score": score, "reason": reason, "used": True}


def _cache_key(path: str, typ: dict, visual: float, info: dict) -> str:
    st = Path(path).stat()
    blob = "|".join(
        [
            str(Path(path).resolve()),
            str(int(st.st_mtime)),
            str(st.st_size),
            typ.get("id") or "",
            typ.get("description") or "",
            "desc-only",
            f"{visual:.3f}",
            info.get("backend") or "",
            info.get("model") or "",
            info.get("mode") or "",
        ]
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _read_cache(key: str) -> Optional[dict]:
    file = CACHE / f"{key}.json"
    if not file.exists():
        return None
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_cache(key: str, payload: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def llm_call(fn, *args, **kwargs):
    with _lock:
        return fn(*args, **kwargs)
