"""Create a small mixed set of images for local preview and testing."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "samples"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name, quality=92)


def landscape() -> None:
    img = Image.new("RGB", (960, 640), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y in range(640):
        t = y / 639
        r = int(32 + 90 * t)
        g = int(58 + 70 * t)
        b = int(92 + 40 * (1 - t))
        draw.line([(0, y), (959, y)], fill=(r, g, b))
    draw.ellipse((520, 40, 700, 220), fill=(255, 214, 140))
    _save(img, "01_landscape.png")


def document() -> None:
    img = Image.new("RGB", (900, 1200), (248, 246, 240))
    draw = ImageDraw.Draw(img)
    title = _font(42)
    body = _font(26)
    draw.text((64, 56), "会议纪要 · 产品评审", font=title, fill=(28, 25, 23))
    paragraphs = [
        "本次评审围绕图片文字占比分类工具展开。目标是把含有明显文字区域",
        "的图片自动分拣到独立文件夹，并提供可调节阈值的可视化预览。",
        "",
        "一、检测方式：识别文字包围盒，计算并集面积占整图比例。",
        "二、阈值可在页面上实时调整，无需重新扫描。",
        "三、导出时按当前阈值复制文件，不改动原始目录。",
        "",
        "参会人认为，文档截图、字幕海报、界面截图应进入「含文字」一类；",
        "风景、实物照片、纯色纹理则保留在其余图片中。",
        "",
        "后续可以补充批量缩略图缓存，以及按文字语言再细分。",
    ]
    y = 140
    for line in paragraphs:
        draw.text((64, y), line, font=body, fill=(55, 48, 42))
        y += 42
    _save(img, "02_document.png")


def captioned_photo() -> None:
    img = Image.new("RGB", (1080, 720), (18, 22, 28))
    draw = ImageDraw.Draw(img)
    for i in range(18):
        x0, y0 = 40 + i * 28, 30 + (i % 5) * 16
        draw.rectangle((x0, y0, x0 + 220, y0 + 140), outline=(70, 90, 110), width=2)
    draw.rectangle((0, 600, 1080, 720), fill=(8, 8, 10))
    draw.text((36, 628), "夜航 · 港口灯火   2024 / 11", font=_font(36), fill=(240, 236, 228))
    _save(img, "03_caption_bar.png")


def ui_screenshot() -> None:
    img = Image.new("RGB", (1280, 800), (244, 244, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1280, 64), fill=(24, 24, 27))
    draw.text((28, 16), "设置  /  账户  /  通知", font=_font(28), fill=(250, 250, 250))
    draw.rounded_rectangle((40, 100, 1240, 760), radius=16, fill=(255, 255, 255))
    rows = [
        ("登录设备管理", "查看最近使用过的浏览器与手机"),
        ("两步验证", "已开启，建议保留备用验证码"),
        ("邮件通知", "产品更新与安全提醒"),
        ("桌面通知", "仅在有人评论时提醒"),
        ("语言与地区", "简体中文 · 中国"),
    ]
    y = 140
    for title, desc in rows:
        draw.text((80, y), title, font=_font(30), fill=(24, 24, 27))
        draw.text((80, y + 44), desc, font=_font(22), fill=(113, 113, 122))
        draw.line((80, y + 96, 1200, y + 96), fill=(228, 228, 231), width=1)
        y += 118
    _save(img, "04_settings_ui.png")


def texture() -> None:
    img = Image.new("RGB", (800, 800), (210, 186, 150))
    draw = ImageDraw.Draw(img)
    for i in range(0, 800, 16):
        draw.line([(i, 0), (0, i)], fill=(188, 160, 122), width=1)
        draw.line([(799, i), (i, 799)], fill=(188, 160, 122), width=1)
    _save(img, "05_texture.png")


def poster() -> None:
    img = Image.new("RGB", (720, 1080), (127, 29, 29))
    draw = ImageDraw.Draw(img)
    draw.rectangle((48, 48, 672, 1032), outline=(252, 211, 77), width=4)
    draw.text((90, 160), "秋日", font=_font(140), fill=(254, 243, 199))
    draw.text((96, 340), "市集", font=_font(140), fill=(254, 243, 199))
    draw.text((100, 820), "SAT  14:00  河边公园", font=_font(36), fill=(254, 215, 170))
    _save(img, "06_poster.png")


def main() -> None:
    landscape()
    document()
    captioned_photo()
    ui_screenshot()
    texture()
    poster()
    print(f"wrote samples to {OUT}")


if __name__ == "__main__":
    main()
