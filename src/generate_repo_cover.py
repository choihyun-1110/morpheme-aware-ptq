from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1280
HEIGHT = 640

ROOT = Path("/home/choihyun/workspace")
OUT_PATH = ROOT / "results" / "git_cover_morpheme_ptq.png"
REGULAR_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def hex_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def gradient_background() -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT))
    px = image.load()

    top = hex_rgba("#061621")
    bottom = hex_rgba("#0d2e46")

    for y in range(HEIGHT):
        t = y / max(HEIGHT - 1, 1)
        row = tuple(int(lerp(top[i], bottom[i], t)) for i in range(3)) + (255,)
        for x in range(WIDTH):
            px[x, y] = row

    return image


def add_glow(base: Image.Image, bbox: tuple[int, int, int, int], color: str, alpha: int, blur: int) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(bbox, fill=hex_rgba(color, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(layer)


def draw_token_grid(base: Image.Image) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    x0 = 760
    y0 = 92
    cols = 6
    rows = 5
    gap_x = 72
    gap_y = 68

    node_colors = ["#8be0d4", "#ffd166", "#ff7b72"]

    for row in range(rows):
        for col in range(cols):
            x = x0 + col * gap_x + (row % 2) * 10
            y = y0 + row * gap_y

            if col < cols - 1:
                draw.line((x, y, x + gap_x - 14, y), fill=hex_rgba("#8be0d4", 80), width=2)
            if row < rows - 1:
                draw.line((x, y, x + 10, y + gap_y), fill=hex_rgba("#55c2ff", 68), width=2)

            r = 10 if (row + col) % 3 else 13
            color = node_colors[(row + col) % len(node_colors)]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=hex_rgba(color, 225))
            draw.ellipse((x - r - 5, y - r - 5, x + r + 5, y + r + 5), outline=hex_rgba("#dff7ff", 48), width=1)

    layer = layer.filter(ImageFilter.GaussianBlur(0.2))
    base.alpha_composite(layer)


def draw_faint_hangul(base: Image.Image, font: ImageFont.FreeTypeFont) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((40, 42), "형태소", font=font, fill=hex_rgba("#d9f2ff", 26))
    draw.text((34, 274), "한국어", font=font, fill=hex_rgba("#8be0d4", 22))
    layer = layer.filter(ImageFilter.GaussianBlur(0.6))
    base.alpha_composite(layer)


def draw_panel(base: Image.Image) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    panel = (718, 58, 1222, 584)
    draw.rounded_rectangle(panel, radius=34, fill=hex_rgba("#0a2030", 168), outline=hex_rgba("#a7e9dd", 90), width=2)

    inner = (746, 326, 1190, 552)
    draw.rounded_rectangle(inner, radius=24, fill=hex_rgba("#082434", 182), outline=hex_rgba("#79d8c8", 70), width=1)

    top_card = (756, 84, 1186, 182)
    draw.rounded_rectangle(top_card, radius=24, fill=hex_rgba("#12364d", 170))

    mid_card = (756, 198, 960, 294)
    draw.rounded_rectangle(mid_card, radius=24, fill=hex_rgba("#103145", 170))

    stat_card = (980, 198, 1186, 294)
    draw.rounded_rectangle(stat_card, radius=24, fill=hex_rgba("#1f3750", 170))

    layer = layer.filter(ImageFilter.GaussianBlur(0.2))
    base.alpha_composite(layer)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], content: str, font: ImageFont.FreeTypeFont, fill: str, alpha: int = 255) -> None:
    draw.text(xy, content, font=font, fill=hex_rgba(fill, alpha))


def draw_chips(draw: ImageDraw.ImageDraw, items: Iterable[str], start: tuple[int, int]) -> None:
    x, y = start
    chip_font = load_font(BOLD_FONT, 22)

    for item in items:
        bbox = draw.textbbox((0, 0), item, font=chip_font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        box = (x, y, x + w + 30, y + h + 18)
        draw.rounded_rectangle(box, radius=16, fill=hex_rgba("#0f3045", 220), outline=hex_rgba("#9ed9d0", 110), width=1)
        draw.text((x + 15, y + 8), item, font=chip_font, fill=hex_rgba("#e9fbff"))
        x += w + 44


def draw_bars(draw: ImageDraw.ImageDraw) -> None:
    chart_x = 790
    chart_y = 360
    chart_w = 350
    chart_h = 140

    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=hex_rgba("#d7f8ff", 120), width=2)
    draw.line((chart_x, chart_y + 10, chart_x, chart_y + chart_h), fill=hex_rgba("#d7f8ff", 120), width=2)

    values = [("A", 0.5981, "#ff7b72"), ("B", 0.6176, "#ffd166"), ("C_v3", 0.6356, "#7ce0d0")]
    baseline = 0.55
    scale = 0.11
    bar_width = 74
    gap = 34

    label_font = load_font(BOLD_FONT, 22)
    value_font = load_font(BOLD_FONT, 20)

    for idx, (label, value, color) in enumerate(values):
        x = chart_x + 36 + idx * (bar_width + gap)
        norm = max(0.05, min(1.0, (value - baseline) / scale))
        bar_h = int(norm * 110)
        y = chart_y + chart_h - bar_h

        draw.rounded_rectangle((x, y, x + bar_width, chart_y + chart_h), radius=18, fill=hex_rgba(color, 230))
        draw.text((x + 11, chart_y + chart_h + 14), label, font=label_font, fill=hex_rgba("#eafcff"))
        draw.text((x + 4, y - 28), f"{value:.3f}", font=value_font, fill=hex_rgba("#d7f8ff"))


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image = gradient_background()
    add_glow(image, (-120, -60, 480, 500), "#0fb4a2", 80, 120)
    add_glow(image, (830, -100, 1370, 340), "#ffd166", 58, 150)
    add_glow(image, (900, 220, 1430, 760), "#1fb6ff", 44, 130)
    add_glow(image, (180, 250, 620, 760), "#ff7b72", 40, 130)
    draw_panel(image)
    draw_token_grid(image)

    huge_font = load_font(BOLD_FONT, 170)
    draw_faint_hangul(image, huge_font)

    draw = ImageDraw.Draw(image)

    label_font = load_font(BOLD_FONT, 24)
    title_font = load_font(BOLD_FONT, 66)
    subtitle_font = load_font(REGULAR_FONT, 34)
    body_font = load_font(REGULAR_FONT, 25)
    body_bold = load_font(BOLD_FONT, 25)
    panel_title = load_font(BOLD_FONT, 28)
    panel_big = load_font(BOLD_FONT, 42)
    small_font = load_font(REGULAR_FONT, 22)
    stat_font = load_font(BOLD_FONT, 36)

    text(draw, (72, 66), "RESEARCH REPOSITORY", label_font, "#9ed9d0")
    text(draw, (72, 112), "Morpheme-Aware", title_font, "#f5fbff")
    text(draw, (72, 188), "PTQ Calibration", title_font, "#f5fbff")
    text(draw, (72, 272), "for Korean LLMs", subtitle_font, "#d5eef7")

    text(draw, (72, 334), "형태소 다양성 기반 calibration으로", body_font, "#dff7ff")
    text(draw, (72, 372), "4-bit GPTQ 성능 보존을 끌어올린 실험 리포지토리", body_bold, "#f9f3d0")

    draw_chips(
        draw,
        ["GPTQ 4-bit", "KoBEST 97.4% retention", "Korean morpheme diversity"],
        (72, 440),
    )

    text(draw, (72, 540), "Calibration data selection for Korean and multilingual LLM quantization", small_font, "#d5eef7", 210)

    text(draw, (784, 98), "Key Signal", panel_title, "#b8fff4")
    text(draw, (784, 130), "Better calibration sentences", small_font, "#dff7ff")
    text(draw, (784, 154), "reduce post-quantization loss", small_font, "#dff7ff")

    text(draw, (780, 214), "Main Result", panel_title, "#dff7ff")
    text(draw, (780, 246), "C_v3 > A/B", panel_big, "#ffd166")

    text(draw, (1000, 214), "Retention", panel_title, "#dff7ff")
    text(draw, (1000, 246), "97.4%", stat_font, "#8be0d4")

    text(draw, (790, 330), "KoBEST avg comparison", panel_title, "#dff7ff")
    draw_bars(draw)

    image.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
