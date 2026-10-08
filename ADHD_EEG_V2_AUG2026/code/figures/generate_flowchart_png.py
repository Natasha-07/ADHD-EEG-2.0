from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Figures" / "updated_figures"
OUT_PATH = OUT_DIR / "flowchart_updated.png"


WIDTH = 1200
HEIGHT = 2100
BG = "#FFFFFF"
BOX_FILL = "#F7FAFC"
BOX_OUTLINE = "#1F3C5A"
TEXT = "#1A1A1A"
SUBTEXT = "#334155"
ARROW = "#1F3C5A"


BOX_W = 820
BOX_H = 225
BOX_X = (WIDTH - BOX_W) // 2
START_Y = 170
STEP_Y = 305
RADIUS = 28


TITLE = "Leakage-Safe EEG-Based ADHD Classification Pipeline"

BOXES = [
    (
        "Data Acquisition & Harmonization",
        [
            "Child and adult EEG cohorts",
            "19-channel alignment",
            "Subject, label, and age-group preservation",
        ],
    ),
    (
        "Epoching & Preprocessing",
        [
            "2-s windows, 50% overlap",
            "Child/adult length harmonization",
            "Train-only channel normalization",
        ],
    ),
    (
        "Leakage-Safe Subject-Wise Splitting",
        [
            "Combined holdout",
            "Child-to-adult",
            "Adult-to-child",
        ],
    ),
    (
        "Deep Model Benchmarking",
        [
            "CNN, LSTM, Bi-LSTM, Temporal Transformer Baseline",
        ],
    ),
    (
        "Evaluation",
        [
            "Accuracy, Precision, Recall",
            "F1-score, ROC-AUC",
            "Confusion matrix",
        ],
    ),
    (
        "Exploratory Interpretability",
        [
            "LIME, SHAP",
            "Approximate channel-to-region grouping",
        ],
    ),
]


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\calibrib.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\segoeui.ttf",
                r"C:\Windows\Fonts\calibri.ttf",
            ]
        )

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_centered_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    start_y: int,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = start_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((center_x - w / 2, y), line, font=font, fill=fill)
        y += h + line_gap
    return y


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    title_font = get_font(42, bold=True)
    heading_font = get_font(33, bold=True)
    body_font = get_font(28, bold=False)

    title_bbox = draw.textbbox((0, 0), TITLE, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((WIDTH - title_w) / 2, 55), TITLE, font=title_font, fill=TEXT)

    for idx, (heading, bullets) in enumerate(BOXES):
        y = START_Y + idx * STEP_Y
        x1 = BOX_X
        y1 = y
        x2 = BOX_X + BOX_W
        y2 = y + BOX_H

        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=RADIUS,
            fill=BOX_FILL,
            outline=BOX_OUTLINE,
            width=4,
        )

        center_x = (x1 + x2) // 2
        text_y = y1 + 24
        text_y = draw_centered_wrapped_text(
            draw,
            heading,
            center_x=center_x,
            start_y=text_y,
            font=heading_font,
            fill=TEXT,
            max_width=BOX_W - 90,
            line_gap=6,
        )

        text_y += 12
        for bullet in bullets:
            text_y = draw_centered_wrapped_text(
                draw,
                bullet,
                center_x=center_x,
                start_y=text_y,
                font=body_font,
                fill=SUBTEXT,
                max_width=BOX_W - 100,
                line_gap=5,
            )
            text_y += 8

        if idx < len(BOXES) - 1:
            arrow_x = WIDTH // 2
            arrow_top = y2 + 16
            arrow_bottom = y2 + (STEP_Y - BOX_H) - 16
            draw.line((arrow_x, arrow_top, arrow_x, arrow_bottom), fill=ARROW, width=6)
            head = 16
            draw.polygon(
                [
                    (arrow_x, arrow_bottom + head),
                    (arrow_x - head, arrow_bottom - 4),
                    (arrow_x + head, arrow_bottom - 4),
                ],
                fill=ARROW,
            )

    image.save(OUT_PATH, format="PNG")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
