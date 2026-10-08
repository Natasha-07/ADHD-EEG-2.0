from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from deep_model_utils import (
    build_experiment_split,
    fit_channel_standardizer,
    load_epoch_dataset,
    split_epoch_dataset,
    transform_epochs,
)


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "CSV Combined and Individual" / "Single_CSV" / "combined_eeg.csv"
MODEL_PATH = ROOT / "Results" / "CleanDeepBenchmark" / "combined_holdout" / "CNN" / "cnn_model.keras"
OUTPUT_DIR = ROOT / "Figures" / "updated_figures"
REFRESH_DIR = ROOT / "Results" / "refreshed_figures"

FIG7_TEMPLATE = ROOT / "Figures" / "fig7.png"
FIG8_TEMPLATE = ROOT / "Figures" / "fig8.png"

CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
]

REGION_MAP = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "Central": ["C3", "C4", "Cz"],
    "Parietal": ["P3", "P4", "P7", "P8", "Pz"],
    "Temporal": ["T7", "T8"],
    "Occipital": ["O1", "O2"],
}

REGION_STYLE = {
    "Frontal": {
        "sample_xy": (180, 210),
        "fig7_rect": (150, 165, 470, 445),
        "fig7_center": (315, 315),
        "fig8_rect": (170, 195, 500, 380),
        "fig8_center": (320, 280),
    },
    "Central": {
        "sample_xy": (650, 150),
        "fig7_rect": (610, 85, 845, 370),
        "fig7_center": (725, 215),
        "fig8_rect": (620, 95, 855, 260),
        "fig8_center": (735, 185),
    },
    "Parietal": {
        "sample_xy": (965, 240),
        "fig7_rect": (815, 225, 1070, 555),
        "fig7_center": (930, 380),
        "fig8_rect": (805, 245, 1070, 440),
        "fig8_center": (930, 325),
    },
    "Temporal": {
        "sample_xy": (300, 575),
        "fig7_rect": (255, 475, 535, 735),
        "fig7_center": (395, 600),
        "fig8_rect": (260, 470, 515, 665),
        "fig8_center": (385, 560),
    },
    "Occipital": {
        "sample_xy": (900, 585),
        "fig7_rect": (790, 470, 1040, 735),
        "fig7_center": (895, 605),
        "fig8_rect": (790, 470, 1040, 670),
        "fig8_center": (895, 560),
    },
}

FONT_PATH = Path(r"C:\Windows\Fonts\arialbd.ttf")
TEXT_COLOR = (35, 35, 35)


def compute_channel_importance(model: tf.keras.Model, X_sample: np.ndarray) -> np.ndarray:
    """Compute mean absolute input gradient per channel."""
    X_tensor = tf.convert_to_tensor(X_sample[np.newaxis, :, :], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor, training=False)

    gradients = tape.gradient(predictions, X_tensor)
    return tf.reduce_mean(tf.abs(gradients), axis=[0, 1]).numpy()


def compute_regional_importance(channel_importance: np.ndarray) -> dict[str, float]:
    regional = {}
    for region, region_channels in REGION_MAP.items():
        indices = [CHANNELS.index(ch) for ch in region_channels]
        regional[region] = float(np.sum(channel_importance[indices]))
    return regional


def fit_multiline_font(text: str, box_size: tuple[int, int], max_size: int) -> ImageFont.FreeTypeFont:
    """Shrink a bold font until the text fits inside the target box."""
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10), "white"))
    width_limit, height_limit = box_size

    for size in range(max_size, 24, -2):
        font = ImageFont.truetype(str(FONT_PATH), size)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=max(4, size // 8), align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= width_limit and text_h <= height_limit:
            return font

    return ImageFont.truetype(str(FONT_PATH), 24)


def smooth_text_region(img: Image.Image, rect: tuple[int, int, int, int], sample_xy: tuple[int, int]) -> None:
    """Cover an old text region with a soft-edged patch that matches the local region color."""
    x0, y0, x1, y1 = rect
    patch_size = (x1 - x0, y1 - y0)
    sample_color = img.getpixel(sample_xy)

    patch = Image.new("RGBA", patch_size, sample_color + (0,))
    alpha = Image.new("L", patch_size, 0)
    alpha_draw = ImageDraw.Draw(alpha)
    alpha_draw.ellipse((0, 0, patch_size[0], patch_size[1]), fill=245)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=18))
    patch.putalpha(alpha)

    base = img.convert("RGBA")
    base.alpha_composite(patch, dest=(x0, y0))
    img.paste(base.convert("RGB"))


def draw_centered_text(
    img: Image.Image,
    rect: tuple[int, int, int, int],
    center: tuple[int, int],
    text: str,
    max_font_size: int,
) -> None:
    """Draw centered multiline text inside a prepared region."""
    box_size = (rect[2] - rect[0] - 12, rect[3] - rect[1] - 12)
    font = fit_multiline_font(text, box_size, max_font_size)
    draw = ImageDraw.Draw(img)
    spacing = max(4, font.size // 8)
    draw.multiline_text(
        center,
        text,
        font=font,
        fill=TEXT_COLOR,
        anchor="mm",
        align="center",
        spacing=spacing,
    )


def render_fig7(base_image: Image.Image, region_top_channels: dict[str, dict[str, float]]) -> Image.Image:
    img = base_image.copy()
    for region, payload in region_top_channels.items():
        style = REGION_STYLE[region]
        smooth_text_region(img, style["fig7_rect"], style["sample_xy"])
        text = f"{payload['channel']}\n{payload['pct']:.1f}%"
        draw_centered_text(img, style["fig7_rect"], style["fig7_center"], text, max_font_size=82)
    return img


def render_fig8(base_image: Image.Image, regional_single_pct: dict[str, float]) -> Image.Image:
    img = base_image.copy()
    for region, pct in regional_single_pct.items():
        style = REGION_STYLE[region]
        smooth_text_region(img, style["fig8_rect"], style["sample_xy"])
        text = f"{pct:.2f}%"
        draw_centered_text(img, style["fig8_rect"], style["fig8_center"], text, max_font_size=66)
    return img


def main() -> None:
    raise RuntimeError(
        "This template-overlay visualization is retired because it can imply an "
        "anatomical mapping from scalp-sensor values. Use "
        "generate_sensor_space_topoplots.py instead."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REFRESH_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_epoch_dataset(CSV_PATH, epoch_seconds=2, overlap=0.5)
    split = build_experiment_split(
        dataset,
        scenario="combined_holdout",
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )
    X_train, _, X_test, y_train, _, y_test = split_epoch_dataset(dataset, split)

    scaler = fit_channel_standardizer(X_train)
    X_test_scaled = transform_epochs(X_test, scaler)

    model = tf.keras.models.load_model(MODEL_PATH)

    sample_idx = 19
    X_sample = X_test_scaled[sample_idx]
    y_sample = int(y_test[sample_idx])
    subject_sample = str(dataset.subject_ids[split.test_idx][sample_idx])

    channel_imp_single = compute_channel_importance(model, X_sample)
    channel_imp_single_pct = channel_imp_single / channel_imp_single.sum() * 100.0

    all_channel_imp = np.stack([compute_channel_importance(model, sample) for sample in X_test_scaled], axis=0)
    channel_imp_avg = all_channel_imp.mean(axis=0)

    regional_single = compute_regional_importance(channel_imp_single)
    regional_avg = compute_regional_importance(channel_imp_avg)

    regional_single_pct = {
        region: value / sum(regional_single.values()) * 100.0
        for region, value in regional_single.items()
    }
    regional_avg_pct = {
        region: value / sum(regional_avg.values()) * 100.0
        for region, value in regional_avg.items()
    }

    region_top_channels: dict[str, dict[str, float]] = {}
    for region, region_channels in REGION_MAP.items():
        best_channel = max(
            region_channels,
            key=lambda ch: channel_imp_single_pct[CHANNELS.index(ch)],
        )
        region_top_channels[region] = {
            "channel": best_channel,
            "pct": float(channel_imp_single_pct[CHANNELS.index(best_channel)]),
        }

    channel_rows = []
    for region in REGION_MAP:
        row = {
            "Region": region,
            "Top_Channel_For_Figure": region_top_channels[region]["channel"],
            "Top_Channel_Percentage_Sample19": region_top_channels[region]["pct"],
            "Regional_Contribution_Sample19": regional_single_pct[region],
            "Regional_Contribution_Average": regional_avg_pct[region],
        }
        channel_rows.append(row)

    values_df = pd.DataFrame(channel_rows)
    values_csv = REFRESH_DIR / "fig7_fig8_values_sample19.csv"
    values_df.to_csv(values_csv, index=False)

    meta = {
        "sample_idx": sample_idx,
        "subject_id": subject_sample,
        "true_label": "ADHD" if y_sample == 1 else "Control",
        "model_path": str(MODEL_PATH),
        "scenario": "combined_holdout",
        "protocol": "deep_subject_safe_combined_v1",
        "note": "fig7 uses top channel per region for sample_idx=19; fig8 uses regional contribution percentages for sample_idx=19",
    }
    meta_path = REFRESH_DIR / "fig7_fig8_values_sample19.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fig7_template = Image.open(FIG7_TEMPLATE).convert("RGB")
    fig8_template = Image.open(FIG8_TEMPLATE).convert("RGB")

    fig7_img = render_fig7(fig7_template, region_top_channels)
    fig8_img = render_fig8(fig8_template, regional_single_pct)

    fig7_out = OUTPUT_DIR / "fig7.png"
    fig8_out = OUTPUT_DIR / "fig8.png"
    fig7_img.save(fig7_out)
    fig8_img.save(fig8_out)

    print(f"Saved {fig7_out}")
    print(f"Saved {fig8_out}")
    print(f"Saved {values_csv}")
    print(f"Saved {meta_path}")
    print(values_df.to_string(index=False))


if __name__ == "__main__":
    main()
