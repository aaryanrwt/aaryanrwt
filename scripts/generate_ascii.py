"""
Generate a cinematic, seamless animated ASCII art GIF from the reference image
for the GitHub profile README hero section.
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths
REF_IMAGE_PATH = r"C:\Users\Aaryan Rawat\.gemini\antigravity\brain\fa8bf351-4a8c-42dd-9666-2d13fb09142f\.user_uploaded\media_1795426772752_261118fa.jpg"
OUTPUT_GIF_PATH = os.path.join("assets", "dragon-ascii.gif")
FONT_PATH = r"C:\Windows\Fonts\consola.ttf"

# Grid & rendering parameters
COLS = 132
ROWS = 136
CELL_W = 6
CELL_H = 10
FONT_SIZE = 10

N_FRAMES = 32
FRAME_DURATION_MS = 100  # 3.2s seamless loop

# Monochromatic cinematic palette + amber eyes + burnt orange embers
PALETTE_STOPS = [
    (0.00, np.array([21, 25, 29], dtype=np.float32)),     # #15191D Charcoal / near-black
    (0.18, np.array([48, 56, 61], dtype=np.float32)),     # #30383D Deep shadow
    (0.35, np.array([98, 107, 112], dtype=np.float32)),   # #626B70 Cool grey
    (0.50, np.array([146, 153, 155], dtype=np.float32)),  # #92999B Misty grey
    (0.67, np.array([200, 204, 205], dtype=np.float32)),  # #C8CCCD Cool silver
    (0.84, np.array([228, 229, 227], dtype=np.float32)),  # #E4E5E3 Silver white
    (1.00, np.array([244, 240, 233], dtype=np.float32)),  # #F4F0E9 Ivory white
]

AMBER_EYE_BASE = np.array([193, 138, 75], dtype=np.float32)   # #C18A4B
AMBER_EYE_PEAK = np.array([235, 175, 92], dtype=np.float32)   # Soft glowing amber
EMBER_BRIGHT = np.array([193, 138, 75], dtype=np.float32)     # #C18A4B
EMBER_DIM = np.array([166, 93, 56], dtype=np.float32)         # #A65D38
BG_COLOR = (21, 25, 29)                                       # #15191D

# Character sets ordered by visual ink density on dark background
BRIGHT_RAMP = " .,:;!+*#%@"
SCALE_RAMP = " ░▒▓█"
SHADOW_RAMP = "  .,:;il!|/\\"
MIST_CHARS = [" ", " ", ".", ",", ":", "░"]
EMBER_CHARS = ["*", "+", "•", "·", "°"]

ALL_CHARS = sorted(
    set(BRIGHT_RAMP + SCALE_RAMP + SHADOW_RAMP + "".join(MIST_CHARS) + "".join(EMBER_CHARS) + "│─╱╲◉◈")
)


def build_glyph_atlas(font_path: str, font_size: int, cell_w: int, cell_h: int) -> dict[str, np.ndarray]:
    """Pre-rasterize each glyph into a float32 alpha mask (cell_h, cell_w) in [0, 1]."""
    font = ImageFont.truetype(font_path, font_size)
    atlas: dict[str, np.ndarray] = {}
    for ch in ALL_CHARS:
        im = Image.new("L", (cell_w, cell_h), 0)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), ch, fill=255, font=font)
        atlas[ch] = np.asarray(im, dtype=np.float32) / 255.0
    return atlas


def interpolate_palette(lum: np.ndarray) -> np.ndarray:
    """Map luminance array (ROWS, COLS) in [0, 1] to RGB float32 (ROWS, COLS, 3)."""
    rgb = np.zeros((*lum.shape, 3), dtype=np.float32)
    for i in range(len(PALETTE_STOPS) - 1):
        t0, c0 = PALETTE_STOPS[i]
        t1, c1 = PALETTE_STOPS[i + 1]
        mask = (lum >= t0) & (lum <= t1) if i == len(PALETTE_STOPS) - 2 else (lum >= t0) & (lum < t1)
        alpha = np.clip((lum[mask] - t0) / (t1 - t0 + 1e-6), 0.0, 1.0)[:, None]
        rgb[mask] = c0 * (1.0 - alpha) + c1 * alpha
    return rgb


def create_motion_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create smooth spatial masks in source image space:
    - breath_mask: dragon head & upper body breathing
    - wind_mask: upper mane, horns, whiskers, and monk's trailing hair
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = yy / h
    xn = xx / w

    monk_dist = ((xn - 0.50) / 0.11) ** 2 + ((yn - 0.69) / 0.09) ** 2
    monk_lock = np.exp(-monk_dist * 1.8)
    rock_lock = np.clip((yn - 0.72) / 0.12, 0.0, 1.0)
    static_lock = np.clip(np.maximum(monk_lock, rock_lock), 0.0, 1.0)

    dragon_body = np.exp(-(((xn - 0.50) / 0.38) ** 2 + ((yn - 0.38) / 0.25) ** 2))
    breath_mask = dragon_body * (1.0 - static_lock)

    mane_mask = np.clip((0.32 - yn) / 0.20, 0.0, 1.0)
    whisker_mask = np.exp(-(((xn - 0.50) / 0.26) ** 2 + ((yn - 0.43) / 0.09) ** 2))
    hair_tail_mask = np.exp(-(((xn - 0.59) / 0.09) ** 2 + ((yn - 0.66) / 0.06) ** 2))
    wind_mask = np.clip(mane_mask * 0.9 + whisker_mask * 0.8 + hair_tail_mask * 0.45, 0.0, 1.0) * (1.0 - rock_lock)

    return breath_mask, wind_mask


def generate_embers(n_embers: int = 36, seed: int = 42) -> list[dict]:
    """Generate seeded ember particles for the lower foreground and mist."""
    rng = np.random.default_rng(seed)
    embers = []
    for i in range(n_embers):
        embers.append(
            {
                "x0": float(rng.uniform(0.05, 0.95)),
                "y0": float(rng.uniform(0.66, 0.98)),
                "rise": float(rng.uniform(0.06, 0.15)),
                "sway_amp": float(rng.uniform(0.008, 0.022)),
                "sway_phase": float(rng.uniform(0, 2 * math.pi)),
                "phase": float(rng.uniform(0, 1)),
                "char": EMBER_CHARS[i % len(EMBER_CHARS)],
                "bright": float(rng.uniform(0.65, 1.0)),
            }
        )
    return embers


def main() -> None:
    if not os.path.exists(REF_IMAGE_PATH):
        raise FileNotFoundError(f"Reference image not found: {REF_IMAGE_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_GIF_PATH), exist_ok=True)

    bgr = cv2.imread(REF_IMAGE_PATH)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    # Contrast-enhanced grayscale preserving dramatic darks and ivory highlights
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_u8).astype(np.float32) / 255.0
    base_gray = gray_u8.astype(np.float32) / 255.0
    lum_src = np.clip(0.64 * base_gray + 0.36 * enhanced, 0.0, 1.0)

    # Warmth map (R - B) dilated slightly so small features (eyes, collar) survive grid downsampling
    r_ch = rgb[:, :, 0].astype(np.float32) / 255.0
    b_ch = rgb[:, :, 2].astype(np.float32) / 255.0
    warmth_src = np.clip((r_ch - b_ch - 0.07) / 0.22, 0.0, 1.0)

    yy_src, xx_src = np.mgrid[0:h, 0:w].astype(np.float32)
    yn_src = yy_src / h
    xn_src = xx_src / w

    eye_region_src = (
        ((yn_src > 0.285) & (yn_src < 0.318))
        & (((xn_src > 0.40) & (xn_src < 0.47)) | ((xn_src > 0.54) & (xn_src < 0.61)))
    ).astype(np.float32)
    eye_map_src = cv2.dilate(warmth_src * eye_region_src, np.ones((7, 7), np.uint8))

    collar_region_src = ((yn_src > 0.61) & (yn_src < 0.73) & (xn_src > 0.44) & (xn_src < 0.56)).astype(np.float32)
    collar_map_src = cv2.dilate(warmth_src * collar_region_src, np.ones((5, 5), np.uint8))

    breath_mask, wind_mask = create_motion_masks(h, w)
    atlas = build_glyph_atlas(FONT_PATH, FONT_SIZE, CELL_W, CELL_H)
    embers = generate_embers()

    gy, gx = np.mgrid[0:ROWS, 0:COLS].astype(np.float32)
    gyn = gy / ROWS
    gxn = gx / COLS

    frames_rgb: list[Image.Image] = []

    for f_idx in range(N_FRAMES):
        t = f_idx / N_FRAMES
        theta = 2.0 * math.pi * t

        dy = (1.5 * math.sin(theta)) * breath_mask
        dx = (
            1.6 * np.sin(theta + yn_src * 9.0) * wind_mask
            + 0.6 * math.cos(theta) * breath_mask * (xn_src - 0.5)
        )

        map_x = np.clip(xx_src - dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(yy_src - dy, 0, h - 1).astype(np.float32)

        warped_lum = cv2.remap(lum_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_eye = cv2.remap(eye_map_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_collar = cv2.remap(collar_map_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        lum_grid = cv2.resize(warped_lum, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        eye_grid = cv2.resize(warped_eye, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        collar_grid = cv2.resize(warped_collar, (COLS, ROWS), interpolation=cv2.INTER_AREA)

        gx_grad = cv2.Sobel(lum_grid, cv2.CV_32F, 1, 0, ksize=3)
        gy_grad = cv2.Sobel(lum_grid, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(gx_grad ** 2 + gy_grad ** 2)
        edge_ang = np.arctan2(gy_grad, gx_grad)

        local_blur = cv2.GaussianBlur(lum_grid, (5, 5), 1.2)
        texture = np.abs(lum_grid - local_blur)

        # Mystical drifting fog
        fog_wave1 = np.sin(gxn * 2.0 * math.pi * 1.0 - theta + gyn * 3.0)
        fog_wave2 = np.cos(gxn * 2.0 * math.pi * 2.0 + theta - gyn * 4.5)
        fog_field = 0.5 + 0.5 * (0.6 * fog_wave1 + 0.4 * fog_wave2)

        edge_side = np.clip(np.abs(gxn - 0.5) * 1.4, 0.0, 1.0)
        mist_zone = np.exp(-((gyn - 0.48) / 0.30) ** 2) * (0.45 + 0.55 * edge_side)
        monk_clear = np.exp(-(((gxn - 0.50) / 0.11) ** 2 + ((gyn - 0.69) / 0.09) ** 2))
        face_clear = np.exp(-(((gxn - 0.50) / 0.15) ** 2 + ((gyn - 0.33) / 0.12) ** 2))
        fog_strength = mist_zone * (1.0 - 0.85 * monk_clear) * (1.0 - 0.75 * face_clear)

        atmo_pulse = 0.015 * math.sin(theta)
        lum_anim = np.clip(lum_grid + (fog_field - 0.5) * 0.11 * fog_strength + atmo_pulse, 0.0, 1.0)

        color_grid = interpolate_palette(lum_anim)

        # Glowing amber eyes
        eye_pulse = 0.5 + 0.5 * math.sin(theta)
        eye_color = AMBER_EYE_BASE * (1.0 - eye_pulse) + AMBER_EYE_PEAK * eye_pulse
        eye_mask = eye_grid > 0.18
        if np.any(eye_mask):
            color_grid[eye_mask] = eye_color

        # Subtle burnt-orange accent on monk's collar
        collar_mask = (collar_grid > 0.20) & (~eye_mask)
        if np.any(collar_mask):
            w_col = np.clip(collar_grid[collar_mask] * 1.1, 0.0, 0.85)[:, None]
            color_grid[collar_mask] = color_grid[collar_mask] * (1.0 - w_col) + EMBER_DIM * w_col

        canvas = np.zeros((ROWS * CELL_H, COLS * CELL_W, 3), dtype=np.float32)
        canvas[:, :] = BG_COLOR

        for r in range(ROWS):
            y0 = r * CELL_H
            y1 = y0 + CELL_H
            for c in range(COLS):
                x0 = c * CELL_W
                x1 = x0 + CELL_W

                l_val = float(lum_anim[r, c])
                e_val = float(edge_mag[r, c])
                t_val = float(texture[r, c])
                f_val = float(fog_field[r, c] * fog_strength[r, c])

                if eye_grid[r, c] > 0.18:
                    ch = "◈" if eye_pulse > 0.45 else "◉"
                elif e_val > 0.74 and l_val > 0.18:
                    ang = abs(float(edge_ang[r, c]))
                    if ang < 0.35 or ang > 2.79:
                        ch = "│"
                    elif 1.22 < ang < 1.92:
                        ch = "─"
                    elif (edge_ang[r, c] > 0 and edge_ang[r, c] < 1.57) or (edge_ang[r, c] < -1.57):
                        ch = "╱"
                    else:
                        ch = "╲"
                elif l_val < 0.21:
                    idx = int(np.clip((l_val / 0.21) * (len(SHADOW_RAMP) - 1), 0, len(SHADOW_RAMP) - 1))
                    ch = SHADOW_RAMP[idx]
                elif t_val > 0.048 and 0.35 < l_val < 0.88:
                    s_idx = int(np.clip(((l_val - 0.35) / 0.53) * (len(SCALE_RAMP) - 1), 0, len(SCALE_RAMP) - 1))
                    ch = SCALE_RAMP[s_idx]
                elif f_val > 0.30 and t_val < 0.03 and l_val < 0.62:
                    m_idx = int(np.clip(f_val * (len(MIST_CHARS) - 1), 0, len(MIST_CHARS) - 1))
                    ch = MIST_CHARS[m_idx]
                else:
                    b_idx = int(np.clip(l_val * (len(BRIGHT_RAMP) - 1), 0, len(BRIGHT_RAMP) - 1))
                    ch = BRIGHT_RAMP[b_idx]

                alpha = atlas.get(ch)
                if alpha is not None:
                    fg = color_grid[r, c]
                    a3 = alpha[:, :, None]
                    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - a3) + fg * a3

        # Overlay floating burnt-orange embers
        for emb in embers:
            life = (emb["phase"] + t) % 1.0
            ey = emb["y0"] - life * emb["rise"]
            ex = emb["x0"] + emb["sway_amp"] * math.sin(theta + emb["sway_phase"])
            er = int(round(ey * (ROWS - 1)))
            ec = int(round(ex * (COLS - 1)))
            if 0 <= er < ROWS and 0 <= ec < COLS:
                fade = math.sin(math.pi * life) * emb["bright"]
                if fade > 0.16:
                    e_col = EMBER_DIM * (1.0 - fade) + EMBER_BRIGHT * fade
                    alpha = atlas[emb["char"]]
                    y0, y1 = er * CELL_H, (er + 1) * CELL_H
                    x0, x1 = ec * CELL_W, (ec + 1) * CELL_W
                    a3 = (alpha * min(1.0, fade * 1.25))[:, :, None]
                    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - a3) + e_col * a3

        frame_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
        frames_rgb.append(Image.fromarray(frame_u8, mode="RGB"))

    # Use 128-color shared palette + explicitly anchor amber/ember swatches so warm colors never get dropped
    swatch_im = Image.new("RGB", (frames_rgb[0].width, frames_rgb[0].height * 2 + 20))
    swatch_im.paste(frames_rgb[0], (0, 0))
    swatch_im.paste(frames_rgb[N_FRAMES // 2], (0, frames_rgb[0].height))
    draw_sw = ImageDraw.Draw(swatch_im)
    warm_swatches = [(193, 138, 75), (235, 175, 92), (166, 93, 56), (244, 240, 233), (21, 25, 29)]
    for idx_sw, sw_col in enumerate(warm_swatches):
        draw_sw.rectangle([idx_sw * 60, frames_rgb[0].height * 2, (idx_sw + 1) * 60, frames_rgb[0].height * 2 + 20], fill=sw_col)

    global_palette = swatch_im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

    quantized_frames = [
        fr.quantize(palette=global_palette, dither=Image.Dither.NONE) for fr in frames_rgb
    ]

    quantized_frames[0].save(
        OUTPUT_GIF_PATH,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )

    size_kb = os.path.getsize(OUTPUT_GIF_PATH) / 1024.0
    print(f"Saved {OUTPUT_GIF_PATH}: {len(quantized_frames)} frames, {frames_rgb[0].size}, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
