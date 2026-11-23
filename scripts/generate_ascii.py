"""
Generate a high-detail, true-palette, seamless animated ASCII art GIF from the
reference image for the GitHub profile README hero section.
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths
REF_IMAGE_PATH = r"C:\Users\Aaryan Rawat\.gemini\antigravity\brain\fa8bf351-4a8c-42dd-9666-2d13fb09142f\.user_uploaded\media_1795426772752_261118fa.jpg"
OUTPUT_GIF_PATH = os.path.join("assets", "dragon-ascii.gif")
FONT_PATH = r"C:\Windows\Fonts\CascadiaMono.ttf"

# Grid & rendering parameters (5x9 cells on 164x156 -> 820 x 1404, exact 0.5840 aspect match)
COLS = 164
ROWS = 156
CELL_W = 5
CELL_H = 9
FONT_SIZE = 9
Y_OFFSET = -1

N_FRAMES = 30
FRAME_DURATION_MS = 100  # 3.0s seamless loop
HYSTERESIS_MARGIN = 0.016

# Pure crisp ASCII + box-drawing glyphs (zero blurry ░▒▓ boxes, zero random alphabet soup)
EDGE_CHARS = list(" .',:;-~=+*!|/\\()<>^v_╱╲╭╮╰╯│─#%@")
TONAL_RAMP = list(" .',:;-~+*=%&#@")
ROCK_RAMP = list(" .',:;~-=+*#%@")
MIST_RAMP = list("      ..,:;")
SCALE_RAMP = list(":+*=%&#@")
HAKAMA_FOLD_RAMP = list(" .,:;-~=/\\()")

OVERLAY_CHARS = ["◉", "◈", "•", "·", "°"]
ALL_CHARS = []
for ch in EDGE_CHARS + TONAL_RAMP + ROCK_RAMP + MIST_RAMP + SCALE_RAMP + HAKAMA_FOLD_RAMP + OVERLAY_CHARS:
    if ch not in ALL_CHARS:
        ALL_CHARS.append(ch)
CHAR_TO_IDX = {ch: i for i, ch in enumerate(ALL_CHARS)}

EDGE_INDICES = np.array([CHAR_TO_IDX[c] for c in EDGE_CHARS], dtype=np.int32)
ROCK_EDGE_EXCLUDE = np.array([c in "()<>^v" for c in EDGE_CHARS], dtype=bool)
TONAL_INDICES = np.array([CHAR_TO_IDX[c] for c in TONAL_RAMP], dtype=np.int32)
ROCK_INDICES = np.array([CHAR_TO_IDX[c] for c in ROCK_RAMP], dtype=np.int32)
MIST_INDICES = np.array([CHAR_TO_IDX[c] for c in MIST_RAMP], dtype=np.int32)
SCALE_INDICES = np.array([CHAR_TO_IDX[c] for c in SCALE_RAMP], dtype=np.int32)
HAKAMA_INDICES = np.array([CHAR_TO_IDX[c] for c in HAKAMA_FOLD_RAMP], dtype=np.int32)

# Palette accent colors
AMBER_EYE_BASE = np.array([193, 138, 75], dtype=np.float32)    # #C18A4B muted amber
AMBER_EYE_PEAK = np.array([242, 174, 78], dtype=np.float32)    # #F2AE4E glowing amber
COLLAR_CRIMSON = np.array([204, 76, 68], dtype=np.float32)     # #CC4C44 monk crimson-rust collar trim
DUSTY_ROSE = np.array([186, 128, 128], dtype=np.float32)       # #BA8080 subtle terracotta-rose rock moss
EMBER_DIM = np.array([186, 98, 52], dtype=np.float32)          # #BA6234
EMBER_MID = np.array([228, 128, 56], dtype=np.float32)         # #E48038
EMBER_BRIGHT = np.array([254, 168, 76], dtype=np.float32)      # #FEA84C
ASH_COLOR = np.array([150, 160, 164], dtype=np.float32)        # Cool misty ash
CAVITY_DARK = np.array([18, 21, 25], dtype=np.float32)         # #121519 deep mouth/nostril shadow
MONK_OUTLINE_DARK = np.array([22, 25, 30], dtype=np.float32)   # #16191E crisp monk ink outline
HAKAMA_DARK = np.array([30, 34, 40], dtype=np.float32)         # #1E2228 deep hakama shadow
HAKAMA_LIGHT = np.array([118, 126, 134], dtype=np.float32)     # #767E86 hakama fold highlight

ANCHOR_SWATCHES = [
    (255, 252, 246),  # Brightest ivory
    (244, 240, 233),  # #F4F0E9 Ivory white
    (228, 229, 227),  # #E4E5E3 Silver white
    (200, 204, 205),  # #C8CCCD Cool light silver
    (146, 153, 155),  # #92999B Misty grey
    (127, 137, 139),  # #7F898B Cool slate mist
    (98, 107, 112),   # #626B70 Cool grey shadow
    (72, 80, 86),     # #485056 Mid rock plane
    (48, 56, 61),     # #30383D Charcoal
    (21, 25, 29),     # #15191D Near black
    (193, 138, 75),   # #C18A4B Muted amber
    (242, 174, 78),   # #F2AE4E Eye glow peak
    (86, 46, 28),     # Deep ember halo base
    (134, 68, 36),    # Mid ember halo
    (186, 98, 52),    # #BA6234 Burnt orange ember
    (228, 128, 56),   # #E48038 Warm ember
    (254, 168, 76),   # #FEA84C Bright ember
    (204, 76, 68),    # #CC4C44 Monk collar trim
    (146, 100, 102),  # Shadow dusty-rose moss
    (186, 128, 128),  # #BA8080 Dusty terracotta moss
]


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(font_path, size)
    try:
        font.set_variation_by_axes([600])  # SemiBold
    except Exception:
        try:
            font.set_variation_by_name("SemiBold")
        except Exception:
            pass
    return font


def build_glyph_atlas(font: ImageFont.FreeTypeFont) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-rasterize all glyphs into float32 (K_all, CELL_H, CELL_W) in [0, 1],
    and compute strictly disjoint 6D sub-cell (3x2) shape vectors for EDGE_CHARS.
    """
    atlas = np.zeros((len(ALL_CHARS), CELL_H, CELL_W), dtype=np.float32)
    for i, ch in enumerate(ALL_CHARS):
        im = Image.new("L", (CELL_W, CELL_H), 0)
        ImageDraw.Draw(im).text((0, Y_OFFSET), ch, fill=255, font=font)
        atlas[i] = np.asarray(im, dtype=np.float32) / 255.0

    g_vecs = np.zeros((len(EDGE_CHARS), 6), dtype=np.float32)
    row_slices = [(0, 3), (3, 6), (6, 9)]
    col_slices = [(0, 2), (3, 5)]
    idx = 0
    for r0, r1 in row_slices:
        for c0, c1 in col_slices:
            g_vecs[:, idx] = atlas[EDGE_INDICES, r0:r1, c0:c1].mean(axis=(1, 2))
            idx += 1

    max_cov = float(np.percentile(g_vecs.max(axis=1), 98))
    g_norm = np.clip(g_vecs / max(max_cov, 1e-5), 0.0, 1.0)
    g_norm[0] = 0.0
    return atlas, g_norm


def create_motion_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth spatial masks in source image space:
    - breath_mask: dragon head, snout & serpentine coils
    - wind_mask: dragon mane/horns, delicate whiskers, and monk trailing hair tip
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = yy / h
    xn = xx / w

    monk_dist = ((xn - 0.50) / 0.115) ** 2 + ((yn - 0.685) / 0.09) ** 2
    monk_lock = np.exp(-monk_dist * 2.2)
    rock_lock = np.clip((yn - 0.70) / 0.10, 0.0, 1.0)
    static_lock = np.clip(np.maximum(monk_lock, rock_lock), 0.0, 1.0)

    dragon_body = np.exp(-(((xn - 0.50) / 0.40) ** 2 + ((yn - 0.38) / 0.26) ** 2))
    breath_mask = dragon_body * (1.0 - static_lock)

    mane_mask = np.clip((0.32 - yn) / 0.20, 0.0, 1.0)
    whisker_mask = np.exp(-(((xn - 0.50) / 0.28) ** 2 + ((yn - 0.44) / 0.09) ** 2))
    hair_tail_mask = np.exp(-(((xn - 0.59) / 0.09) ** 2 + ((yn - 0.66) / 0.055) ** 2))
    wind_mask = np.clip(mane_mask * 0.95 + whisker_mask * 0.85 + hair_tail_mask * 0.45, 0.0, 1.0) * (1.0 - rock_lock)

    return breath_mask, wind_mask


def generate_embers(n_embers: int = 28, seed: int = 42) -> list[dict]:
    """Generate seeded burnt-orange ember particles rising in the lower foreground."""
    rng = np.random.default_rng(seed)
    choices = ["*", "+", "•", "·", "°"]
    embers = []
    for i in range(n_embers):
        embers.append(
            {
                "x0": float(rng.uniform(0.05, 0.95)),
                "y0": float(rng.uniform(0.67, 0.98)),
                "rise": float(rng.uniform(0.07, 0.17)),
                "sway_amp": float(rng.uniform(0.008, 0.024)),
                "sway_phase": float(rng.uniform(0, 2 * math.pi)),
                "phase": float(rng.uniform(0, 1)),
                "char": choices[i % len(choices)],
                "bright": float(rng.uniform(0.65, 1.0)),
            }
        )
    return embers


def generate_ash_specks(n_ash: int = 18, seed: int = 99) -> list[dict]:
    """Generate subtle drifting ash specks in the upper and mid mist."""
    rng = np.random.default_rng(seed)
    choices = [".", ",", "·", "'"]
    ash = []
    for i in range(n_ash):
        x0 = float(rng.uniform(0.04, 0.28) if i % 2 == 0 else rng.uniform(0.72, 0.96))
        ash.append(
            {
                "x0": x0,
                "y0": float(rng.uniform(0.12, 0.66)),
                "drift_x": float(rng.uniform(-0.028, 0.028)),
                "drift_y": float(rng.uniform(-0.045, 0.020)),
                "phase": float(rng.uniform(0, 1)),
                "char": choices[i % len(choices)],
            }
        )
    return ash


def build_feature_masks(rgb_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Surgical source masks for:
    - eye_map_src: dragon's left & right glowing amber eyes (slanted horizontal shape)
    - collar_map_src: monk's crimson-rust collar & wrist trim (x=274..330, y=655..730)
    """
    h, w, _ = rgb_f32.shape
    r, g, b = rgb_f32[:, :, 0], rgb_f32[:, :, 1], rgb_f32[:, :, 2]
    lum_raw = 0.299 * r + 0.587 * g + 0.114 * b
    warm = np.clip((r - b - 0.065) / 0.14, 0.0, 1.0)

    yy, xx = np.mgrid[0:h, 0:w]
    left_box = (xx >= 250) & (xx <= 263) & (yy >= 303) & (yy <= 311)
    right_box = (xx >= 345) & (xx <= 358) & (yy >= 303) & (yy <= 311)
    eye_raw = warm * (left_box | right_box).astype(np.float32)
    eye_map = cv2.dilate(eye_raw, np.ones((3, 3), np.uint8))

    monk_box = (xx >= 274) & (xx <= 330) & (yy >= 655) & (yy <= 730)
    collar_warm = np.clip((0.55 * (r - g) + 0.45 * (r - b) - 0.048) / 0.095, 0.0, 1.0) * (lum_raw < 0.72)
    collar_raw = collar_warm * monk_box.astype(np.float32)
    collar_map = cv2.dilate(collar_raw, np.ones((3, 3), np.uint8))
    return eye_map, collar_map


def main() -> None:
    if not os.path.exists(REF_IMAGE_PATH):
        raise FileNotFoundError(f"Reference image not found: {REF_IMAGE_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_GIF_PATH), exist_ok=True)

    bgr = cv2.imread(REF_IMAGE_PATH)
    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_f32 = rgb_u8.astype(np.float32) / 255.0
    h, w, _ = rgb_u8.shape

    # Perceptual luminance + strong ridge/crevice preservation
    lum = 0.299 * rgb_f32[:, :, 0] + 0.587 * rgb_f32[:, :, 1] + 0.114 * rgb_f32[:, :, 2]
    lum_blur = cv2.GaussianBlur(lum, (0, 0), 2.2)
    bright_ridge = np.clip(lum - lum_blur, 0.0, 1.0)
    dark_crevice = np.clip(lum_blur - lum, 0.0, 1.0)
    lum_sharp = np.clip(lum + 0.85 * bright_ridge - 0.80 * dark_crevice, 0.0, 1.0)

    # Richer local CLAHE + DoG relief for sculpted stone shelves & boulders (calm bottom mist avoids horizontal striping)
    yy_src, xx_src = np.mgrid[0:h, 0:w].astype(np.float32)
    yn_src = yy_src / h
    xn_src = xx_src / w
    bot_src_fade = np.clip((yn_src - 0.87) / 0.08, 0.0, 1.0)

    lum_u8 = np.clip(lum_sharp * 255.0, 0, 255).astype(np.uint8)
    clahe_rock = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    lum_clahe = clahe_rock.apply(lum_u8).astype(np.float32) / 255.0
    rock_dog = lum - cv2.GaussianBlur(lum, (0, 0), 6.0)
    clahe_w = 0.45 * (1.0 - 0.72 * bot_src_fade)
    dog_w = 0.35 * (1.0 - 0.75 * bot_src_fade)
    lum_rock_src = np.clip(clahe_w * lum_clahe + (1.0 - clahe_w) * lum_sharp + dog_w * rock_dog, 0.0, 1.0)

    ridge_mask_src = np.clip((bright_ridge - 0.10) * 5.0, 0.0, 1.0)

    eye_map_src, collar_map_src = build_feature_masks(rgb_f32)
    breath_mask, wind_mask = create_motion_masks(h, w)
    font = load_font(FONT_PATH, FONT_SIZE)
    atlas, g_edge = build_glyph_atlas(font)
    g_edge_sq = np.sum(g_edge ** 2, axis=1)

    embers = generate_embers()
    ash_specks = generate_ash_specks()

    gy, gx = np.mgrid[0:ROWS, 0:COLS].astype(np.float32)
    gyn = gy / ROWS
    gxn = gx / COLS

    # Ordered 4x4 Bayer micro-dither matrix (eliminates repetitive horizontal/diagonal banding in coils)
    bayer4 = (
        np.array(
            [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ],
            dtype=np.float32,
        )
        / 15.0
        - 0.5
    ) * 0.068
    bayer_grid = np.tile(bayer4, (ROWS // 4 + 1, COLS // 4 + 1))[:ROWS, :COLS]

    edge_side = np.clip(np.abs(gxn - 0.5) * 1.35, 0.0, 1.0)
    mist_zone = np.exp(-((gyn - 0.42) / 0.34) ** 2) * (0.55 + 0.45 * edge_side)
    monk_clear = np.exp(-(((gxn - 0.50) / 0.12) ** 2 + ((gyn - 0.685) / 0.095) ** 2))
    face_clear = np.exp(-(((gxn - 0.50) / 0.145) ** 2 + ((gyn - 0.32) / 0.12) ** 2))
    fog_strength = mist_zone * (1.0 - 0.94 * monk_clear) * (1.0 - 0.85 * face_clear)

    # Surgical monk region mask (x in [0.35, 0.67], y in [0.605, 0.772]) & seated protection
    monk_x_in = np.clip((gxn - 0.34) / 0.02, 0.0, 1.0) * np.clip((0.68 - gxn) / 0.02, 0.0, 1.0)
    monk_y_in = np.clip((gyn - 0.60) / 0.015, 0.0, 1.0) * np.clip((0.775 - gyn) / 0.015, 0.0, 1.0)
    monk_region_mask = monk_x_in * monk_y_in
    monk_bool = (gxn >= 0.35) & (gxn <= 0.67) & (gyn >= 0.605) & (gyn <= 0.772)

    monk_prot_x = np.clip((gxn - 0.38) / 0.025, 0.0, 1.0) * np.clip((0.62 - gxn) / 0.025, 0.0, 1.0)
    monk_prot_y = np.clip((gyn - 0.60) / 0.025, 0.0, 1.0) * np.clip((0.766 - gyn) / 0.025, 0.0, 1.0)
    monk_protect = monk_prot_x * monk_prot_y

    prev_edge_pos = None
    prev_l_dith = None
    prev_tonal_flat = None
    prev_is_edge = None
    frames_rgb: list[Image.Image] = []

    frame_indices = [-1] + list(range(N_FRAMES))
    for step_idx in frame_indices:
        f_idx = 0 if step_idx == -1 else step_idx
        t = f_idx / N_FRAMES
        theta = 2.0 * math.pi * t

        dy = (2.3 * math.sin(theta)) * breath_mask
        dx = (
            2.4 * np.sin(theta + yn_src * 7.8) * wind_mask
            + 0.8 * math.cos(theta) * breath_mask * (xn_src - 0.5)
        )

        map_x = np.clip(xx_src - dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(yy_src - dy, 0, h - 1).astype(np.float32)

        warped_rgb = cv2.remap(rgb_f32, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_lum = cv2.remap(lum_sharp, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_lum_rock = cv2.remap(lum_rock_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_ridge = cv2.remap(ridge_mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_eye = cv2.remap(eye_map_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_collar = cv2.remap(collar_map_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        rgb_grid = cv2.resize(warped_rgb, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        rgb_max = cv2.resize(cv2.dilate(warped_rgb, np.ones((3, 3), np.uint8)), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        l_area = cv2.resize(warped_lum, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        l_area_rock = cv2.resize(warped_lum_rock, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        l_max = cv2.resize(cv2.dilate(warped_lum, np.ones((3, 3), np.uint8)), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        l_min = cv2.resize(cv2.erode(warped_lum, np.ones((3, 3), np.uint8)), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        ridge_grid = cv2.resize(cv2.dilate(warped_ridge, np.ones((3, 3), np.uint8)), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        eye_grid = cv2.resize(warped_eye, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        collar_grid = cv2.resize(warped_collar, (COLS, ROWS), interpolation=cv2.INTER_AREA)

        # Boost thin bright ridges & dark crevices, keeping monk sleeves immune to erode blackening
        ridge_boost = np.clip(ridge_grid * 0.72, 0.0, 0.72)
        crevice_boost = np.clip((l_area - l_min - 0.08) * 2.8, 0.0, 0.55) * (1.0 - monk_region_mask)
        l_grid = np.clip(l_area + ridge_boost * (l_max - l_area) - crevice_boost * (l_area - l_min), 0.0, 1.0)

        warped_lum_sub = np.clip(warped_lum + 0.45 * warped_ridge, 0.0, 1.0)
        lum_sub = cv2.resize(warped_lum_sub, (COLS * 2, ROWS * 3), interpolation=cv2.INTER_AREA)
        s_cells = lum_sub.reshape(ROWS, 3, COLS, 2).transpose(0, 2, 1, 3).reshape(ROWS, COLS, 6)

        # Sub-cell rock ridges preserving bold dark boulder shadows (start above 0.08, scale by 0.52)
        lum_sub_rock = cv2.resize(warped_lum_rock, (COLS * 2, ROWS * 3), interpolation=cv2.INTER_AREA)
        s_cells_rock = lum_sub_rock.reshape(ROWS, 3, COLS, 2).transpose(0, 2, 1, 3).reshape(ROWS, COLS, 6)
        s_rock_cells = np.clip((s_cells_rock - 0.08) / 0.50, 0.0, 1.0)

        # Mystical drifting fog
        fog_w1 = np.sin(gxn * 2.0 * math.pi * 1.0 - theta + gyn * 3.0)
        fog_w2 = np.cos(gxn * 2.0 * math.pi * 2.0 + theta - gyn * 4.5)
        fog_field = 0.5 + 0.5 * (0.6 * fog_w1 + 0.4 * fog_w2)
        fog_delta = (fog_field - 0.5) * 0.18 * fog_strength

        l_anim = np.clip(l_grid + fog_delta, 0.0, 1.0)
        s_anim = np.clip(s_cells + fog_delta[:, :, None], 0.0, 1.0)

        # Organic horizon-aware rock blend (zero horizontal knife-edge seam across mist)
        rock_y_soft = np.clip((gyn - 0.71) / 0.065, 0.0, 1.0)
        rock_y_deep = np.clip((gyn - 0.765) / 0.025, 0.0, 1.0)
        rock_dark_gate = np.clip((0.66 - l_area) / 0.24, 0.0, 1.0)
        rock_blend = np.maximum(rock_y_deep, rock_y_soft * rock_dark_gate) * (1.0 - monk_protect)
        rock_blend = cv2.GaussianBlur(rock_blend.astype(np.float32), (3, 3), 0.8) * (1.0 - monk_protect)

        # Valley mist drifting gently between foreground rocks (boulder cores stay deep & sculpted)
        valley_mist_mask = (gyn > 0.78) & (l_area > 0.145) & (rock_blend > 0.45)
        l_rock_base = np.clip((l_area_rock - 0.105) / 0.355, 0.0, 1.0) ** 0.72
        l_rock = np.clip(l_rock_base + (fog_field - 0.5) * 0.14 * valley_mist_mask, 0.0, 1.0)

        # Smooth rock gradient magnitude so boulders, ledge slabs & spires get directional edges
        rock_smooth = cv2.GaussianBlur(l_rock_base, (3, 3), 0.7)
        rx_g = cv2.Sobel(rock_smooth, cv2.CV_32F, 1, 0, ksize=3)
        ry_g = cv2.Sobel(rock_smooth, cv2.CV_32F, 0, 1, ksize=3)
        rock_grad_mag = np.sqrt(rx_g ** 2 + ry_g ** 2)

        s_rock_anim = np.clip(s_rock_cells + ((fog_field - 0.5) * 0.14 * valley_mist_mask)[:, :, None], 0.0, 1.0)
        s_anim_final = s_anim * (1.0 - rock_blend[:, :, None]) + s_rock_anim * rock_blend[:, :, None]

        s_span = np.ptp(s_anim_final, axis=2)
        local_blur = cv2.GaussianBlur(l_anim, (5, 5), 1.1)
        texture = np.abs(l_anim - local_blur)

        # 1) Directional edge matching on dragon, calligraphic monk outline/hair/reeds & structural rock ridges
        s_mean = s_anim_final.mean(axis=2, keepdims=True)
        s_boosted = np.clip(s_mean + 1.50 * (s_anim_final - s_mean), 0.0, 1.0)

        l_env = cv2.GaussianBlur(l_anim, (9, 9), 2.3)
        collar_mask = collar_grid > 0.16

        # Calming field for genuinely flat pale mist around the monk silhouette (zero rectangular box exclusion!)
        lower_mist_y = np.clip((gyn - 0.57) / 0.07, 0.0, 1.0) * (1.0 - rock_blend)
        mist_calm = (
            np.clip((0.020 - texture) / 0.012, 0.0, 1.0)
            * np.clip((0.050 - s_span) / 0.032, 0.0, 1.0)
            * np.clip((0.055 - (l_max - l_min)) / 0.035, 0.0, 1.0)
            * lower_mist_y
        )

        # True undarkened luminance inside monk so white robe/lapels stay luminous while outlines remain thin & crisp
        l_monk_raw = 0.299 * rgb_grid[:, :, 0] + 0.587 * rgb_grid[:, :, 1] + 0.114 * rgb_grid[:, :, 2]
        hakama_mask = (gxn >= 0.39) & (gxn <= 0.61) & (gyn >= 0.712) & (gyn <= 0.770) & (l_monk_raw < 0.38) & (~collar_mask)
        monk_ink_line = (
            monk_bool
            & (~collar_mask)
            & (~hakama_mask)
            & (
                ((l_monk_raw < 0.35) & (l_env > 0.48))
                | ((l_min < 0.22) & ((l_max - l_min) > 0.42) & (l_monk_raw < 0.46))
            )
        )
        monk_white_cloth = (
            monk_bool
            & (~monk_ink_line)
            & (~hakama_mask)
            & (~collar_mask)
            & (l_monk_raw >= 0.44)
            & ((mist_calm < 0.45) | (l_monk_raw >= 0.87))
        )
        grass_stalk = (
            (gyn >= 0.665) & (gyn <= 0.785)
            & ((gxn < 0.38) | (gxn > 0.64))
            & ((l_env - l_min) > 0.22)
            & (l_min < 0.40)
            & (l_area > 0.38)
        )

        dark_cavity_dragon = (~monk_bool) & (rock_blend < 0.4) & (l_anim < (l_env - 0.14)) & (l_anim < 0.40) & (gyn < 0.72)
        dark_cavity = dark_cavity_dragon | monk_ink_line | grass_stalk

        s_target = np.where(dark_cavity[:, :, None], 1.0 - s_boosted, s_boosted)
        s_flat = s_target.reshape(-1, 6)
        s_sq = np.sum(s_flat ** 2, axis=1, keepdims=True)
        dists = s_sq + g_edge_sq[None, :] - 2.0 * (s_flat @ g_edge.T)
        # Keep rock facets crisp geometric stone ridges (no parentheses clutter on boulders)
        rock_flat_mask = (rock_blend > 0.4).reshape(-1)
        dists[np.ix_(rock_flat_mask, ROCK_EDGE_EXCLUDE)] += 1e6

        best_edge_pos = np.argmin(dists, axis=1)
        if prev_edge_pos is not None:
            best_err = dists[np.arange(len(best_edge_pos)), best_edge_pos]
            prev_err = dists[np.arange(len(best_edge_pos)), prev_edge_pos]
            edge_pos = np.where((best_err + HYSTERESIS_MARGIN) < prev_err, best_edge_pos, prev_edge_pos)
        else:
            edge_pos = best_edge_pos
        prev_edge_pos = edge_pos
        edge_char_grid = EDGE_INDICES[edge_pos].reshape(ROWS, COLS)

        # 2) Clean figure-ground separation: calm airy lower mist wraps cleanly around monk silhouette
        l_curved = np.clip((l_anim - 0.08) / 0.86, 0.0, 1.0) ** 1.45
        l_monk_clean = np.clip((l_monk_raw - 0.14) / 0.70, 0.0, 1.0) ** 0.70
        monk_cloth_blend = monk_white_cloth.astype(np.float32) * 0.90 + monk_clear * 0.45 * (1.0 - mist_calm)
        monk_cloth_blend = np.clip(monk_cloth_blend, 0.0, 0.92)
        l_upper = l_curved * (1.0 - monk_cloth_blend) + l_monk_clean * monk_cloth_blend

        l_upper_effective = l_upper * (1.0 - 0.42 * mist_calm * (1.0 - monk_white_cloth.astype(np.float32)))

        l_dithered = np.clip(l_upper_effective + bayer_grid, 0.0, 1.0)
        tonal_pos = np.clip((l_dithered * (len(TONAL_RAMP) - 1)).astype(np.int32), 0, len(TONAL_RAMP) - 1)
        tonal_char_grid = TONAL_INDICES[tonal_pos]

        # Dragon scales texture
        scale_mask = (texture > 0.038) & (l_anim > 0.44) & (l_anim < 0.91) & (s_span <= 0.19) & (~monk_bool) & (rock_blend < 0.4)
        scale_pos = np.clip(((l_dithered - 0.22) / 0.72 * (len(SCALE_RAMP) - 1)).astype(np.int32), 0, len(SCALE_RAMP) - 1)
        tonal_char_grid = np.where(scale_mask, SCALE_INDICES[scale_pos], tonal_char_grid)

        # Sculpted dark cloth folds inside monk's black hakama (x=0.39..0.61, y=0.712..0.770)
        hakama_val = np.clip((l_monk_raw - 0.05) / 0.32, 0.0, 1.0)
        hakama_pos = np.clip((hakama_val * (len(HAKAMA_FOLD_RAMP) - 1)).astype(np.int32), 0, len(HAKAMA_FOLD_RAMP) - 1)
        tonal_char_grid = np.where(hakama_mask, HAKAMA_INDICES[hakama_pos], tonal_char_grid)

        # Sculptural rocky foreground (deep boulders dark & faceted, mist & sunlit stone ledge detailed)
        rock_mask = rock_blend > 0.4
        l_rock_dith = np.clip(l_rock + bayer_grid, 0.0, 1.0)
        rock_pos = np.clip((l_rock_dith * (len(ROCK_RAMP) - 1)).astype(np.int32), 0, len(ROCK_RAMP) - 1)
        tonal_char_grid = np.where(rock_mask, ROCK_INDICES[rock_pos], tonal_char_grid)

        # Quiet sparse mist over smooth sky/background areas
        smooth_mist = (
            (texture < 0.028)
            & (s_span <= 0.12)
            & (l_anim >= 0.24)
            & (l_anim <= 0.63)
            & (ridge_grid < 0.14)
            & (~monk_bool)
            & (rock_blend < 0.25)
        )
        mist_val = np.clip((l_dithered - 0.10) / 0.36, 0.0, 1.0)
        mist_pos = np.clip((mist_val * (len(MIST_RAMP) - 1)).astype(np.int32), 0, len(MIST_RAMP) - 1)
        tonal_char_grid = np.where(smooth_mist, MIST_INDICES[mist_pos], tonal_char_grid)

        l_hyst_src = l_dithered * (1.0 - rock_blend) + l_rock_dith * rock_blend
        tonal_flat = tonal_char_grid.reshape(-1)
        l_flat = l_hyst_src.reshape(-1)
        if prev_tonal_flat is not None:
            tonal_keep = np.abs(l_flat - prev_l_dith) < 0.022
            tonal_flat = np.where(tonal_keep, prev_tonal_flat, tonal_flat)
            l_flat = np.where(tonal_keep, prev_l_dith, l_flat)
        prev_tonal_flat = tonal_flat
        prev_l_dith = l_flat

        is_edge_dragon = (s_span > 0.185) | (ridge_grid > 0.24) | dark_cavity
        is_edge_cloth = s_span > 0.30
        is_edge_hakama = s_span > 0.22
        rock_grad_gate = np.where(gyn > 0.88, 0.28, 0.21)
        is_edge_rock = (s_span > 0.145) & (rock_grad_mag > rock_grad_gate)
        is_edge_upper = np.where(monk_white_cloth, is_edge_cloth, np.where(hakama_mask, is_edge_hakama, is_edge_dragon)) | collar_mask
        is_edge_now = np.where(rock_mask, is_edge_rock, is_edge_upper).reshape(-1)
        if prev_is_edge is not None:
            edge_thresh_grid = np.where(
                rock_mask, 0.145, np.where(monk_white_cloth, 0.30, np.where(hakama_mask, 0.22, 0.185))
            )
            is_edge_final = (
                np.where(np.abs(s_span - edge_thresh_grid).reshape(-1) < 0.02, prev_is_edge, is_edge_now)
                | dark_cavity.reshape(-1)
                | collar_mask.reshape(-1)
                | ((ridge_grid.reshape(-1) > 0.24) & (~monk_white_cloth.reshape(-1)) & (~rock_mask.reshape(-1)))
            )
        else:
            is_edge_final = is_edge_now
        prev_is_edge = is_edge_final

        char_idx_grid = np.where(is_edge_final.reshape(ROWS, COLS), edge_char_grid, tonal_flat.reshape(ROWS, COLS)).copy()

        # Backdrop glow: tighter local monk backdrop via smooth radial monk_clear (zero rectangular seams)
        ambient_upper = cv2.GaussianBlur(rgb_grid, (0, 0), 8.0)
        ambient_monk_local = cv2.GaussianBlur(rgb_grid, (0, 0), 1.4)
        ambient_upper_mix = (
            ambient_upper * (1.0 - 0.85 * monk_clear[:, :, None])
            + ambient_monk_local * (0.85 * monk_clear[:, :, None])
        )
        ambient_rock = cv2.GaussianBlur(rgb_grid, (0, 0), 1.8)
        ambient_grid = ambient_upper_mix * (1.0 - rock_blend[:, :, None]) + ambient_rock * rock_blend[:, :, None]

        charcoal = np.array([18.0, 21.0, 25.0], dtype=np.float32) / 255.0
        ambient_lum = 0.299 * ambient_grid[:, :, 0] + 0.587 * ambient_grid[:, :, 1] + 0.114 * ambient_grid[:, :, 2]
        cloth_lift = 0.12 * monk_clear * (1.0 - 0.85 * mist_calm) * np.clip((ambient_lum - 0.45) / 0.35, 0.0, 1.0)
        bg_weight_upper = np.clip(0.08 + 0.56 * (ambient_lum ** 1.55) + cloth_lift, 0.08, 0.76)
        bg_weight_rock = np.clip(0.04 + 0.58 * (l_rock ** 1.0), 0.04, 0.62)

        bg_weight = (bg_weight_upper * (1.0 - rock_blend) + bg_weight_rock * rock_blend)[:, :, None]
        bg_grid = charcoal * (1.0 - bg_weight) + ambient_grid * bg_weight
        bg_canvas = cv2.resize(
            (bg_grid * 255.0).astype(np.float32),
            (COLS * CELL_W, ROWS * CELL_H),
            interpolation=cv2.INTER_LINEAR,
        )

        # Upper/monk foreground colors
        rgb_src_col = rgb_grid * (1.0 - ridge_boost[:, :, None]) + rgb_max * ridge_boost[:, :, None]
        rgb_255 = rgb_src_col * 255.0
        rgb_lum_255 = (0.299 * rgb_255[:, :, 0] + 0.587 * rgb_255[:, :, 1] + 0.114 * rgb_255[:, :, 2])[:, :, None]
        rgb_chroma = rgb_lum_255 + 1.28 * (rgb_255 - rgb_lum_255)
        fg_upper = np.clip(rgb_chroma * 1.22 + 12.0, 0.0, 255.0)

        # Sculptural rock color grading: non-crushed boulders + valley fog + authentic dusty-rose terracotta moss
        rgb_255_rock = rgb_grid * 255.0
        rock_gain = (1.38 + 1.08 * l_rock)[:, :, None]
        rock_base_rgb = np.clip(rgb_255_rock * rock_gain + (18.0 + 38.0 * l_rock)[:, :, None], 36.0, 238.0)
        rose_score = 0.70 * (rgb_grid[:, :, 0] - rgb_grid[:, :, 1]) + 0.30 * (rgb_grid[:, :, 0] - rgb_grid[:, :, 2])
        rose_w = (
            np.clip((rose_score - 0.020) / 0.042, 0.0, 1.0)
            * np.clip((gyn - 0.73) / 0.03, 0.0, 1.0)
            * np.clip((0.91 - gyn) / 0.04, 0.0, 1.0)
        )
        rose_target = DUSTY_ROSE[None, None, :] * (0.62 + 0.42 * l_rock)[:, :, None]
        fg_rock = np.clip(
            rock_base_rgb * (1.0 - 0.58 * rose_w[:, :, None]) + rose_target * (0.58 * rose_w[:, :, None]),
            36.0,
            238.0,
        )

        fg_rgb_grid = fg_upper * (1.0 - rock_blend[:, :, None]) + fg_rock * rock_blend[:, :, None]

        # Crisp ivory monk robe/hair foreground highlight
        if np.any(monk_white_cloth):
            fg_rgb_grid[monk_white_cloth] = np.clip(fg_rgb_grid[monk_white_cloth] * 1.14 + 22.0, 0.0, 255.0)

        # Sculpted hakama dark fold colors (#1E2228 to #767E86)
        if np.any(hakama_mask):
            hv = hakama_val[hakama_mask][:, None]
            fg_rgb_grid[hakama_mask] = HAKAMA_DARK * (1.0 - hv) + HAKAMA_LIGHT * hv

        # Dragon cavities, crisp monk ink outlines & horizon reeds
        fg_rgb_grid[dark_cavity_dragon] = CAVITY_DARK
        fg_rgb_grid[monk_ink_line] = MONK_OUTLINE_DARK
        fg_rgb_grid[grass_stalk] = np.array([34.0, 40.0, 46.0], dtype=np.float32)

        # Monk crimson-rust kimono V-collar & wrist trim
        if np.any(collar_mask):
            w_col = np.clip(collar_grid[collar_mask] * 1.5, 0.45, 0.95)[:, None]
            fg_rgb_grid[collar_mask] = fg_rgb_grid[collar_mask] * (1.0 - w_col) + COLLAR_CRIMSON * w_col

        # Subtle mist brightness drift (upper fog + lower valley mist)
        mist_tint = (fog_field - 0.5) * 22.0 * np.maximum(fog_strength, valley_mist_mask.astype(np.float32) * 0.65)
        fg_rgb_grid = np.clip(fg_rgb_grid + mist_tint[:, :, None], 0.0, 255.0)

        # Glowing amber dragon eyes + tight slanted halo
        eye_pulse = 0.5 - 0.5 * math.cos(theta)
        eye_color = AMBER_EYE_BASE * (1.0 - eye_pulse) + AMBER_EYE_PEAK * eye_pulse
        eye_core = eye_grid > 0.30
        eye_center = eye_grid > 0.56

        eye_halo = cv2.GaussianBlur(eye_grid, (5, 5), 1.0)
        halo_mask = (eye_halo > 0.08) & (~eye_core)
        if np.any(halo_mask):
            w_halo = np.clip(eye_halo[halo_mask] * (0.40 + 0.25 * eye_pulse), 0.0, 0.38)[:, None]
            fg_rgb_grid[halo_mask] = fg_rgb_grid[halo_mask] * (1.0 - w_halo) + eye_color * w_halo

        if np.any(eye_core):
            fg_rgb_grid[eye_core] = eye_color
            char_idx_grid[eye_core] = CHAR_TO_IDX["◉"]
        if np.any(eye_center):
            char_idx_grid[eye_center] = CHAR_TO_IDX["◈"]

        alpha_grid = atlas[char_idx_grid]
        alpha_full = alpha_grid.transpose(0, 2, 1, 3).reshape(ROWS * CELL_H, COLS * CELL_W, 1)

        fg_full = np.repeat(np.repeat(fg_rgb_grid, CELL_H, axis=0), CELL_W, axis=1)
        canvas = bg_canvas * (1.0 - alpha_full) + fg_full * alpha_full

        # Tight slanted horizontal elliptical glow around the dragon's eyes
        if np.any(eye_core):
            er_list, ec_list = np.where(eye_center if np.any(eye_center) else eye_core)
            for er_i, ec_i in zip(er_list, ec_list):
                cy_px, cx_px = er_i * CELL_H + CELL_H // 2, ec_i * CELL_W + CELL_W // 2
                y_lo, y_hi = max(0, cy_px - 10), min(ROWS * CELL_H, cy_px + 11)
                x_lo, x_hi = max(0, cx_px - 12), min(COLS * CELL_W, cx_px + 13)
                py, px = np.mgrid[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
                glow = np.exp(-(((py - cy_px) / 2.6) ** 2 + ((px - cx_px) / 4.8) ** 2) * 0.5) * (0.14 + 0.16 * eye_pulse)
                canvas[y_lo:y_hi, x_lo:x_hi] = (
                    canvas[y_lo:y_hi, x_lo:x_hi] * (1.0 - glow[:, :, None]) + eye_color * glow[:, :, None]
                )

        # Drifting ash specks
        for sp in ash_specks:
            life = (sp["phase"] + t) % 1.0
            sx = sp["x0"] + sp["drift_x"] * math.sin(theta)
            sy = sp["y0"] + sp["drift_y"] * math.sin(theta + sp["phase"] * 6.28)
            sr = int(round(sy * (ROWS - 1)))
            sc = int(round(sx * (COLS - 1)))
            if 0 <= sr < ROWS and 0 <= sc < COLS:
                fade = 0.40 + 0.45 * math.sin(math.pi * life)
                y0, y1 = sr * CELL_H, (sr + 1) * CELL_H
                x0, x1 = sc * CELL_W, (sc + 1) * CELL_W
                a3 = (atlas[CHAR_TO_IDX[sp["char"]]] * fade)[:, :, None]
                canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - a3) + ASH_COLOR * a3

        # Smooth circular-halo burnt-orange embers
        for emb in embers:
            life = (emb["phase"] + t) % 1.0
            ey = emb["y0"] - life * emb["rise"]
            ex = emb["x0"] + emb["sway_amp"] * math.sin(theta + emb["sway_phase"])
            er = int(round(ey * (ROWS - 1)))
            ec = int(round(ex * (COLS - 1)))
            if 0 <= er < ROWS and 0 <= ec < COLS:
                fade = math.sin(math.pi * life) * emb["bright"]
                if fade > 0.14:
                    e_col = (
                        EMBER_DIM * (1.0 - fade) + EMBER_MID * fade
                        if fade < 0.7
                        else EMBER_MID * (1.0 - (fade - 0.7) / 0.3) + EMBER_BRIGHT * ((fade - 0.7) / 0.3)
                    )
                    cy_px, cx_px = er * CELL_H + CELL_H // 2, ec * CELL_W + CELL_W // 2
                    y_lo, y_hi = max(0, cy_px - 10), min(ROWS * CELL_H, cy_px + 11)
                    x_lo, x_hi = max(0, cx_px - 10), min(COLS * CELL_W, cx_px + 11)
                    py, px = np.mgrid[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
                    glow = np.exp(-((py - cy_px) ** 2 + (px - cx_px) ** 2) / (2.0 * (3.2 ** 2))) * (0.42 * fade)
                    canvas[y_lo:y_hi, x_lo:x_hi] = (
                        canvas[y_lo:y_hi, x_lo:x_hi] * (1.0 - glow[:, :, None]) + e_col * glow[:, :, None]
                    )

                    y0, y1 = er * CELL_H, (er + 1) * CELL_H
                    x0, x1 = ec * CELL_W, (ec + 1) * CELL_W
                    a3 = (atlas[CHAR_TO_IDX[emb["char"]]] * min(1.0, fade * 1.35))[:, :, None]
                    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - a3) + e_col * a3

        if step_idx >= 0:
            frame_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
            frames_rgb.append(Image.fromarray(frame_u8, mode="RGB"))

    sample_indices = [0, N_FRAMES // 4, N_FRAMES // 2, (3 * N_FRAMES) // 4]
    fw, fh = frames_rgb[0].size
    strip = Image.new("RGB", (fw * 2, fh * 2 + 24))
    strip.paste(frames_rgb[sample_indices[0]], (0, 0))
    strip.paste(frames_rgb[sample_indices[1]], (fw, 0))
    strip.paste(frames_rgb[sample_indices[2]], (0, fh))
    strip.paste(frames_rgb[sample_indices[3]], (fw, fh))
    draw_sw = ImageDraw.Draw(strip)
    sw_w = (fw * 2) // len(ANCHOR_SWATCHES)
    for i_sw, col in enumerate(ANCHOR_SWATCHES):
        draw_sw.rectangle([i_sw * sw_w, fh * 2, (i_sw + 1) * sw_w, fh * 2 + 24], fill=col)

    global_palette = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized_frames = [fr.quantize(palette=global_palette, dither=Image.Dither.NONE) for fr in frames_rgb]

    quantized_frames[0].save(
        OUTPUT_GIF_PATH,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )

    size_kb = os.path.getsize(OUTPUT_GIF_PATH) / 1024.0
    print(f"Saved {OUTPUT_GIF_PATH}: {len(quantized_frames)} frames, {frames_rgb[0].size}, {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")


if __name__ == "__main__":
    main()
