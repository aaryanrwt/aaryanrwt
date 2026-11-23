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

# Grid & rendering parameters (6x10 cells on 152x156 -> 912 x 1560, matches 598x1024 aspect)
COLS = 152
ROWS = 156
CELL_W = 6
CELL_H = 10
FONT_SIZE = 9
Y_OFFSET = 0

N_FRAMES = 30
FRAME_DURATION_MS = 100  # 3.0s seamless loop
HYSTERESIS_MARGIN = 0.016

# Pure crisp ASCII + box-drawing glyphs (zero blurry ░▒▓ boxes, zero random alphabet soup)
EDGE_CHARS = list(" .',:;-~=+*!|/\\()[]{}<>^v_╱╲╭╮╰╯│─#%@")
TONAL_RAMP = list(" .',:;-~+*=%&#@")
SHADOW_RAMP = list("   ..',,:;!/\\")
MIST_RAMP = list("      ..,:;")
SCALE_RAMP = list(":+*=%&#@")

OVERLAY_CHARS = ["◉", "◈", "•", "·", "°"]
ALL_CHARS = []
for ch in EDGE_CHARS + TONAL_RAMP + SHADOW_RAMP + MIST_RAMP + SCALE_RAMP + OVERLAY_CHARS:
    if ch not in ALL_CHARS:
        ALL_CHARS.append(ch)
CHAR_TO_IDX = {ch: i for i, ch in enumerate(ALL_CHARS)}

EDGE_INDICES = np.array([CHAR_TO_IDX[c] for c in EDGE_CHARS], dtype=np.int32)
TONAL_INDICES = np.array([CHAR_TO_IDX[c] for c in TONAL_RAMP], dtype=np.int32)
SHADOW_INDICES = np.array([CHAR_TO_IDX[c] for c in SHADOW_RAMP], dtype=np.int32)
MIST_INDICES = np.array([CHAR_TO_IDX[c] for c in MIST_RAMP], dtype=np.int32)
SCALE_INDICES = np.array([CHAR_TO_IDX[c] for c in SCALE_RAMP], dtype=np.int32)

# Palette accent colors
AMBER_EYE_BASE = np.array([196, 134, 58], dtype=np.float32)    # #C4863A muted amber
AMBER_EYE_PEAK = np.array([255, 206, 98], dtype=np.float32)    # #FFCE62 glowing amber
COLLAR_CRIMSON = np.array([204, 88, 76], dtype=np.float32)     # #CC584C monk kimono trim
EMBER_DIM = np.array([176, 96, 56], dtype=np.float32)          # #B06038
EMBER_MID = np.array([218, 138, 66], dtype=np.float32)         # #DA8A42
EMBER_BRIGHT = np.array([252, 168, 78], dtype=np.float32)      # #FCA84E
ASH_COLOR = np.array([150, 160, 164], dtype=np.float32)        # Cool misty ash
CAVITY_DARK = np.array([18, 21, 25], dtype=np.float32)         # #121519 deep mouth/nostril/hair

ANCHOR_SWATCHES = [
    (255, 252, 246),  # Brightest ivory
    (244, 240, 233),  # #F4F0E9 Ivory white
    (228, 229, 227),  # #E4E5E3 Silver white
    (200, 204, 205),  # #C8CCCD Cool light silver
    (146, 153, 155),  # #92999B Misty grey
    (127, 137, 139),  # #7F898B Cool slate mist
    (98, 107, 112),   # #626B70 Cool grey shadow
    (48, 56, 61),     # #30383D Charcoal
    (21, 25, 29),     # #15191D Near black
    (188, 126, 56),   # #BC7E38 Muted amber
    (232, 168, 76),   # Mid amber
    (255, 200, 94),   # #FFC85E Eye glow peak
    (166, 93, 56),    # #A65D38 Burnt orange
    (245, 162, 76),   # Bright ember
    (176, 86, 74),    # #B0564A Monk collar trim
    (138, 96, 92),    # #8A605C Dusty terracotta rock highlight
]


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(font_path, size)
    try:
        font.set_variation_by_axes([650])  # SemiBold-Bold for luminous ivory glyph coverage
    except Exception:
        try:
            font.set_variation_by_name("SemiBold")
        except Exception:
            pass
    return font


def build_glyph_atlas(font: ImageFont.FreeTypeFont) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-rasterize all glyphs into float32 (K_all, CELL_H, CELL_W) in [0, 1],
    and compute disjoint 6D sub-cell (3x2) shape vectors for EDGE_CHARS.
    """
    atlas = np.zeros((len(ALL_CHARS), CELL_H, CELL_W), dtype=np.float32)
    for i, ch in enumerate(ALL_CHARS):
        im = Image.new("L", (CELL_W, CELL_H), 0)
        ImageDraw.Draw(im).text((0, Y_OFFSET), ch, fill=255, font=font)
        atlas[i] = np.asarray(im, dtype=np.float32) / 255.0

    g_vecs = np.zeros((len(EDGE_CHARS), 6), dtype=np.float32)
    row_slices = [(0, 3), (3, 7), (7, 10)]
    col_slices = [(0, 3), (3, 6)]
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

    monk_dist = ((xn - 0.50) / 0.105) ** 2 + ((yn - 0.69) / 0.085) ** 2
    monk_lock = np.exp(-monk_dist * 2.0)
    rock_lock = np.clip((yn - 0.72) / 0.11, 0.0, 1.0)
    static_lock = np.clip(np.maximum(monk_lock, rock_lock), 0.0, 1.0)

    dragon_body = np.exp(-(((xn - 0.50) / 0.40) ** 2 + ((yn - 0.38) / 0.26) ** 2))
    breath_mask = dragon_body * (1.0 - static_lock)

    mane_mask = np.clip((0.32 - yn) / 0.20, 0.0, 1.0)
    whisker_mask = np.exp(-(((xn - 0.50) / 0.28) ** 2 + ((yn - 0.44) / 0.09) ** 2))
    hair_tail_mask = np.exp(-(((xn - 0.59) / 0.09) ** 2 + ((yn - 0.66) / 0.055) ** 2))
    wind_mask = np.clip(mane_mask * 0.95 + whisker_mask * 0.85 + hair_tail_mask * 0.45, 0.0, 1.0) * (1.0 - rock_lock)

    return breath_mask, wind_mask


def generate_embers(n_embers: int = 38, seed: int = 42) -> list[dict]:
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
    - collar_map_src: monk's crimson-rust collar & wrist trim
    """
    h, w, _ = rgb_f32.shape
    r, b = rgb_f32[:, :, 0], rgb_f32[:, :, 2]
    warm = np.clip((r - b - 0.07) / 0.16, 0.0, 1.0)

    yy, xx = np.mgrid[0:h, 0:w]
    left_box = (xx >= 250) & (xx <= 263) & (yy >= 303) & (yy <= 311)
    right_box = (xx >= 345) & (xx <= 358) & (yy >= 303) & (yy <= 311)
    eye_raw = warm * (left_box | right_box).astype(np.float32)
    eye_map = cv2.dilate(eye_raw, np.ones((3, 5), np.uint8))

    monk_box = (xx >= 274) & (xx <= 328) & (yy >= 655) & (yy <= 728)
    collar_raw = warm * monk_box.astype(np.float32)
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

    # Perceptual luminance + strong ridge/crevice preservation for thin whiskers, fangs & monk hair
    lum = 0.299 * rgb_f32[:, :, 0] + 0.587 * rgb_f32[:, :, 1] + 0.114 * rgb_f32[:, :, 2]
    lum_blur = cv2.GaussianBlur(lum, (0, 0), 2.2)
    bright_ridge = np.clip(lum - lum_blur, 0.0, 1.0)
    dark_crevice = np.clip(lum_blur - lum, 0.0, 1.0)
    lum_sharp = np.clip(lum + 0.85 * bright_ridge - 0.80 * dark_crevice, 0.0, 1.0)

    # Also keep a crisp ridge mask so 1-2px bright whiskers/fangs survive grid downsampling!
    ridge_mask_src = np.clip((bright_ridge - 0.10) * 5.0, 0.0, 1.0)

    eye_map_src, collar_map_src = build_feature_masks(rgb_f32)
    breath_mask, wind_mask = create_motion_masks(h, w)
    font = load_font(FONT_PATH, FONT_SIZE)
    atlas, g_edge = build_glyph_atlas(font)
    g_edge_sq = np.sum(g_edge ** 2, axis=1)

    embers = generate_embers()
    ash_specks = generate_ash_specks()

    yy_src, xx_src = np.mgrid[0:h, 0:w].astype(np.float32)
    yn_src = yy_src / h
    xn_src = xx_src / w

    gy, gx = np.mgrid[0:ROWS, 0:COLS].astype(np.float32)
    gyn = gy / ROWS
    gxn = gx / COLS

    # Ordered 2x2 Bayer micro-dither matrix for organic scale/ivory texture
    bayer2 = np.array([[-0.030, 0.020], [0.030, -0.020]], dtype=np.float32)
    bayer_grid = np.tile(bayer2, (ROWS // 2 + 1, COLS // 2 + 1))[:ROWS, :COLS]

    edge_side = np.clip(np.abs(gxn - 0.5) * 1.35, 0.0, 1.0)
    mist_zone = np.exp(-((gyn - 0.42) / 0.34) ** 2) * (0.55 + 0.45 * edge_side)
    monk_clear = np.exp(-(((gxn - 0.50) / 0.115) ** 2 + ((gyn - 0.69) / 0.095) ** 2))
    face_clear = np.exp(-(((gxn - 0.50) / 0.145) ** 2 + ((gyn - 0.32) / 0.12) ** 2))
    fog_strength = mist_zone * (1.0 - 0.92 * monk_clear) * (1.0 - 0.85 * face_clear)

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
        warped_ridge = cv2.remap(ridge_mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_eye = cv2.remap(eye_map_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warped_collar = cv2.remap(collar_map_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        rgb_grid = cv2.resize(warped_rgb, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        rgb_max = cv2.resize(cv2.dilate(warped_rgb, np.ones((3, 3), np.uint8)), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        l_area = cv2.resize(warped_lum, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        l_max = cv2.resize(cv2.dilate(warped_lum, np.ones((3, 3), np.uint8)), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        l_min = cv2.resize(cv2.erode(warped_lum, np.ones((3, 3), np.uint8)), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        ridge_grid = cv2.resize(cv2.dilate(warped_ridge, np.ones((3, 3), np.uint8)), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        eye_grid = cv2.resize(warped_eye, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        collar_grid = cv2.resize(warped_collar, (COLS, ROWS), interpolation=cv2.INTER_AREA)

        # Boost thin bright ridges (whiskers, fangs, horn tips) and thin dark crevices into l_grid AND sub-cells
        ridge_boost = np.clip(ridge_grid * 0.72, 0.0, 0.72)
        crevice_boost = np.clip((l_area - l_min - 0.08) * 2.8, 0.0, 0.55)
        l_grid = np.clip(l_area + ridge_boost * (l_max - l_area) - crevice_boost * (l_area - l_min), 0.0, 1.0)

        # Inject ridge highlights into warped_lum before 3x2 sub-cell sampling so directional shape catches thin whiskers!
        warped_lum_sub = np.clip(warped_lum + 0.45 * warped_ridge, 0.0, 1.0)
        lum_sub = cv2.resize(warped_lum_sub, (COLS * 2, ROWS * 3), interpolation=cv2.INTER_AREA)
        s_cells = lum_sub.reshape(ROWS, 3, COLS, 2).transpose(0, 2, 1, 3).reshape(ROWS, COLS, 6)

        # Mystical drifting fog
        fog_w1 = np.sin(gxn * 2.0 * math.pi * 1.0 - theta + gyn * 3.0)
        fog_w2 = np.cos(gxn * 2.0 * math.pi * 2.0 + theta - gyn * 4.5)
        fog_field = 0.5 + 0.5 * (0.6 * fog_w1 + 0.4 * fog_w2)
        fog_delta = (fog_field - 0.5) * 0.18 * fog_strength

        l_anim = np.clip(l_grid + fog_delta, 0.0, 1.0)
        s_anim = np.clip(s_cells + fog_delta[:, :, None], 0.0, 1.0)

        s_span = np.ptp(s_anim, axis=2)
        local_blur = cv2.GaussianBlur(l_anim, (5, 5), 1.1)
        texture = np.abs(l_anim - local_blur)

        # 1) Directional edge matching on real structural contours & thin whiskers
        s_mean = s_anim.mean(axis=2, keepdims=True)
        s_boosted = np.clip(s_mean + 1.50 * (s_anim - s_mean), 0.0, 1.0)

        l_env = cv2.GaussianBlur(l_anim, (9, 9), 2.3)
        dark_cavity = (l_anim < (l_env - 0.14)) & (l_anim < 0.40) & (gyn < 0.76)

        s_target = np.where(dark_cavity[:, :, None], 1.0 - s_boosted, s_boosted)
        s_flat = s_target.reshape(-1, 6)
        s_sq = np.sum(s_flat ** 2, axis=1, keepdims=True)
        dists = s_sq + g_edge_sq[None, :] - 2.0 * (s_flat @ g_edge.T)

        best_edge_pos = np.argmin(dists, axis=1)
        if prev_edge_pos is not None:
            best_err = dists[np.arange(len(best_edge_pos)), best_edge_pos]
            prev_err = dists[np.arange(len(best_edge_pos)), prev_edge_pos]
            edge_pos = np.where((best_err + HYSTERESIS_MARGIN) < prev_err, best_edge_pos, prev_edge_pos)
        else:
            edge_pos = best_edge_pos
        prev_edge_pos = edge_pos
        edge_char_grid = EDGE_INDICES[edge_pos].reshape(ROWS, COLS)

        # 2) Clean tonal contrast curve: quiet misty sky (~0.30) vs luminous ivory dragon (~0.95)
        l_curved = np.clip((l_anim - 0.08) / 0.86, 0.0, 1.0) ** 1.45
        l_dithered = np.clip(l_curved + bayer_grid, 0.0, 1.0)

        tonal_pos = np.clip((l_dithered * (len(TONAL_RAMP) - 1)).astype(np.int32), 0, len(TONAL_RAMP) - 1)
        tonal_char_grid = TONAL_INDICES[tonal_pos]

        # Dragon scales texture
        scale_mask = (texture > 0.038) & (l_anim > 0.44) & (l_anim < 0.91) & (s_span <= 0.17)
        scale_pos = np.clip(((l_dithered - 0.22) / 0.72 * (len(SCALE_RAMP) - 1)).astype(np.int32), 0, len(SCALE_RAMP) - 1)
        tonal_char_grid = np.where(scale_mask, SCALE_INDICES[scale_pos], tonal_char_grid)

        # Sparse dark rock texture in lower foreground
        shadow_mask = (l_anim < 0.22) & (s_span <= 0.15)
        shadow_pos = np.clip((l_dithered / 0.14 * (len(SHADOW_RAMP) - 1)).astype(np.int32), 0, len(SHADOW_RAMP) - 1)
        tonal_char_grid = np.where(shadow_mask, SHADOW_INDICES[shadow_pos], tonal_char_grid)

        # Quiet sparse mist over ALL smooth sky/background areas so dragon, horns & whiskers pop off!
        smooth_mist = (texture < 0.028) & (s_span <= 0.11) & (l_anim >= 0.24) & (l_anim <= 0.63) & (ridge_grid < 0.14) & (gyn < 0.74)
        mist_val = np.clip((l_dithered - 0.10) / 0.36, 0.0, 1.0)
        mist_pos = np.clip((mist_val * (len(MIST_RAMP) - 1)).astype(np.int32), 0, len(MIST_RAMP) - 1)
        tonal_char_grid = np.where(smooth_mist, MIST_INDICES[mist_pos], tonal_char_grid)

        tonal_flat = tonal_char_grid.reshape(-1)
        l_flat = l_dithered.reshape(-1)
        if prev_tonal_flat is not None:
            tonal_keep = np.abs(l_flat - prev_l_dith) < 0.022
            tonal_flat = np.where(tonal_keep, prev_tonal_flat, tonal_flat)
            l_flat = np.where(tonal_keep, prev_l_dith, l_flat)
        prev_tonal_flat = tonal_flat
        prev_l_dith = l_flat

        is_edge_now = ((s_span > 0.15) | (ridge_grid > 0.24) | dark_cavity).reshape(-1)
        if prev_is_edge is not None:
            is_edge_final = np.where(np.abs(s_span.reshape(-1) - 0.15) < 0.02, prev_is_edge, is_edge_now) | dark_cavity.reshape(-1) | (ridge_grid.reshape(-1) > 0.24)
        else:
            is_edge_final = is_edge_now
        prev_is_edge = is_edge_final

        char_idx_grid = np.where(is_edge_final.reshape(ROWS, COLS), edge_char_grid, tonal_flat.reshape(ROWS, COLS)).copy()

        # True-to-reference low-frequency backdrop glow (sigma = 8 cells for glowing ivory crown/coils vs dark sky/rocks)
        ambient_grid = cv2.GaussianBlur(rgb_grid, (0, 0), 8.5)
        charcoal = np.array([18.0, 22.0, 26.0], dtype=np.float32) / 255.0
        ambient_lum = 0.299 * ambient_grid[:, :, 0] + 0.587 * ambient_grid[:, :, 1] + 0.114 * ambient_grid[:, :, 2]
        # Bright ivory dragon/monk radiate luminous #928E89 core glow; cool sky mist stays dark #2A3236; rocks stay #14181C!
        bg_weight = np.clip(0.08 + 0.54 * (ambient_lum ** 1.65), 0.08, 0.62)[:, :, None]
        bg_grid = charcoal * (1.0 - bg_weight) + ambient_grid * bg_weight
        bg_canvas = cv2.resize(
            (bg_grid * 255.0).astype(np.float32),
            (COLS * CELL_W, ROWS * CELL_H),
            interpolation=cv2.INTER_LINEAR,
        )

        # Blend ridge_max RGB onto bright thin ridges (whiskers, fangs, horn tips) so they stay crisp bright ivory!
        rgb_src_col = rgb_grid * (1.0 - ridge_boost[:, :, None]) + rgb_max * ridge_boost[:, :, None]
        rgb_255 = rgb_src_col * 255.0
        rgb_lum_255 = (0.299 * rgb_255[:, :, 0] + 0.587 * rgb_255[:, :, 1] + 0.114 * rgb_255[:, :, 2])[:, :, None]
        rgb_chroma = rgb_lum_255 + 1.28 * (rgb_255 - rgb_lum_255)
        fg_rgb_grid = np.clip(rgb_chroma * 1.22 + 12.0, 0.0, 255.0)

        fg_rgb_grid[dark_cavity] = CAVITY_DARK

        # Apply subtle crimson-rust trim on monk's kimono collar
        collar_mask = collar_grid > 0.22
        if np.any(collar_mask):
            w_col = np.clip(collar_grid[collar_mask] * 1.15, 0.0, 0.85)[:, None]
            fg_rgb_grid[collar_mask] = fg_rgb_grid[collar_mask] * (1.0 - w_col) + COLLAR_CRIMSON * w_col

        mist_tint = (fog_field - 0.5) * 22.0 * fog_strength
        fg_rgb_grid = np.clip(fg_rgb_grid + mist_tint[:, :, None], 0.0, 255.0)

        # Glowing amber dragon eyes + tight slanted halo
        eye_pulse = 0.5 - 0.5 * math.cos(theta)
        eye_color = AMBER_EYE_BASE * (1.0 - eye_pulse) + AMBER_EYE_PEAK * eye_pulse
        eye_core = eye_grid > 0.24
        eye_center = eye_grid > 0.52

        eye_halo = cv2.GaussianBlur(eye_grid, (5, 5), 1.2)
        halo_mask = (eye_halo > 0.06) & (~eye_core)
        if np.any(halo_mask):
            w_halo = np.clip(eye_halo[halo_mask] * (0.45 + 0.30 * eye_pulse), 0.0, 0.42)[:, None]
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

        # Tight circular radial glow around the dragon's eyes
        if np.any(eye_core):
            er_list, ec_list = np.where(eye_center if np.any(eye_center) else eye_core)
            for er_i, ec_i in zip(er_list, ec_list):
                cy_px, cx_px = er_i * CELL_H + CELL_H // 2, ec_i * CELL_W + CELL_W // 2
                y_lo, y_hi = max(0, cy_px - 14), min(ROWS * CELL_H, cy_px + 15)
                x_lo, x_hi = max(0, cx_px - 14), min(COLS * CELL_W, cx_px + 15)
                py, px = np.mgrid[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
                glow = np.exp(-((py - cy_px) ** 2 + (px - cx_px) ** 2) / (2.0 * (5.0 ** 2))) * (0.16 + 0.18 * eye_pulse)
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
                    glow = np.exp(-((py - cy_px) ** 2 + (px - cx_px) ** 2) / (2.0 * (4.0 ** 2))) * (0.30 * fade)
                    canvas[y_lo:y_hi, x_lo:x_hi] = (
                        canvas[y_lo:y_hi, x_lo:x_hi] * (1.0 - glow[:, :, None]) + e_col * glow[:, :, None]
                    )

                    y0, y1 = er * CELL_H, (er + 1) * CELL_H
                    x0, x1 = ec * CELL_W, (ec + 1) * CELL_W
                    a3 = (atlas[CHAR_TO_IDX[emb["char"]]] * min(1.0, fade * 1.25))[:, :, None]
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
