import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy import ndimage

SRC = "hero_crop_4x5.jpg"  # path to your source photo
GRID_W, GRID_H = 176, 200  # coarser grid, scaled toward ~17k dot target

im = Image.open(SRC).convert("RGB")
target_ratio = GRID_W / GRID_H
w, h = im.size
src_ratio = w / h
if src_ratio > target_ratio:
    new_w = int(h * target_ratio)
    left = (w - new_w) // 2
    im = im.crop((left, 0, left + new_w, h))
else:
    new_h = int(w / target_ratio)
    top = int((h - new_h) * 0.15)
    im = im.crop((0, top, w, top + new_h))

im = im.resize((GRID_W, GRID_H), Image.LANCZOS)
gray = ImageOps.grayscale(im)
gray = ImageOps.autocontrast(gray, cutoff=1)
gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
gray = ImageEnhance.Contrast(gray).enhance(1.3)
arr = np.array(gray).astype(np.float32) + 18  # lift black point so navy suit/hair don't go near-solid
gray = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

# --- background segmentation for dark mode ---
rgb_arr = np.array(im).astype(np.float32)
corner_size = 6
corners = np.concatenate([
    rgb_arr[:corner_size, :corner_size].reshape(-1, 3),
    rgb_arr[:corner_size, -corner_size:].reshape(-1, 3),
    rgb_arr[-corner_size:, :corner_size].reshape(-1, 3),
    rgb_arr[-corner_size:, -corner_size:].reshape(-1, 3),
])
bg_color = np.median(corners, axis=0)
dist = np.linalg.norm(rgb_arr - bg_color, axis=2)
thresh = np.percentile(dist, 35)
fg_mask = dist > max(thresh, 18)
fg_mask = ndimage.binary_closing(fg_mask, structure=np.ones((3, 3)), iterations=2)
fg_mask = ndimage.binary_fill_holes(fg_mask)
labeled, n = ndimage.label(fg_mask)
if n > 0:
    sizes = ndimage.sum(fg_mask, labeled, range(1, n + 1))
    fg_mask = labeled == (np.argmax(sizes) + 1)
fg_mask_eroded = ndimage.binary_erosion(fg_mask, structure=np.ones((2, 2)), iterations=1)

# --- Floyd-Steinberg dithering, serpentine order ---
def floyd_steinberg_serpentine(img_gray):
    arr = np.array(img_gray).astype(np.float32)
    h, w = arr.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        xr = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        direction = 1 if y % 2 == 0 else -1
        for x in xr:
            old = arr[y, x]
            new = 255.0 if old > 127 else 0.0
            out[y, x] = 1 if new > 127 else 0
            err = old - new
            if 0 <= x + direction < w:
                arr[y, x + direction] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - direction < w:
                    arr[y + 1, x - direction] += err * 3 / 16
                arr[y + 1, x] += err * 5 / 16
                if 0 <= x + direction < w:
                    arr[y + 1, x + direction] += err * 1 / 16
    return out  # 0 = dot(dark), 1 = no dot(light)

dith = floyd_steinberg_serpentine(gray)
dark_dots = (dith == 0) & fg_mask_eroded       # dark mode: dots only inside segmented subject
light_dots = (dith == 0)                        # light mode: keep background, draw dark parts

print("v3 grid", GRID_W, "x", GRID_H)
print("v3 dark mode dot count:", dark_dots.sum())
print("v3 light mode dot count:", light_dots.sum())

np.save("dark_dots_v3.npy", dark_dots)
np.save("light_dots_v3.npy", light_dots)
import numpy as np
from PIL import Image, ImageDraw

W, H = 380, 431  # portrait frame pixel size (matches portrait box in banner)

def sample_points_from_mask(mask, n_target):
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) == 0:
        return pts
    if len(pts) > n_target:
        idx = np.random.choice(len(pts), n_target, replace=False)
        pts = pts[idx]
    return pts

rng = np.random.default_rng(42)

# --- Glyph 1: Waveform (literal digital timing diagram — clock + data transitions) ---
img1 = Image.new("L", (W, H), 0)
d1 = ImageDraw.Draw(img1)
cx, cy = W / 2, H / 2
line_w = 12

def square_wave_path(x0, y_hi, y_lo, x_step, n, start_low=True):
    cur_x = x0
    cur_y = y_lo if start_low else y_hi
    high = not start_low
    path = [(cur_x, cur_y)]
    for i in range(n):
        nxt_x = cur_x + x_step
        nxt_y = y_hi if not high else y_lo
        path.append((cur_x, cur_y))
        path.append((nxt_x, cur_y))
        path.append((nxt_x, nxt_y))
        cur_x, cur_y = nxt_x, nxt_y
        high = not high
    return path

x_step = W * 0.11
path_clk = square_wave_path(W * 0.06, cy - 95, cy - 15, x_step, 9)
d1.line(path_clk, fill=255, width=line_w, joint="curve")

path_data = square_wave_path(W * 0.06, cy + 95, cy + 15, x_step * 1.35, 7, start_low=False)
d1.line(path_data, fill=255, width=line_w, joint="curve")

mask1 = np.array(img1) > 127
pts1 = sample_points_from_mask(mask1, 900)

# --- Glyph 2: Vivado-inspired diamond/hex motif (stylized, not the literal brand mark) ---
img2 = Image.new("L", (W, H), 0)
d2 = ImageDraw.Draw(img2)
cx, cy = W / 2, H / 2
r = 150
hexagon = [
    (cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a)))
    for a in range(0, 360, 60)
]
d2.polygon(hexagon, outline=255, width=16)
inner_r = 70
diamond = [
    (cx, cy - inner_r), (cx + inner_r, cy), (cx, cy + inner_r), (cx - inner_r, cy)
]
d2.polygon(diamond, fill=255)
mask2 = np.array(img2) > 127
pts2 = sample_points_from_mask(mask2, 900)

# --- Glyph 3: Python-inspired twin-curve motif (stylized, not the literal brand mark) ---
img3 = Image.new("L", (W, H), 0)
d3 = ImageDraw.Draw(img3)
cx, cy = W / 2, H / 2
box_a = [cx - 130, cy - 150, cx + 40, cy + 20]
box_b = [cx - 40, cy - 20, cx + 130, cy + 150]
d3.arc(box_a, start=200, end=430, fill=255, width=46)
d3.arc(box_b, start=20, end=250, fill=255, width=46)
d3.rectangle([cx - 108, cy - 118, cx - 84, cy - 94], fill=255)
d3.rectangle([cx + 84, cy + 94, cx + 108, cy + 118], fill=255)
mask3 = np.array(img3) > 127
pts3 = sample_points_from_mask(mask3, 900)

print("Glyph point counts:", len(pts1), len(pts2), len(pts3))

np.save("glyph_waveform.npy", pts1)
np.save("glyph_vivado.npy", pts2)
np.save("glyph_python.npy", pts3)
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(7)

W, H = 380, 431  # portrait frame pixel box
GRID_W, GRID_H = 176, 200

dark_dots = np.load("dark_dots_v3.npy")  # bool grid [row, col] = [y, x]
ys, xs = np.where(dark_dots)
px = xs.astype(np.float32) / GRID_W * W
py = ys.astype(np.float32) / GRID_H * H
portrait_full = np.stack([px, py], axis=1)
print("Full portrait dot count:", len(portrait_full))

# --- drift bands: cluster into ~94 groups, WITH per-dot jitter added before clustering
# to avoid the "grid trap" (quantized linear drift recreating a square grid) ---
jitter_sigma = 4.0
pts_for_cluster = portrait_full + rng.normal(0, jitter_sigma, portrait_full.shape)
n_bands = 94
centroids, band_labels = kmeans2(pts_for_cluster.astype(np.float64), n_bands, minit="++", seed=3)
print("Band count actual:", len(np.unique(band_labels)))

# --- intro shimmer groups: ~60 groups, scattered across whole portrait (not spatial regions) ---
n_intro_groups = 60
shuffled_idx = rng.permutation(len(portrait_full))
intro_group_id = np.empty(len(portrait_full), dtype=int)
intro_group_id[shuffled_idx] = shuffled_idx % n_intro_groups  # random scatter, not spatial

global_centroid = portrait_full.mean(axis=0)
group_centroid_spread = []
for g in range(n_intro_groups):
    mask = intro_group_id == g
    if mask.sum() > 0:
        gc = portrait_full[mask].mean(axis=0)
        group_centroid_spread.append(np.linalg.norm(gc - global_centroid))
evenness = np.std(group_centroid_spread) / (np.linalg.norm([W, H]))
print(f"Evenness metric: {evenness:.3f} (target ~0.05 good, ~0.7 patchy)")

# --- traveler dots: ~900 sparse subset for logo morph, matched via greedy nearest-neighbor
# (an approximation of optimal transport — exact Hungarian on 900x900 is expensive) ---
n_travelers = 900
trav_idx = rng.choice(len(portrait_full), n_travelers, replace=False)
travelers_start = portrait_full[trav_idx]

def match_greedy(src, dst):
    """Greedy nearest-neighbor bipartite matching, approximating optimal transport."""
    order = rng.permutation(len(src))
    assignment = np.zeros(len(src), dtype=int)
    free_tree_pts = dst.copy()
    free_idx_map = list(range(len(dst)))
    for i in order:
        if len(free_idx_map) == 0:
            break
        ft = cKDTree(free_tree_pts)
        d, j = ft.query(src[i])
        real_j = free_idx_map[j]
        assignment[i] = real_j
        free_tree_pts = np.delete(free_tree_pts, j, axis=0)
        free_idx_map.pop(j)
    return assignment

glyph_waveform = np.load("glyph_waveform.npy")
glyph_vivado = np.load("glyph_vivado.npy")
glyph_python = np.load("glyph_python.npy")

def fit_count(pts, n):
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
        return pts[idx]
    else:
        idx = rng.choice(len(pts), n - len(pts), replace=True)
        return np.concatenate([pts, pts[idx]], axis=0)

glyph_waveform = fit_count(glyph_waveform, n_travelers)
glyph_vivado = fit_count(glyph_vivado, n_travelers)
glyph_python = fit_count(glyph_python, n_travelers)

assign_1 = match_greedy(travelers_start, glyph_waveform)
target_1 = glyph_waveform[assign_1]
assign_2 = match_greedy(target_1, glyph_vivado)
target_2 = glyph_vivado[assign_2]
assign_3 = match_greedy(target_2, glyph_python)
target_3 = glyph_python[assign_3]

np.savez("portrait_layers.npz",
         portrait_full=portrait_full, band_labels=band_labels, centroids=centroids,
         intro_group_id=intro_group_id, travelers_start=travelers_start, trav_idx=trav_idx,
         target_1=target_1, target_2=target_2, target_3=target_3)
print("Saved portrait_layers.npz — traveler count:", len(travelers_start))
import numpy as np

data = np.load("portrait_layers.npz")
portrait_full = data["portrait_full"]
band_labels = data["band_labels"]
intro_group_id = data["intro_group_id"]
travelers_start = data["travelers_start"]
target_1 = data["target_1"]  # waveform
target_2 = data["target_2"]  # vivado
target_3 = data["target_3"]  # python

PORTRAIT_W, PORTRAIT_H = 380, 431
PORTRAIT_X, PORTRAIT_Y = 34, 96
DOT = 380 / 176
DOT_SIZE = DOT * 0.72
BANNER_W, BANNER_H = 1180, 610

def dot_path(points, size=DOT_SIZE, ox=PORTRAIT_X, oy=PORTRAIT_Y):
    """Small squares, crispEdges. Integer-rounded coords -- ~2x smaller file."""
    s = max(round(size), 1)
    parts = []
    for x, y in points:
        px, py = round(x + ox), round(y + oy)
        parts.append(f"M{px} {py}h{s}v{s}h-{s}z")
    return "".join(parts)

logo_centroid = target_1.mean(axis=0)
portrait_centroid = portrait_full.mean(axis=0)
drift_vec = (logo_centroid - portrait_centroid) * 0.42

T = 14.2
kt = [0, 3.0/T, 4.3/T, 6.3/T, 7.6/T, 9.6/T, 10.9/T, 12.9/T, 1.0]
kt_str = ";".join(f"{v:.4f}" for v in kt)

def build_bands_svg(color):
    out = []
    n_bands = band_labels.max() + 1
    for b in range(n_bands):
        mask = band_labels == b
        pts = portrait_full[mask]
        if len(pts) == 0:
            continue
        d = dot_path(pts)
        dx, dy = drift_vec
        tx_vals = f"0,0;0,0;{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};0,0;0,0"
        op_vals = "1;1;0.15;0.15;0.15;0.15;0.15;1;1"
        out.append(
            f'<g fill="{color}" opacity="0" shape-rendering="crispEdges">'
            f'<animate attributeName="opacity" begin="3.2s" dur="{T}s" repeatCount="indefinite" '
            f'keyTimes="{kt_str}" values="{op_vals}" fill="freeze"/>'
            f'<path d="{d}">'
            f'<animateTransform attributeName="transform" attributeType="XML" type="translate" '
            f'begin="3.2s" dur="{T}s" repeatCount="indefinite" keyTimes="{kt_str}" values="{tx_vals}"/>'
            f'</path></g>'
        )
    return "".join(out)

def build_intro_svg(color):
    out = []
    n_groups = 60
    for g in range(n_groups):
        mask = intro_group_id == g
        pts = portrait_full[mask]
        if len(pts) == 0:
            continue
        d = dot_path(pts)
        begin_t = (g / n_groups) * 1.7
        out.append(
            f'<path fill="{color}" shape-rendering="crispEdges" opacity="0" d="{d}">'
            f'<animate attributeName="opacity" begin="{begin_t:.3f}s" dur="1.1s" '
            f'values="0;1" keyTimes="0;1" fill="freeze"/>'
            f'<animate attributeName="opacity" begin="3.2s" dur="0.01s" values="0;0" fill="freeze"/>'
            f'</path>'
        )
    return "".join(out)

def build_travelers_svg(color):
    n = len(travelers_start)
    op_vals = "0;0;1;1;1;1;1;0;0"
    out = []
    r = 1.7
    for i in range(n):
        sx, sy = round(travelers_start[i][0]+PORTRAIT_X), round(travelers_start[i][1]+PORTRAIT_Y)
        wx, wy = round(target_1[i][0]+PORTRAIT_X), round(target_1[i][1]+PORTRAIT_Y)
        vx, vy = round(target_2[i][0]+PORTRAIT_X), round(target_2[i][1]+PORTRAIT_Y)
        pxx, pyy = round(target_3[i][0]+PORTRAIT_X), round(target_3[i][1]+PORTRAIT_Y)
        cx_seq = f"{sx};{sx};{wx};{wx};{vx};{vx};{pxx};{pxx};{sx}"
        cy_seq = f"{sy};{sy};{wy};{wy};{vy};{vy};{pyy};{pyy};{sy}"
        out.append(
            f'<circle r="{r}" fill="{color}" opacity="0" cx="{sx}" cy="{sy}">'
            f'<animate attributeName="cx" begin="3.2s" dur="{T}s" repeatCount="indefinite" keyTimes="{kt_str}" values="{cx_seq}"/>'
            f'<animate attributeName="cy" begin="3.2s" dur="{T}s" repeatCount="indefinite" keyTimes="{kt_str}" values="{cy_seq}"/>'
            f'<animate attributeName="opacity" begin="3.2s" dur="{T}s" repeatCount="indefinite" keyTimes="{kt_str}" values="{op_vals}"/>'
            f'</circle>'
        )
    return "".join(out)

def build_info_panel(rows, panel_x, panel_y, chrome_color, text_color, dim_color, font="14"):
    out = []
    row_h = 23
    value_x_end = BANNER_W - 48
    for i, (label, value) in enumerate(rows):
        y = panel_y + i * row_h
        label_full = label.upper()
        leader_start = panel_x + len(label_full) * 7.3 + 10
        leader_end = value_x_end - len(value) * 7.0 - 10
        leader_width = max(leader_end - leader_start, 10)
        n_dots = int(leader_width // 6)
        dots = " ".join(["."] * max(n_dots, 1))
        out.append(f'<text x="{panel_x}" y="{y}" font-family="Fira Code, monospace" font-size="{font}" fill="{text_color}" opacity="0.92">{label_full}</text>')
        out.append(f'<text x="{leader_start:.1f}" y="{y}" font-family="Fira Code, monospace" font-size="12" fill="{dim_color}" opacity="0.35">{dots}</text>')
        out.append(f'<text x="{value_x_end}" y="{y}" text-anchor="end" font-family="Fira Code, monospace" font-size="{font}" fill="{chrome_color}" textLength="{len(value)*7.0:.0f}" lengthAdjust="spacingAndGlyphs">{value}</text>')
    return "".join(out)

ROWS = [
    ("Subject", "P. Ramareddy Huchchanagowdra"),
    ("Role", "RTL Design &amp; Verification Eng."),
    ("Origin", "Bengaluru, India"),
    ("Education", "B.E. ECE — VTU, Exp. 2027"),
    ("Status", "Shipping RTL"),
    ("ToolChain", "Vivado / ModelSim / Icarus"),
    ("Core.Lang", "Verilog · SystemVerilog · Python"),
    ("Core.Verify", "Self-Checking TBs · UVM Fund."),
    ("Core.Flow", "STA · CDC · DFT · Synthesis"),
    ("Core.Embedded", "STM32 · ARM · AVR"),
    ("Grid.Mail", "rahulgowda3511@gmail.com"),
    ("Grid.Portfolio", "coming soon"),
    ("Grid.LinkedIn", "/in/pakeergowda-rh"),
    ("Grid.GitHub", "@Pakeergowda-VLSI"),
]

def build_svg(mode):
    if mode == "dark":
        bg, portrait_color, traveler_color = "#0A101F", "#A78BFA", "#22D3EE"
        chrome, text_color, dim = "#22D3EE", "#C4B5FD", "#334155"
        panel_bg, border = "#0d1424", "#1e293b"
    else:
        bg, portrait_color, traveler_color = "#F5F3FF", "#7C3AED", "#0891B2"
        chrome, text_color, dim = "#0891B2", "#4C1D95", "#C4B5FD"
        panel_bg, border = "#FFFFFF", "#DDD6FE"

    bands_svg = build_bands_svg(portrait_color)
    intro_svg = build_intro_svg(portrait_color)
    travelers_svg = build_travelers_svg(traveler_color)
    info_svg = build_info_panel(ROWS, PORTRAIT_X + PORTRAIT_W + 46, 110, chrome, text_color, dim)
    accent = "#10B981"

    return f'''<svg viewBox="0 0 {BANNER_W} {BANNER_H}" xmlns="http://www.w3.org/2000/svg" font-family="Fira Code, monospace">
<rect width="{BANNER_W}" height="{BANNER_H}" fill="{bg}"/>
<rect x="1" y="1" width="{BANNER_W-2}" height="{BANNER_H-2}" fill="none" stroke="{border}" stroke-width="1"/>
<rect x="0" y="0" width="{BANNER_W}" height="40" fill="{panel_bg}" opacity="0.6"/>
<line x1="0" y1="40" x2="{BANNER_W}" y2="40" stroke="{border}" stroke-width="1"/>
<circle cx="24" cy="20" r="6" fill="#EF4444"/>
<circle cx="46" cy="20" r="6" fill="#F59E0B"/>
<circle cx="68" cy="20" r="6" fill="#10B981"/>
<text x="{BANNER_W/2}" y="25" text-anchor="middle" font-size="13" fill="{text_color}" opacity="0.7">profile.sh --live</text>
<circle cx="{BANNER_W-140}" cy="20" r="4" fill="#EF4444">
  <animate attributeName="opacity" values="1;0.3;1" dur="1.6s" repeatCount="indefinite"/>
</circle>
<text x="{BANNER_W-128}" y="24" font-size="12" fill="#EF4444" letter-spacing="1">LIVE</text>
<rect x="{BANNER_W-96}" y="10" width="80" height="20" rx="10" fill="{accent}" opacity="0.15" stroke="{accent}" stroke-width="1"/>
<text x="{BANNER_W-56}" y="24" text-anchor="middle" font-size="12" fill="{accent}">@Pakeergowda</text>
<rect x="{PORTRAIT_X-8}" y="{PORTRAIT_Y-8}" width="{PORTRAIT_W+16}" height="{PORTRAIT_H+16}" fill="none" stroke="{border}" stroke-width="1"/>
<text x="{PORTRAIT_X}" y="{PORTRAIT_Y-16}" font-size="11" letter-spacing="2" fill="{chrome}" opacity="0.75">VISUAL.MAP</text>
{intro_svg}
{bands_svg}
{travelers_svg}
<text x="{PORTRAIT_X + PORTRAIT_W + 46}" y="88" font-size="11" letter-spacing="2" fill="{chrome}" opacity="0.75">SYSTEM.INFO</text>
<line x1="{PORTRAIT_X + PORTRAIT_W + 46}" y1="96" x2="{BANNER_W-48}" y2="96" stroke="{border}" stroke-width="1"/>
{info_svg}
</svg>'''

with open("dark.svg", "w") as f:
    f.write(build_svg("dark"))
with open("light.svg", "w") as f:
    f.write(build_svg("light"))
print("Done")
name: generate contribution snake

on:
  schedule:
    - cron: "0 */12 * * *"   # every 12 hours
  workflow_dispatch: {}
  push:
    branches:
      - main

jobs:
  generate:
    permissions:
      contents: write
    runs-on: ubuntu-latest
    steps:
      - name: generate snake svgs
        uses: Platane/snk/svg-only@v3
        with:
          github_user_name: Pakeergowda-VLSI
          outputs: |
            dist/snake-dark.svg?palette=github-dark&color_snake=A78BFA&color_dots=2d3343,7C3AED,8B5CF6,A78BFA,22D3EE
            dist/snake-light.svg?palette=github-light&color_snake=7C3AED&color_dots=EDE9FE,C4B5FD,A78BFA,8B5CF6,0891B2

      - name: push to output branch
        uses: crazy-max/ghaction-github-pages@v3.1.0
        with:
          target_branch: output
          build_dir: dist
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          <p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Pakeergowda-VLSI/Pakeergowda-VLSI/main/dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Pakeergowda-VLSI/Pakeergowda-VLSI/main/light.svg" />
    <img src="https://raw.githubusercontent.com/Pakeergowda-VLSI/Pakeergowda-VLSI/main/dark.svg" alt="Pakeergowda R H — RTL Design & Verification Engineer" />
  </picture>
</p>

<br/>

<img src="https://streak-stats.demolab.com/?user=Pakeergowda-VLSI&background=0A101F&ring=22D3EE&fire=A78BFA&currStreakLabel=A78BFA&sideLabels=C4B5FD&dates=64748B&border=1e293b" width="100%"/>

<img src="https://github-readme-stats-seven-taupe-25.vercel.app/api?username=Pakeergowda-VLSI&show_icons=true&hide_rank=true&bg_color=0A101F&title_color=A78BFA&icon_color=22D3EE&text_color=C4B5FD&border_color=1e293b" width="49%"/>
<img src="https://github-readme-stats-seven-taupe-25.vercel.app/api/top-langs/?username=Pakeergowda-VLSI&layout=compact&bg_color=0A101F&title_color=A78BFA&text_color=C4B5FD&border_color=1e293b" width="49%"/>

<br/>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Pakeergowda-VLSI/Pakeergowda-VLSI/output/dist/snake-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Pakeergowda-VLSI/Pakeergowda-VLSI/output/dist/snake-light.svg" />
    <img src="https://raw.githubusercontent.com/Pakeergowda-VLSI/Pakeergowda-VLSI/output/dist/snake-dark.svg" alt="contribution snake" />
  </picture>
</p>

<br/>

<p align="center">
  <a href="mailto:rahulgowda3511@gmail.com"><img src="https://img.shields.io/badge/Gmail-0A101F?style=for-the-badge&logo=gmail&logoColor=A78BFA" /></a>&nbsp;&nbsp;
  <a href="https://www.linkedin.com/in/pakeergowda-rh/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" /></a>&nbsp;&nbsp;
  <a href="#"><img src="https://img.shields.io/badge/Portfolio-0A101F?style=for-the-badge&logo=vercel&logoColor=22D3EE" /></a>
</p>
