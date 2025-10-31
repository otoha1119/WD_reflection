#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水滴候補マスク生成（元のHSV + Top-hatに、局所z-score と DoG を枝として追加）
- 入力:  /workspace/results/03_fully_preprocessed.png
- 出力:  /workspace/results/mask.png, /workspace/results/mask_visualization.png
- 可視化: 半透明オーバレイ + 線幅1px
"""
import cv2
import numpy as np
from pathlib import Path

# =================== パス設定 ===================
INPUT_PREPROC = Path("/workspace/results/03_fully_preprocessed.png")
OUT_DIR       = Path("/workspace/results")

# =================== 既存ロジックの既定値（必要ならここだけ触る） ===================
# HSV 閾値（元の発想を保持）
V_K      = 1.5     # v_mean + V_K * v_std
V_FLOOR  = 150     # V の床
S_MAX    = 80      # 低彩度上限（S < S_MAX）

# Top-hat 閾値（元の発想を保持）
TOPHAT_SE_SIZE = 9           # 構造要素のサイズ（奇数）。大きくすると太い/にじみも反応
TOPHAT_K       = 1.0         # tophat_mean + TOPHAT_K * tophat_std
TOPHAT_FLOOR   = 10          # Top-hat の床（最少）

# 形態学と面積
OPEN_K   = 3                 # オープン核
CLOSE_K  = 5                 # クローズ核
MIN_AREA = 20                # 面積下限（小片除去）

# =================== 見逃し対策“追加枝”の既定値 ===================
# 1) 局所 z-score（照明変動に頑健）
Z_WIN = 31                   # 局所平均/分散の窓（奇数）
Z_THR = 2.2                  # z > Z_THR を採用

# 2) DoG（Difference of Gaussians, バンドパス）
DOG_SIGMA1 = 1.2
DOG_SIGMA2 = 2.0
DOG_PERCENTILE = 98.5        # 上位何％を採用するか（大域依存を減らす）

# =================== ユーティリティ ===================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def area_filter(mask, min_area=MIN_AREA):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def overlay_and_contours(img_bgr, mask_bin):
    """半透明で塗ってから輪郭を1pxで描画"""
    vis = img_bgr.copy()
    overlay = vis.copy()
    #overlay[mask_bin > 0] = (0, 255, 0)  # 緑に塗る
    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)  # 線幅1px

    # 統計表示
    pixels = int((mask_bin > 0).sum())
    text1 = f"Detected: {len(contours)} regions"
    text2 = f"Pixels: {pixels}"
    cv2.putText(vis, text1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, text2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return vis

# =================== メイン処理 ===================
def main():
    if not INPUT_PREPROC.exists():
        raise FileNotFoundError(f"前処理画像が見つかりません: {INPUT_PREPROC}")
    ensure_dir(OUT_DIR)

    img = cv2.imread(str(INPUT_PREPROC), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("画像読込に失敗しました。")
    gray = to_gray(img)

    # ---- HSV 枝（元のロジック） ----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.float32)
    s = s.astype(np.float32)

    v_mean, v_std = float(np.mean(v)), float(np.std(v))
    v_thr = max(v_mean + V_K * v_std, float(V_FLOOR))
    s_thr = float(S_MAX)
    specular_mask = ((v > v_thr) & (s < s_thr)).astype(np.uint8) * 255

    # ---- Top-hat 枝（元のロジック）----
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_SE_SIZE, TOPHAT_SE_SIZE))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)

    th_mean, th_std = float(np.mean(tophat)), float(np.std(tophat))
    th_thr = max(th_mean + TOPHAT_K * th_std, float(TOPHAT_FLOOR))
    _, tophat_mask = cv2.threshold(tophat, int(th_thr), 255, cv2.THRESH_BINARY)

    # 和集合（ここに追加枝を OR していく）
    combined_mask = cv2.bitwise_or(specular_mask, tophat_mask)

    # ---- 追加枝1: 局所 z-score 亮度 ----
    g32 = gray.astype(np.float32)
    mean = cv2.boxFilter(g32, ddepth=-1, ksize=(Z_WIN, Z_WIN), normalize=True)
    mean_sq = cv2.boxFilter(g32 * g32, ddepth=-1, ksize=(Z_WIN, Z_WIN), normalize=True)
    var = np.clip(mean_sq - mean * mean, 1e-6, None)
    zmap = (g32 - mean) / np.sqrt(var)
    zmask = (zmap > Z_THR).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    zmask = cv2.morphologyEx(zmask, cv2.MORPH_CLOSE, k5)
    zmask = cv2.morphologyEx(zmask, cv2.MORPH_OPEN,  k3)

    combined_mask = cv2.bitwise_or(combined_mask, zmask)

    # ---- 追加枝2: DoG（Difference of Gaussians）----
    g1 = cv2.GaussianBlur(gray, (0, 0), DOG_SIGMA1)
    g2 = cv2.GaussianBlur(gray, (0, 0), DOG_SIGMA2)
    dog = cv2.subtract(g1, g2).astype(np.float32)
    p = np.percentile(dog, DOG_PERCENTILE)
    dogmask = (dog >= p).astype(np.uint8) * 255
    dogmask = cv2.morphologyEx(dogmask, cv2.MORPH_CLOSE, k5)

    combined_mask = cv2.bitwise_or(combined_mask, dogmask)

    # ---- 後処理（形状整え + 面積フィルタ）----
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))

    # 穴埋め優先（Close → Open）
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask,          cv2.MORPH_OPEN,  open_k)

    # 面積で小片除去
    mask = area_filter(mask, MIN_AREA)

    # ---- 出力 ----
    mask_fp = OUT_DIR / "mask.png"
    vis_fp  = OUT_DIR / "mask_visualization.png"
    cv2.imwrite(str(mask_fp), mask)
    vis = overlay_and_contours(img, mask)
    cv2.imwrite(str(vis_fp), vis)

    print(f"[OK] mask  -> {mask_fp}")
    print(f"[OK] vis   -> {vis_fp}")

if __name__ == "__main__":
    main()
