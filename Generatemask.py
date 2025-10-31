#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水滴候補マスク生成（統合安定版）
- 手法: HSV + Top-hat + 局所 z-score + DoG
- 合体しすぎ対策: Open->Close(3x3) + Watershed(弱) で分割
- 重要救済: Open/Close 版 と Watershed 版を OR で結合（取りこぼし回避）
- 可視化: 線のみ(1px)。サイズ・解像度は変更しない
- デバッグ: 中間マスクを保存、非ゼロ保険あり
入:  /workspace/results/03_fully_preprocessed.png
出:  /workspace/results/mask.png, mask_visualization.png, および _a〜_h*.png
"""
from pathlib import Path
import cv2
import numpy as np

# =================== パス ===================
INPUT_PREPROC = Path("/workspace/results/03_fully_preprocessed.png")
OUT_DIR       = Path("/workspace/results")

# =================== しきい・核（改良前の再現性を保ちつつ、ほんの少しだけ堅実に） ===================
# HSV
V_K      = 1.5      # v_threshold = mean(V) + V_K * std(V)  ←改良前相当
V_FLOOR  = 150      # 暗コマでの暴走防止
S_MAX    = 80       # 低彩度上限（緑寄りの鏡面を落としすぎない）

# Top-hat
TOPHAT_SE_SIZE = 9          # 小粒前提のスケールは維持（大粒は他枝で拾う）
TOPHAT_K       = 1.1        # 1.0より少しだけ厳しく（過剰結合を抑制）
TOPHAT_FLOOR   = 10

# z-score（局所的に明るい部を拾う柱。弱め緩和）
Z_WIN = 31
Z_THR = 2.1                 # 2.0よりわずかに上げ、ベタ塗りを抑えつつ回収力は残す

# DoG（筋状/にじみの反射を救う柱。やや緩め）
DOG_SIGMA1 = 1.2
DOG_SIGMA2 = 2.0
DOG_PERCENTILE = 98.3       # 98.0よりわずかに厳しく（結合し過ぎ防止）

# 形態学（分割優先・弱め）
OPEN_K   = 3
CLOSE_K  = 3
MIN_AREA = 20               # 改良前の粒度を保つ（必要なら後で上げる）

# Watershed（弱め設定：分割はするが消えにくい）
WS_MIN_PEAK_DIST = 2        # 近接ピーク距離（小さめ=細かく分ける）
WS_REL_PEAK      = 0.20     # 相対ピーク強度（低め=種を作りやすい）

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
    """元画像に1px輪郭のみ重ねる（解像度・見た目を維持）"""
    vis = img_bgr.copy()
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    return vis

def split_touching(mask_bin, min_peak_dist=WS_MIN_PEAK_DIST, rel_peak=WS_REL_PEAK):
    """距離変換 + マーカー制御 Watershed で接触粒子を分割（弱め設定）"""
    if mask_bin.max() == 0:
        return mask_bin
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax < 1e-3:
        return mask_bin

    k = 2 * int(min_peak_dist) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    local_max = (dist == cv2.dilate(dist, kernel)).astype(np.uint8)

    peaks = np.logical_and(local_max > 0, dist >= rel_peak * dmax).astype(np.uint8)
    num, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    if num <= 1:
        return mask_bin

    markers = markers + 1
    markers[mask_bin == 0] = 0

    color = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    split = (markers > 1).astype(np.uint8) * 255
    return split

# =================== メイン ===================
def main():
    if not INPUT_PREPROC.exists():
        raise FileNotFoundError(f"前処理画像が見つかりません: {INPUT_PREPROC}")
    ensure_dir(OUT_DIR)

    img = cv2.imread(str(INPUT_PREPROC), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("画像読込に失敗しました。")
    gray = to_gray(img)

    # ---- HSV 枝（改良前の核）----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.float32); s = s.astype(np.float32)
    v_mean, v_std = float(np.mean(v)), float(np.std(v))
    v_thr = max(v_mean + V_K * v_std, float(V_FLOOR))
    s_thr = float(S_MAX)
    specular_mask = ((v > v_thr) & (s < s_thr)).astype(np.uint8) * 255

    # ---- Top-hat 枝（小粒重視）----
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_SE_SIZE, TOPHAT_SE_SIZE))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    th_mean, th_std = float(np.mean(tophat)), float(np.std(tophat))
    th_thr = max(th_mean + TOPHAT_K * th_std, float(TOPHAT_FLOOR))
    _, tophat_mask = cv2.threshold(tophat, int(th_thr), 255, cv2.THRESH_BINARY)

    # ---- z-score 枝（照明変動に頑健）----
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

    # ---- DoG 枝（筋/にじみ）----
    g1 = cv2.GaussianBlur(gray, (0, 0), DOG_SIGMA1)
    g2 = cv2.GaussianBlur(gray, (0, 0), DOG_SIGMA2)
    dog = cv2.subtract(g1, g2).astype(np.float32)
    p = np.percentile(dog, DOG_PERCENTILE)
    dogmask = (dog >= p).astype(np.uint8) * 255
    dogmask = cv2.morphologyEx(dogmask, cv2.MORPH_CLOSE, k5)

    # ---- 統合 OR ----
    combined_mask = cv2.bitwise_or(specular_mask, tophat_mask)
    combined_mask = cv2.bitwise_or(combined_mask, zmask)
    combined_mask = cv2.bitwise_or(combined_mask, dogmask)

    # デバッグ保存
    cv2.imwrite(str(OUT_DIR / "_a_specular.png"), specular_mask)
    cv2.imwrite(str(OUT_DIR / "_b_tophat.png"),   tophat_mask)
    cv2.imwrite(str(OUT_DIR / "_c_zmask.png"),    zmask)
    cv2.imwrite(str(OUT_DIR / "_d_dogmask.png"),  dogmask)
    cv2.imwrite(str(OUT_DIR / "_e_or_all.png"),   combined_mask)

    # 非ゼロ保険
    if combined_mask.max() == 0:
        combined_mask = cv2.max(specular_mask, tophat_mask)
    if combined_mask.max() == 0:
        combined_mask = cv2.max(zmask, dogmask)
    cv2.imwrite(str(OUT_DIR / "_e_or_all_safed.png"), combined_mask)

    print("nonzero px:",
          "specular", int((specular_mask > 0).sum()),
          "tophat",   int((tophat_mask   > 0).sum()),
          "z",        int((zmask         > 0).sum()),
          "dog",      int((dogmask       > 0).sum()),
          "OR",       int((combined_mask > 0).sum()))

    # ---- 後処理 A: Open -> Close（弱め）----
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    mask_oc = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  open_k)
    mask_oc = cv2.morphologyEx(mask_oc,       cv2.MORPH_CLOSE, close_k)
    cv2.imwrite(str(OUT_DIR / "_f_after_openclose.png"), mask_oc)

    # ---- 後処理 B: Watershed（弱め）----
    mask_ws = split_touching(mask_oc, min_peak_dist=WS_MIN_PEAK_DIST, rel_peak=WS_REL_PEAK)
    cv2.imwrite(str(OUT_DIR / "_g_after_watershed.png"), mask_ws)

    # ---- 救済結合：分割（mask_ws）と非分割（mask_oc）の OR ----
    #   → 分割で生じた取りこぼしを回収しつつ、合体しすぎも抑止
    mask_merged = cv2.bitwise_or(mask_oc, mask_ws)
    cv2.imwrite(str(OUT_DIR / "_h_merge_ws_oc.png"), mask_merged)

    # ---- 面積フィルタ ----
    mask_final = area_filter(mask_merged, MIN_AREA)

    # ---- 出力 ----
    mask_fp = OUT_DIR / "mask.png"
    vis_fp  = OUT_DIR / "mask_visualization.png"
    cv2.imwrite(str(mask_fp), mask_final)
    vis = overlay_and_contours(img, mask_final)
    cv2.imwrite(str(vis_fp), vis)

    # 形状確認（解像度維持）
    print("IMG shape:", img.shape, "MASK shape:", mask_final.shape)
    print(f"[OK] mask  -> {mask_fp}")
    print(f"[OK] vis   -> {vis_fp}")

if __name__ == "__main__":
    main()
