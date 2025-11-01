# crate_mask.py
# 入力 : /workspace/data/images/1.png
# 出力 : /workspace/results/1_mask.png, /workspace/results/1_outline.png

import cv2
import numpy as np
from pathlib import Path

IN_PATH   = Path("/workspace/data/images/120.png")
OUT_DIR   = Path("/workspace/results")
MASK_PATH = OUT_DIR / "1_mask.png"
OUTLINE_PATH = OUT_DIR / "1_outline.png"

def remove_long_horizontal_edges(img_shape, edges, min_len=300, gap=30, thick=22):
    """Houghでレーンの強い水平エッジを見つけ、太らせてマスク化（のちに除去用）。"""
    h, w = img_shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=min_len, maxLineGap=gap)
    rail = np.zeros((h, w), np.uint8)
    if lines is None:
        return rail
    for (x1,y1,x2,y2) in lines[:,0]:
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if ang < 12:  # ほぼ水平のみ（レーン）
            cv2.line(rail, (x1,y1), (x2,y2), 255, thick)
    rail = cv2.dilate(rail, cv2.getStructuringElement(cv2.MORPH_RECT,(19,19)), 1)
    return rail

def main():
    assert IN_PATH.exists(), f"入力が見つかりません: {IN_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img  = cv2.imread(str(IN_PATH), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # 1) 前処理
    blur = cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=7)

    # 2) 色抽出（HSV）
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower1 = np.array([45,  60, 15], dtype=np.uint8)
    upper1 = np.array([95, 255,140], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)

    # 明るすぎるハイライト除外
    v = hsv[:,:,2]
    hi = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY)[1]
    m1 = cv2.bitwise_and(m1, cv2.bitwise_not(hi))

    # 3) 形態学で整形
    m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)), 1)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), 1)

    # 4) レーン（水平方向の強エッジ）を検出して差し引く
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 60, 160)
    rails = remove_long_horizontal_edges(img.shape, edges)
    m2 = cv2.bitwise_and(m1, cv2.bitwise_not(rails))

    # 5) 連結成分から“箱っぽい最大領域”を選択
    cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h,w), np.uint8)
    if not cnts:
        cnts, _ = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            cv2.imwrite(str(MASK_PATH), mask)
            cv2.imwrite(str(OUTLINE_PATH), img)  # 何も描けない場合は原画像を保存
            return

    best = None
    best_score = -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.04*h*w:
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        ar = ww / max(1,hh)
        if not (1.1 <= ar <= 5.0):
            continue
        cy = y + hh/2
        bias = 1.0 - abs((cy/h) - 0.65)
        score = area * bias
        if score > best_score:
            best_score = score
            best = c
    if best is None:
        best = max(cnts, key=cv2.contourArea)

    # 6) 台形化（凸包→多角形近似）
    hull = cv2.convexHull(best)
    eps  = 0.015 * cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, eps, True)
    if len(poly) > 8:
        eps2 = 0.03 * cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, eps2, True)

    # マスク塗りつぶし
    cv2.fillPoly(mask, [poly], 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)

    # 7) 輪郭線だけを元画像に重ねた可視化（赤線）
    outline = img.copy()
    cv2.polylines(outline, [poly], isClosed=True, color=(0,0,255), thickness=4, lineType=cv2.LINE_AA)

    # 8) 保存
    cv2.imwrite(str(MASK_PATH), mask)
    cv2.imwrite(str(OUTLINE_PATH), outline)

if __name__ == "__main__":
    main()
