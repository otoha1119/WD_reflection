# 水滴反射検出 前処理プログラム 使用ガイド

## 概要

このプログラムは、プラスチックコンテナなどの透明物体表面の水滴や反射を検出するための画像前処理システムです。
最新の画像処理技術（CLAHE、ガンマ補正、Top-hat変換、HSV色空間分析）を組み合わせて、高精度な水滴検出を実現します。

## ファイル構成

### プログラムファイル

1. **water_droplet_preprocessing.py** - 標準版
   - 保守的なパラメータ設定
   - 精度重視の検出
   - 検出割合: 約0.86%

2. **improved_preprocessing.py** - 改善版（推奨）
   - 適応的なパラメータ設定
   - より多くの水滴を検出
   - 検出割合: 約2.28%
   - 検出領域数: 198個

### 出力ディレクトリ

- **preprocessing_results/** - 標準版の出力
- **improved_preprocessing_results/** - 改善版の出力（推奨）

## 使い方

### 基本的な使用方法

```bash
# 改善版を実行（推奨）
python3 improved_preprocessing.py

# 標準版を実行
python3 water_droplet_preprocessing.py
```

### カスタム画像での実行

プログラム内の以下の行を編集：

```python
# 画像パスを変更
input_image = "/path/to/your/image.png"
output_dir = "/path/to/output/directory"
```

## 出力ファイルの説明

### 標準版の出力

1. **01_original.png** - 元の画像
2. **02_gamma_correction.png** - ガンマ補正後
3. **03_clahe.png** - CLAHE適用後
4. **04_tophat.png** - Top-hat変換結果
5. **13_integrated_mask.png** - 最終検出マスク
6. **14_overlay_mask.png** - オーバーレイ表示

### 改善版の出力（推奨）

1. **01_original.png** - 元の画像
2. **02_gamma.png** - ガンマ補正後（γ=0.5）
3. **03_clahe.png** - CLAHE適用後（clipLimit=3.0）
4. **04_tophat.png / 04_tophat_heatmap.png** - Top-hat変換結果
5. **05_tophat_mask.png** - Top-hatベースのマスク
6. **06_specular_mask.png** - 適応的鏡面反射マスク
7. **07_combined_mask.png** - 統合マスク
8. **08_final_mask.png** - 最終検出マスク（★重要）
9. **09_overlay_red.png** - 赤色オーバーレイ
10. **10_overlay_green.png** - 緑色オーバーレイ
11. **11_contours.png** - 輪郭可視化（★推奨）
12. **detailed_report.txt** - 詳細レポート

## テスト画像の検出結果

### 画像: 2.png（コンテナ画像）

#### 改善版の結果
- **画像サイズ**: 1176 x 1036 ピクセル
- **検出ピクセル数**: 27,796 ピクセル
- **検出割合**: 2.28%
- **検出領域数**: 198個
- **最大領域面積**: 9,380 ピクセル
- **平均領域面積**: 112.8 ピクセル

#### 検出された水滴の特徴
- コンテナ内部表面の小さな水滴
- エッジ部分の反射
- コンテナ壁面の水滴
- 底部の大きな水たまり

## 主要な技術とパラメータ

### 1. ガンマ補正
- **標準版**: γ = 0.6
- **改善版**: γ = 0.5（より強い強調）
- **効果**: 暗い領域を明るくし、反射を強調

### 2. CLAHE（局所コントラスト強調）
- **標準版**: clipLimit = 2.5
- **改善版**: clipLimit = 3.0（より強い強調）
- **タイルサイズ**: 8x8
- **効果**: 局所的なコントラストを強調

### 3. Top-hat変換
- **標準版**: kernel_size = 9
- **改善版**: kernel_size = 7（小さな水滴に対応）
- **効果**: 明るい反射を背景から分離

### 4. 適応的鏡面反射検出（改善版のみ）
- **V閾値**: max(mean + 1.5*std, 150)
- **S閾値**: 80
- **効果**: 画像の明度に基づいて動的に閾値を調整

### 5. 面積フィルタリング
- **最小面積**: 20 ピクセル
- **効果**: ノイズを除去し、有意義な水滴のみを保持

## パラメータ調整のヒント

### より多くの水滴を検出したい場合
```python
# ガンマ値を下げる（0.4～0.5）
gamma = 0.45

# CLAHEのclipLimitを上げる（3.0～4.0）
clip_limit = 3.5

# 最小面積を下げる（10～20）
min_area = 15
```

### 誤検出を減らしたい場合
```python
# ガンマ値を上げる（0.6～0.7）
gamma = 0.65

# CLAHEのclipLimitを下げる（2.0～2.5）
clip_limit = 2.0

# 最小面積を上げる（30～50）
min_area = 40
```

## 次のステップ：反射除去

検出されたマスク（08_final_mask.png）を使用して、OpenCVのinpainting機能で反射を除去できます。

### サンプルコード

```python
import cv2

# 画像とマスクを読み込む
image = cv2.imread('01_original.png')
mask = cv2.imread('08_final_mask.png', cv2.IMREAD_GRAYSCALE)

# inpaintingで反射を除去
result = cv2.inpaint(image, mask, inpaintRadius=3, 
                    flags=cv2.INPAINT_TELEA)

# 結果を保存
cv2.imwrite('reflection_removed.png', result)
```

### inpaintingのパラメータ

- **inpaintRadius**: 3～7（小さいほど詳細、大きいほど滑らか）
- **flags**:
  - `cv2.INPAINT_TELEA`: 高速、シャープ
  - `cv2.INPAINT_NS`: 遅い、滑らか

## 評価指標の提案

反射除去の効果を評価するための指標：

### 1. PSNR（Peak Signal-to-Noise Ratio）
```python
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
```

### 2. SSIM（Structural Similarity Index）
```python
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)
```

### 3. 反射領域の減少率
```python
def reflection_reduction_rate(mask_before, mask_after):
    pixels_before = np.sum(mask_before > 0)
    pixels_after = np.sum(mask_after > 0)
    reduction = (pixels_before - pixels_after) / pixels_before * 100
    return reduction
```

## トラブルシューティング

### 問題: 検出された領域が少なすぎる
**解決策**:
- ガンマ値を下げる（0.4～0.5）
- CLAHEのclipLimitを上げる（3.0～4.0）
- 最小面積を下げる（10～15）

### 問題: 誤検出が多すぎる
**解決策**:
- ガンマ値を上げる（0.6～0.7）
- 最小面積を上げる（30～50）
- モルフォロジー演算のiterationsを増やす

### 問題: 小さな水滴が検出されない
**解決策**:
- Top-hatのkernel_sizeを小さくする（5～7）
- 最小面積を下げる（10～20）
- CLAHEのtileSizeを小さくする（4x4）

### 問題: 画像全体が暗すぎる
**解決策**:
- ガンマ値をさらに下げる（0.3～0.4）
- CLAHEのclipLimitを上げる（4.0～5.0）

## 技術的な詳細

### 処理の流れ

1. **前処理段階**
   - ガンマ補正で暗い領域を明るく
   - CLAHEで局所コントラストを強調

2. **特徴抽出段階**
   - Top-hat変換で明るい反射を抽出
   - HSV色空間で鏡面反射を検出

3. **統合段階**
   - 複数のマスクをOR結合
   - モルフォロジー演算でノイズ除去

4. **フィルタリング段階**
   - 面積ベースで小さすぎる領域を除去
   - 最終マスクを生成

### 使用ライブラリ

- **OpenCV**: 画像処理の中核
- **NumPy**: 数値計算
- **pathlib**: ファイル操作

### システム要件

- Python 3.6以上
- OpenCV 4.0以上
- NumPy 1.18以上

## 参考文献

このプログラムは以下の技術文書に基づいて実装されています：

- "Water Droplet Detection Image Preprocessing Technologies: A Complete Guide"
- 最新の研究（2024年）に基づくベストプラクティス
- CLAHE、Top-hat変換、HSV色空間分析などの確立された技術

## サポート

質問や問題がある場合は、以下を確認してください：

1. **詳細レポート**: `detailed_report.txt`に処理の詳細情報
2. **可視化画像**: 各処理段階の結果を確認
3. **パラメータ調整**: 上記のヒントを参考に調整

## ライセンスと著作権

このプログラムは教育目的で作成されました。
商用利用の際は適切なライセンスを確認してください。

---

**最終更新**: 2025年10月31日
**バージョン**: 1.0
**推奨プログラム**: improved_preprocessing.py
