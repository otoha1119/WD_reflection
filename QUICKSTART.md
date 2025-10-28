# クイックスタートガイド

## 緑色コンテナ反射除去システム

このプロジェクトは、暗所で撮影された緑色プラスチックコンテナの画像から水滴や鏡面反射を自動的に検出・除去するPythonシステムです。

## 1. セットアップ

```bash
# 1. ZIPファイルを解凍
unzip workspace_reflection_removal.zip
cd workspace

# 2. Python仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 必要なパッケージをインストール
pip install -r requirements.txt
```

## 2. 基本的な使い方

### サンプル画像でテスト（3枚の画像で動作確認）

```bash
python -m app --input data/samples
```

### 本番画像の処理（120枚の画像を処理）

```bash
# data/imagesフォルダに画像を配置してから実行
python -m app --input data/images
```

### 単一画像の処理

```bash
python -m app --single path/to/your/image.png
```

## 3. 出力ファイル

処理後、以下のファイルが生成されます：

- `out/mask/` - 反射部分のマスク画像（白黒）
- `out/result/` - 反射除去後の画像
- `out/eval/metrics.csv` - 評価指標の数値データ
- `out/eval/panels/` - Before/Mask/Afterの比較画像

## 4. アルゴリズムの特徴

### 4つの主要ステップ：

1. **コンテナ領域抽出**
   - Lab色空間で緑色を検出
   - 回転矩形で安定した領域抽出

2. **反射検出**
   - 局所Zスコアで相対的な明るさを検出
   - 白っぽさと飽和度を考慮

3. **形状分類**
   - Thin（線状）: 水滴の流れた跡
   - Blob（塊状）: 水滴本体

4. **インペインティング**
   - 形状に応じた最適な補間手法
   - Lab色空間で自然な仕上がり

## 5. 評価指標

- **SPP (Saturated Pixel Percentage)**: 飽和画素の削減率
- **EPR (Edge Preservation Ratio)**: エッジ保存率（目標≥0.9）
- **Colorfulness**: 色彩の保持率
- **SSIM**: 構造的類似性

## 6. パラメータ調整

`configs/config.yaml`を編集して調整可能：

```yaml
detect:
  z_thresh: 2.0    # 小さくすると検出感度UP
  s_thresh: 40     # 大きくすると白以外も検出

shape:
  dilate_blob: 2   # 大きくすると膨張強化

inpaint:
  radius: 3        # 大きくすると補間範囲拡大
```

## 7. トラブルシューティング

### 反射が検出されない場合
- `z_thresh`を1.5〜1.8に下げる
- `s_thresh`を50〜60に上げる

### 過検出（ラベル等を誤検出）
- `z_thresh`を2.5〜3.0に上げる
- `min_area`を30〜50に上げる

### デバッグモード
```bash
python -m app --input data/samples --debug --verbose
```

## 8. システムテスト

動作確認用のテストスクリプト：

```bash
python test_system.py
```

## 9. 実装の工夫点

### 精度向上の工夫
- 局所統計量（Zスコア）による照明変化への頑健性
- 形状別処理による最適化
- Lab色空間での処理による自然な色再現

### コードの可読性
- MVCアーキテクチャによる明確な責務分離
- 型ヒントとdocstringによる文書化
- 包括的なロギングとエラーハンドリング

### エラー耐性
- 自動的なパラメータ調整
- フォールバック機構
- 詳細なエラーログ

## 10. 提出物

1. **ソースコード** - このZIPファイル内の全ファイル
2. **マスク画像** - `out/mask/`フォルダ
3. **処理後画像** - `out/result/`フォルダ
4. **評価結果** - `out/eval/metrics.csv`
5. **このドキュメント** - 実装の説明

## お問い合わせ

問題が発生した場合は、`out/logs/processing.log`を確認してください。
