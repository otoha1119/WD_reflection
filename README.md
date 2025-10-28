# Green Container Reflection Detection and Removal System

## 概要

暗所で撮影された緑色プラスチックコンテナ画像から水滴・鏡面反射を自動検出し、除去するPythonプロジェクトです。MVCアーキテクチャで設計され、堅牢なアルゴリズムにより高精度な反射除去を実現します。

## 特徴

- **高精度な反射検出**: 局所Zスコアと色情報を組み合わせた適応的検出
- **形状別処理**: 線状反射と塊状反射を分類し、それぞれに最適な処理を適用
- **Lab色空間でのインペインティング**: 明度と色を分離処理することで自然な補間
- **包括的な評価指標**: SPP、EPR、Colorfulness、SSIM等による定量評価
- **エラー耐性**: 自動的なパラメータ調整とフォールバック機構

## システム要件

- Python 3.8以上
- OpenCV 4.8.0以上
- NumPy 1.23.0以上
- その他の依存関係は`requirements.txt`参照

## インストール

```bash
# リポジトリのクローン
git clone <repository_url>
cd workspace

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 基本的な実行

```bash
# デフォルト設定でバッチ処理
python -m app --input data/images

# 設定ファイルを指定して実行
python -m app --input data/images --config configs/config.yaml

# 出力ディレクトリを個別指定
python -m app \
  --input data/images \
  --outmask out/mask \
  --outresult out/result \
  --outeval out/eval \
  --outlogs out/logs
```

### 単一画像の処理

```bash
python -m app --single path/to/image.png --config configs/config.yaml
```

### デバッグモード

```bash
# 詳細なログと可視化を有効化
python -m app --input data/images --debug --verbose
```

## ディレクトリ構成

```
workspace/
├── app/                      # Pythonパッケージ（MVCアーキテクチャ）
│   ├── controllers/         # 制御ロジック
│   │   ├── cli.py          # CLIインターフェース
│   │   └── pipeline.py     # 処理パイプライン
│   ├── models/             # コアアルゴリズム
│   │   ├── container_mask.py  # コンテナ領域抽出
│   │   ├── highlight_detect.py # 反射候補検出
│   │   ├── shape_classify.py   # 形状分類
│   │   ├── inpaint.py         # インペインティング
│   │   └── metrics.py          # 評価指標
│   ├── views/              # 出力処理
│   │   ├── writers.py      # ファイル保存
│   │   └── viz.py          # 可視化
│   └── utils/              # ユーティリティ
│       ├── io.py           # 画像入出力
│       └── config.py       # 設定管理
├── configs/
│   └── config.yaml         # 設定ファイル
├── data/
│   ├── images/            # 入力画像
│   └── samples/           # サンプル画像
├── out/                   # 出力ディレクトリ
│   ├── mask/             # 反射マスク
│   ├── result/           # 処理後画像
│   ├── eval/             # 評価結果
│   └── logs/             # ログファイル
└── README.md             # このファイル
```

## アルゴリズムの詳細

### 1. コンテナ領域抽出

- Lab色空間のa*チャンネルで緑色を検出
- 回転矩形フィッティングで安定した領域抽出
- 境界安全帯により周辺反射を除外

### 2. 反射候補検出

- **局所Zスコア**: ガウシアンブラーによる局所統計量から相対的な明るさを検出
- **白っぽさゲート**: 低彩度または低RGB範囲の画素を選択
- **白飛び強制検出**: RGB値245以上を確実に検出

### 3. 形状分類

- **Thin（線状）**: アスペクト比 > 4.0、面積 < 400px
- **Blob（塊状）**: それ以外の形状
- 形状に応じた可変膨張処理

### 4. インペインティング

- **Lチャンネル**: 
  - Thin → Navier-Stokes法
  - Blob → Telea法
- **a,bチャンネル**: Blob領域のみ弱いTelea法
- **境界フェザリング**: 距離変換による自然な境界

### 5. 評価指標

- **SPP** (Saturated Pixel Percentage): 飽和画素の割合
- **EPR** (Edge Preservation Ratio): エッジ保存率（目標 ≥ 0.9）
- **Colorfulness**: Hasler-Süsstrunk法による色彩度
- **SSIM**: 構造類似性
- **PSNR**: マスク領域でのピーク信号対雑音比

## 設定ファイル (config.yaml)

主要なパラメータ：

```yaml
container:
  a_threshold: 126      # 緑色検出閾値（Lab a*）
  erode_iter: 1        # 安全帯のエロージョン回数

detect:
  z_sigma: 11          # 局所統計のガウシアンσ
  z_thresh: 2.0        # Zスコア閾値
  sat_cut: 245         # 飽和判定閾値
  s_thresh: 40         # 彩度閾値
  rgb_range_thresh: 25 # RGB範囲閾値

shape:
  thin_min_short: 8    # 線状判定の最小短辺
  thin_aspect_min: 4.0 # 線状判定の最小アスペクト比
  thin_area_max: 400   # 線状判定の最大面積
  dilate_thin: 1       # 線状膨張回数
  dilate_blob: 2       # 塊状膨張回数

inpaint:
  radius: 3            # インペイント半径
  feather: 2           # フェザリング幅
```

## エラー対処

### コンテナ検出失敗時
- a*閾値を自動的に+1ずつ緩和（最大3回）
- 低彩度＆中暗度を併用した適応的検出

### 反射検出が弱い場合
- z_threshを1.8まで段階的に緩和
- s_threshを50まで段階的に緩和
- 検出履歴をログに記録

### 過検出（白地ラベル等）
- 大面積低勾配成分を自動除外
- 境界距離3px未満を除外

## 出力ファイル

1. **マスク画像** (`out/mask/`)
   - 8ビット単一チャンネルPNG（0/255）
   - 検出された反射領域を白で表示

2. **処理後画像** (`out/result/`)
   - BGRフォーマットPNG/JPEG
   - 反射除去後の画像

3. **評価結果** (`out/eval/`)
   - `metrics.csv`: 全画像の評価指標
   - `panels/`: Before/Mask/Afterの比較画像
   - `summary.txt`: 処理結果サマリー

4. **ログ** (`out/logs/`)
   - `processing.log`: 詳細な処理ログ

## トラブルシューティング

### メモリ不足エラー
- 大きな画像は自動的にリサイズされます
- `--single`オプションで1枚ずつ処理

### 処理速度が遅い
- デバッグ可視化を無効化: `--no-panels`
- ログレベルを下げる: `--quiet`

### 検出精度が低い
- `config.yaml`のパラメータを調整
- `--debug`モードで中間結果を確認

## 性能目標

- SPP削減率: > 50%
- EPR: ≥ 0.9
- Colorfulness保持率: 0.8 - 1.2
- 処理時間: < 2秒/画像（1920x1080）

## ライセンス

[ライセンス情報をここに記載]

## 作者

Reflection Removal System Development Team

## 更新履歴

- v1.0.0 (2024-XX-XX): 初回リリース
  - MVCアーキテクチャによる実装
  - 局所Zスコアベースの検出
  - 形状別インペインティング
  - 包括的な評価指標

## 参考文献

- Telea, A. (2004). An image inpainting technique based on the fast marching method
- Bertalmio, M. et al. (2001). Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting
- Hasler, D., & Süsstrunk, S. (2003). Measuring colorfulness in natural images

## サポート

問題が発生した場合は、以下を確認してください：
1. エラーログ (`out/logs/processing.log`)
2. デバッグ画像 (`out/eval/debug/`)
3. 設定ファイルのパラメータ

それでも解決しない場合は、Issueを作成してください。
