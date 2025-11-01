#前処理1枚

#!/usr/bin/env python3
"""
前処理で補正された画像を確認するスクリプト
マスクは作らず、補正後の画像のみを出力

使い方:
    python3 preview_preprocessed_images.py
"""

import cv2
import numpy as np
from pathlib import Path

class PreprocessingPreview:
    """前処理の各段階を確認"""
    
    def __init__(self, input_image, output_dir):
        self.input_image = input_image
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def apply_gamma_correction(self, image, gamma=0.7):
        """
        ガンマ補正
        gamma < 1.0: 明るくなる（暗い部分を強調）
        gamma > 1.0: 暗くなる（明るい部分を抑制）
        """
        # 正規化
        normalized = image.astype(np.float32) / 255.0
        # ガンマ補正を適用
        corrected = np.power(normalized, gamma)
        # 0-255にスケール
        return (corrected * 255).astype(np.uint8)
    
    def apply_clahe(self, image, clip_limit=1.0, tile_size=8):
        """CLAHE（コントラスト強調）"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                tileGridSize=(tile_size, tile_size))
        l_clahe = clahe.apply(l)
        
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    def apply_full_preprocessing(self, image):
        """ガンマ補正 + CLAHE の完全前処理"""
        # ステップ1: ガンマ補正
        gamma_corrected = self.apply_gamma_correction(image, gamma=0.7)
        
        # ステップ2: CLAHE
        fully_processed = self.apply_clahe(gamma_corrected, clip_limit=1.0)
        
        return gamma_corrected, fully_processed
    
    def create_comparison_grid(self, original, gamma, clahe):
        """3枚を並べた比較画像を作成"""
        # すべて同じサイズにリサイズ
        h, w = original.shape[:2]
        
        # 横に3枚並べる
        grid = np.hstack([original, gamma, clahe])
        
        # タイトルバーを追加
        title_height = 80
        title_bar = np.zeros((title_height, grid.shape[1], 3), dtype=np.uint8)
        
        # 各画像の上にタイトルを追加
        section_w = w
        cv2.putText(title_bar, "Original", (section_w//2 - 50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(title_bar, "Gamma Corrected", (section_w + section_w//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(title_bar, "Gamma + CLAHE", (2*section_w + section_w//2 - 90, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 統計情報を追加
        stats_bar = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
        
        # 各画像の明度の平均値を計算
        orig_mean = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        gamma_mean = np.mean(cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY))
        clahe_mean = np.mean(cv2.cvtColor(clahe, cv2.COLOR_BGR2GRAY))
        
        cv2.putText(stats_bar, f"Brightness: {orig_mean:.1f}", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(stats_bar, f"Brightness: {gamma_mean:.1f}", (section_w + 20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(stats_bar, f"Brightness: {clahe_mean:.1f}", (2*section_w + 20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 全部を結合
        final = np.vstack([title_bar, grid, stats_bar])
        
        return final
    
    def process_and_save(self):
        """処理を実行して保存"""
        print("=" * 70)
        print("前処理画像プレビュー")
        print("=" * 70)
        
        # 画像を読み込む
        original = cv2.imread(self.input_image)
        if original is None:
            print(f"❌ 画像を読み込めませんでした: {self.input_image}")
            return
        
        print(f"\n✅ 画像を読み込みました: {original.shape}")
        
        # 前処理を実行
        print("\n処理中...")
        gamma_corrected, fully_processed = self.apply_full_preprocessing(original)
        
        # 個別に保存
        print("\n📁 個別画像を保存中...")
        #cv2.imwrite(str(self.output_dir / "01_original.png"), original)
        #cv2.imwrite(str(self.output_dir / "02_gamma_corrected.png"), gamma_corrected)
        cv2.imwrite(str(self.output_dir / "03_fully_preprocessed.png"), fully_processed)
        
        print(f"   - 元画像: 01_original.png")
        print(f"   - ガンマ補正: 02_gamma_corrected.png")
        print(f"   - 完全前処理: 03_fully_preprocessed.png")
        
        # 比較画像を作成
        print("\n📊 比較画像を作成中...")
        comparison = self.create_comparison_grid(original, gamma_corrected, fully_processed)
        cv2.imwrite(str(self.output_dir / "00_comparison.png"), comparison)
        print(f"   - 比較画像: 00_comparison.png")
        
        # 統計情報を表示
        print("\n📈 画像統計:")
        print(f"   元画像の平均輝度: {np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)):.1f}")
        print(f"   ガンマ補正後: {np.mean(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)):.1f}")
        print(f"   完全前処理後: {np.mean(cv2.cvtColor(fully_processed, cv2.COLOR_BGR2GRAY)):.1f}")
        
        # レポートを作成
        self.create_report(original, gamma_corrected, fully_processed)
        
        print("\n" + "=" * 70)
        print("✅ 完了！")
        print("=" * 70)
        print(f"\n📁 出力先: {self.output_dir}")
        print("\n確認すべきファイル:")
        print("  1. 00_comparison.png  ← まずこれを見る（3枚並び）")
        print("  2. 03_fully_preprocessed.png ← 最終的な補正画像")
        print("  3. preprocessing_preview_report.txt ← 詳細情報")
        print("=" * 70)
    
    def create_report(self, original, gamma, clahe):
        """詳細レポートを作成"""
        report_path = self.output_dir / "preprocessing_preview_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("前処理画像プレビュー レポート\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("【画像情報】\n")
            f.write(f"  サイズ: {original.shape[1]} x {original.shape[0]}\n")
            f.write(f"  チャンネル数: {original.shape[2]}\n\n")
            
            f.write("【適用した処理】\n\n")
            
            f.write("  1. ガンマ補正（γ=0.5）\n")
            f.write("     - 暗い領域を明るくする\n")
            f.write("     - 反射を強調する\n")
            f.write("     - 数式: output = input^(1/0.5) = input^2\n\n")
            
            f.write("  2. CLAHE（clipLimit=3.0, tileSize=8x8）\n")
            f.write("     - 局所的なコントラストを強調\n")
            f.write("     - 画像を8x8のタイルに分割して処理\n")
            f.write("     - 水滴の微細な境界を明確化\n\n")
            
            f.write("【画像統計】\n\n")
            
            # 各画像の統計
            for name, img in [("元画像", original), 
                             ("ガンマ補正後", gamma), 
                             ("完全前処理後", clahe)]:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                f.write(f"  {name}:\n")
                f.write(f"    - 平均輝度: {np.mean(gray):.2f}\n")
                f.write(f"    - 標準偏差: {np.std(gray):.2f}\n")
                f.write(f"    - 最小値: {np.min(gray)}\n")
                f.write(f"    - 最大値: {np.max(gray)}\n\n")
            
            f.write("【出力ファイル】\n\n")
            f.write("  - 00_comparison.png         3枚を並べた比較画像\n")
            f.write("  - 01_original.png           元画像\n")
            f.write("  - 02_gamma_corrected.png    ガンマ補正後\n")
            f.write("  - 03_fully_preprocessed.png 完全前処理後（最終）\n\n")
            
            f.write("【使い方】\n\n")
            f.write("  この前処理後の画像は:\n")
            f.write("  1. 水滴検出の前段階として使用\n")
            f.write("  2. 画質改善の効果確認\n")
            f.write("  3. パラメータ調整の参考\n\n")
            
            f.write("【次のステップ】\n\n")
            f.write("  この補正画像に満足したら:\n")
            f.write("  → improved_preprocessing.py で水滴検出を実行\n")
            f.write("  → マスクを生成\n")
            f.write("  → remove_reflections.py で反射除去\n\n")
            
            f.write("=" * 70 + "\n")


def main():
    """メイン処理"""
    print("\n" + "=" * 70)
    print("🎨 前処理画像プレビュースクリプト")
    print("=" * 70)
    print("\nこのスクリプトは:")
    print("  ✓ ガンマ補正とCLAHEを適用した画像を生成")
    print("  ✓ 元画像と補正画像を比較")
    print("  ✓ マスクは作らず、補正画像のみ出力")
    print("=" * 70)
    
    # パスを設定
    input_image = "/workspace/data/images/100.png"  # ← 画像パスを変更可能
    output_dir = "/workspace/results"
    
    # プレビューを実行
    preview = PreprocessingPreview(input_image, output_dir)
    #preview.process_and_save()


if __name__ == "__main__":
    main()
