import cv2
import numpy as np
import os
from pathlib import Path

class WaterDropletPreprocessor:
    """水滴反射検出のための前処理システム"""
    
    def __init__(self):
        """初期化"""
        self.original_image = None
        self.results = {}
        
    def load_image(self, image_path):
        """画像を読み込む"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"画像を読み込めませんでした: {image_path}")
        print(f"画像を読み込みました: {self.original_image.shape}")
        return self.original_image
    
    def gamma_correction(self, image, gamma=0.6):
        """ガンマ補正で反射を強調"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in range(256)]).astype("uint8")
        corrected = cv2.LUT(image, table)
        self.results['gamma_correction'] = corrected
        print(f"ガンマ補正完了 (γ={gamma})")
        return corrected
    
    def apply_clahe(self, image, clip_limit=2.5, tile_size=8):
        """CLAHE（局所コントラスト強調）を適用"""
        # LAB色空間に変換
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Lチャネルに適用
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                tileGridSize=(tile_size, tile_size))
        l_clahe = clahe.apply(l)
        
        # 結合して戻す
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        self.results['clahe'] = result
        print(f"CLAHE完了 (clipLimit={clip_limit}, tileSize={tile_size}x{tile_size})")
        return result
    
    def tophat_transform(self, image, kernel_size=9):
        """Top-hat変換で明るい反射を抽出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 楕円形の構造化要素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (kernel_size, kernel_size))
        
        # White top-hat
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Black-hat
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # 結合
        combined = tophat + (255 - blackhat)
        
        self.results['tophat'] = tophat
        self.results['blackhat'] = blackhat
        self.results['tophat_combined'] = combined
        print(f"Top-hat変換完了 (kernel_size={kernel_size})")
        return tophat, blackhat, combined
    
    def difference_of_gaussians(self, image, sigma1=1.0, sigma2=1.6):
        """DoG（ガウシアンの差分）で反射エッジを強調"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2つの異なるσでぼかし
        gaussian1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
        gaussian2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
        
        # 差分を計算
        dog = cv2.subtract(gaussian1, gaussian2)
        self.results['dog'] = dog
        print(f"DoG完了 (σ1={sigma1}, σ2={sigma2})")
        return dog
    
    def hsv_analysis(self, image):
        """HSV色空間で水滴を分析"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 高輝度かつ低彩度（鏡面反射の特徴）
        specular_mask = ((v > 200) & (s < 50)).astype(np.uint8) * 255
        
        # 水滴検出用マスク
        lower_hsv = np.array([90, 30, 50])
        upper_hsv = np.array([140, 255, 255])
        water_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        self.results['hsv_h'] = h
        self.results['hsv_s'] = s
        self.results['hsv_v'] = v
        self.results['specular_mask'] = specular_mask
        self.results['water_mask'] = water_mask
        print("HSV分析完了")
        return specular_mask, water_mask
    
    def comprehensive_pipeline(self, image):
        """包括的な前処理パイプライン"""
        print("\n=== 包括的前処理パイプライン開始 ===")
        
        # ステップ1: ガンマ補正
        gamma_corrected = self.gamma_correction(image, gamma=0.6)
        
        # ステップ2: CLAHE
        clahe_enhanced = self.apply_clahe(gamma_corrected, clip_limit=2.5, tile_size=8)
        
        # ステップ3: Top-hat変換
        tophat, blackhat, combined = self.tophat_transform(clahe_enhanced, kernel_size=9)
        
        # ステップ4: DoG
        dog = self.difference_of_gaussians(clahe_enhanced, sigma1=1.0, sigma2=1.6)
        
        # ステップ5: HSV分析
        specular_mask, water_mask = self.hsv_analysis(clahe_enhanced)
        
        # ステップ6: 統合マスク作成
        # Top-hatとHSV分析を組み合わせる
        _, tophat_binary = cv2.threshold(tophat, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        integrated_mask = cv2.bitwise_or(tophat_binary, specular_mask)
        integrated_mask = cv2.bitwise_or(integrated_mask, water_mask)
        
        # モルフォロジー演算でクリーンアップ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        integrated_mask = cv2.morphologyEx(integrated_mask, cv2.MORPH_OPEN, 
                                          kernel, iterations=2)
        integrated_mask = cv2.morphologyEx(integrated_mask, cv2.MORPH_CLOSE, 
                                          kernel, iterations=2)
        
        self.results['integrated_mask'] = integrated_mask
        print("統合マスク作成完了")
        
        return integrated_mask
    
    def create_visualization(self, output_dir):
        """結果を可視化して保存"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # オリジナル画像を保存
        cv2.imwrite(str(output_path / "01_original.png"), self.original_image)
        
        # 各処理結果を保存
        save_list = [
            ('02_gamma_correction', 'gamma_correction'),
            ('03_clahe', 'clahe'),
            ('04_tophat', 'tophat'),
            ('05_blackhat', 'blackhat'),
            ('06_tophat_combined', 'tophat_combined'),
            ('07_dog', 'dog'),
            ('08_hsv_h', 'hsv_h'),
            ('09_hsv_s', 'hsv_s'),
            ('10_hsv_v', 'hsv_v'),
            ('11_specular_mask', 'specular_mask'),
            ('12_water_mask', 'water_mask'),
            ('13_integrated_mask', 'integrated_mask'),
        ]
        
        for filename, key in save_list:
            if key in self.results:
                result = self.results[key]
                # グレースケール画像をカラーマップで可視化
                if len(result.shape) == 2:
                    # 熱マップ（色付き）で保存
                    colored = cv2.applyColorMap(result, cv2.COLORMAP_JET)
                    cv2.imwrite(str(output_path / f"{filename}_heatmap.png"), colored)
                    # グレースケールでも保存
                    cv2.imwrite(str(output_path / f"{filename}.png"), result)
                else:
                    cv2.imwrite(str(output_path / f"{filename}.png"), result)
        
        # 統合マスクをオリジナル画像に重ねて表示
        if 'integrated_mask' in self.results:
            overlay = self.original_image.copy()
            mask_colored = cv2.cvtColor(self.results['integrated_mask'], 
                                       cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 0] = 0  # Blue channel
            mask_colored[:, :, 1] = self.results['integrated_mask']  # Green channel
            mask_colored[:, :, 2] = 0  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            cv2.imwrite(str(output_path / "14_overlay_mask.png"), overlay)
        
        print(f"\n結果を保存しました: {output_dir}")
        return output_path
    
    def generate_report(self, output_dir):
        """処理結果のレポートを生成"""
        report_path = Path(output_dir) / "preprocessing_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("水滴反射検出 前処理レポート\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. 画像情報\n")
            f.write(f"   - サイズ: {self.original_image.shape[1]} x {self.original_image.shape[0]}\n")
            f.write(f"   - チャンネル数: {self.original_image.shape[2]}\n\n")
            
            f.write("2. 適用した前処理手法\n")
            f.write("   a) ガンマ補正 (γ=0.6)\n")
            f.write("      - 暗い領域を明るくし、反射のダイナミックレンジを拡大\n")
            f.write("      - 水滴の鏡面反射を強調\n\n")
            
            f.write("   b) CLAHE（Contrast Limited Adaptive Histogram Equalization）\n")
            f.write("      - clipLimit=2.5, tileSize=8x8\n")
            f.write("      - 局所的なコントラストを強調し、水滴の微細な境界を強調\n\n")
            
            f.write("   c) Top-hat変換\n")
            f.write("      - kernel_size=9（楕円形）\n")
            f.write("      - 明るい反射部分を背景から分離\n")
            f.write("      - Black-hatで暗い境界も抽出\n\n")
            
            f.write("   d) DoG（Difference of Gaussians）\n")
            f.write("      - σ1=1.0, σ2=1.6\n")
            f.write("      - 反射のエッジを精密に検出\n\n")
            
            f.write("   e) HSV色空間分析\n")
            f.write("      - 高輝度（V>200）かつ低彩度（S<50）で鏡面反射を検出\n")
            f.write("      - H[90-140], S[30-255], V[50-255]で水滴領域を検出\n\n")
            
            f.write("3. 統合マスク生成\n")
            f.write("   - Top-hat、HSV鏡面反射、HSV水滴検出を統合\n")
            f.write("   - モルフォロジー演算（Opening + Closing）でノイズ除去\n\n")
            
            if 'integrated_mask' in self.results:
                mask = self.results['integrated_mask']
                detected_pixels = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                percentage = (detected_pixels / total_pixels) * 100
                
                f.write("4. 検出結果\n")
                f.write(f"   - 検出ピクセル数: {detected_pixels:,}\n")
                f.write(f"   - 全ピクセル数: {total_pixels:,}\n")
                f.write(f"   - 検出割合: {percentage:.2f}%\n\n")
            
            f.write("5. 出力ファイル\n")
            f.write("   - 01_original.png: オリジナル画像\n")
            f.write("   - 02_gamma_correction.png: ガンマ補正後\n")
            f.write("   - 03_clahe.png: CLAHE適用後\n")
            f.write("   - 04_tophat.png: White top-hat変換\n")
            f.write("   - 05_blackhat.png: Black-hat変換\n")
            f.write("   - 06_tophat_combined.png: Top-hat統合\n")
            f.write("   - 07_dog.png: DoG結果\n")
            f.write("   - 08~10_hsv_*.png: HSVチャンネル\n")
            f.write("   - 11_specular_mask.png: 鏡面反射マスク\n")
            f.write("   - 12_water_mask.png: 水滴検出マスク\n")
            f.write("   - 13_integrated_mask.png: 統合マスク（最終結果）\n")
            f.write("   - 14_overlay_mask.png: オリジナル画像に重ねて表示\n\n")
            
            f.write("=" * 70 + "\n")
        
        print(f"レポートを生成しました: {report_path}")
        return report_path


def main():
    """メイン処理"""
    # 画像パスを設定
    input_image = "/mnt/project/2.png"
    output_dir = "/mnt/user-data/outputs/preprocessing_results"
    
    # 前処理器を初期化
    preprocessor = WaterDropletPreprocessor()
    
    try:
        # 画像を読み込む
        print("=" * 70)
        print("水滴反射検出 前処理システム")
        print("=" * 70)
        image = preprocessor.load_image(input_image)
        
        # 包括的な前処理パイプラインを実行
        integrated_mask = preprocessor.comprehensive_pipeline(image)
        
        # 結果を可視化して保存
        output_path = preprocessor.create_visualization(output_dir)
        
        # レポート生成
        report_path = preprocessor.generate_report(output_dir)
        
        print("\n" + "=" * 70)
        print("処理が完了しました！")
        print("=" * 70)
        print(f"\n出力ディレクトリ: {output_dir}")
        print(f"レポート: {report_path}")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
