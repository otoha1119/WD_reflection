import cv2
import numpy as np
import os
from pathlib import Path

class ImprovedWaterDropletPreprocessor:
    """改善版：水滴反射検出のための前処理システム"""
    
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
    
    def gamma_correction(self, image, gamma=0.5):
        """ガンマ補正で反射を強調（より強く）"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in range(256)]).astype("uint8")
        corrected = cv2.LUT(image, table)
        self.results['gamma_correction'] = corrected
        print(f"ガンマ補正完了 (γ={gamma})")
        return corrected
    
    def apply_clahe(self, image, clip_limit=3.0, tile_size=8):
        """CLAHE（局所コントラスト強調）を適用"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                tileGridSize=(tile_size, tile_size))
        l_clahe = clahe.apply(l)
        
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        self.results['clahe'] = result
        print(f"CLAHE完了 (clipLimit={clip_limit}, tileSize={tile_size}x{tile_size})")
        return result
    
    def tophat_transform(self, image, kernel_size=7):
        """Top-hat変換で明るい反射を抽出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (kernel_size, kernel_size))
        
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        self.results['tophat'] = tophat
        self.results['blackhat'] = blackhat
        print(f"Top-hat変換完了 (kernel_size={kernel_size})")
        return tophat, blackhat
    
    def adaptive_specular_detection(self, image):
        """適応的な鏡面反射検出"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 画像の明度統計を計算
        v_mean = np.mean(v)
        v_std = np.std(v)
        
        # 適応的な閾値を設定
        v_threshold = max(v_mean + 1.5 * v_std, 150)  # 最低でも150
        s_threshold = 80  # 彩度の閾値を緩和
        
        # 鏡面反射マスク
        specular_mask = ((v > v_threshold) & (s < s_threshold)).astype(np.uint8) * 255
        
        print(f"適応的鏡面反射検出: V閾値={v_threshold:.1f}, S閾値={s_threshold}")
        print(f"  検出ピクセル数: {np.sum(specular_mask > 0):,}")
        
        self.results['adaptive_specular_mask'] = specular_mask
        self.results['v_channel'] = v
        return specular_mask
    
    def improved_comprehensive_pipeline(self, image):
        """改善版：包括的な前処理パイプライン"""
        print("\n=== 改善版前処理パイプライン開始 ===")
        
        # ステップ1: より強いガンマ補正
        gamma_corrected = self.gamma_correction(image, gamma=0.5)
        
        # ステップ2: より強いCLAHE
        clahe_enhanced = self.apply_clahe(gamma_corrected, clip_limit=3.0, tile_size=8)
        
        # ステップ3: Top-hat変換（小さいカーネル）
        tophat, blackhat = self.tophat_transform(clahe_enhanced, kernel_size=7)
        
        # ステップ4: 適応的鏡面反射検出
        specular_mask = self.adaptive_specular_detection(clahe_enhanced)
        
        # ステップ5: Top-hatからマスクを作成（より緩い閾値）
        # 平均値ベースの適応的閾値
        tophat_mean = np.mean(tophat)
        tophat_std = np.std(tophat)
        tophat_threshold = max(tophat_mean + 1.0 * tophat_std, 10)
        
        _, tophat_mask = cv2.threshold(tophat, tophat_threshold, 255, cv2.THRESH_BINARY)
        
        print(f"Top-hat閾値: {tophat_threshold:.1f}")
        print(f"  検出ピクセル数: {np.sum(tophat_mask > 0):,}")
        
        self.results['tophat_mask'] = tophat_mask
        
        # ステップ6: マスクを統合
        # OR演算で統合（より多くの領域を保持）
        combined_mask = cv2.bitwise_or(tophat_mask, specular_mask)
        
        # 軽いモルフォロジー演算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, 
                                        kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, 
                                        kernel, iterations=2)
        
        # 小さすぎる領域を除去
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8)
        
        # 面積が最小閾値以上の領域のみ保持
        min_area = 20  # 最小面積を小さく設定
        filtered_mask = np.zeros_like(combined_mask)
        for i in range(1, num_labels):  # 0はバックグラウンド
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 255
        
        print(f"統合マスク作成完了")
        print(f"  最終検出ピクセル数: {np.sum(filtered_mask > 0):,}")
        
        self.results['combined_mask'] = combined_mask
        self.results['filtered_mask'] = filtered_mask
        
        return filtered_mask
    
    def create_comparison_visualization(self, output_dir):
        """詳細な比較可視化を作成"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # オリジナル画像
        cv2.imwrite(str(output_path / "01_original.png"), self.original_image)
        
        # ガンマ補正
        if 'gamma_correction' in self.results:
            cv2.imwrite(str(output_path / "02_gamma.png"), 
                       self.results['gamma_correction'])
        
        # CLAHE
        if 'clahe' in self.results:
            cv2.imwrite(str(output_path / "03_clahe.png"), 
                       self.results['clahe'])
        
        # Top-hat（ヒートマップ付き）
        if 'tophat' in self.results:
            tophat = self.results['tophat']
            cv2.imwrite(str(output_path / "04_tophat.png"), tophat)
            colored = cv2.applyColorMap(tophat, cv2.COLORMAP_JET)
            cv2.imwrite(str(output_path / "04_tophat_heatmap.png"), colored)
        
        # Top-hatマスク
        if 'tophat_mask' in self.results:
            cv2.imwrite(str(output_path / "05_tophat_mask.png"), 
                       self.results['tophat_mask'])
        
        # 適応的鏡面反射マスク
        if 'adaptive_specular_mask' in self.results:
            cv2.imwrite(str(output_path / "06_specular_mask.png"), 
                       self.results['adaptive_specular_mask'])
        
        # 統合マスク
        if 'combined_mask' in self.results:
            cv2.imwrite(str(output_path / "07_combined_mask.png"), 
                       self.results['combined_mask'])
        
        # フィルタリング後のマスク（最終結果）
        if 'filtered_mask' in self.results:
            cv2.imwrite(str(output_path / "08_final_mask.png"), 
                       self.results['filtered_mask'])
        
        # オーバーレイ可視化
        self._create_overlay_visualizations(output_path)
        
        print(f"\n結果を保存しました: {output_dir}")
        return output_path
    
    def _create_overlay_visualizations(self, output_path):
        """オーバーレイ可視化を作成"""
        if 'filtered_mask' not in self.results:
            return
        
        mask = self.results['filtered_mask']
        
        # マスクを赤色で可視化
        overlay_red = self.original_image.copy()
        mask_colored = np.zeros_like(self.original_image)
        mask_colored[:, :, 2] = mask  # Red channel
        overlay_red = cv2.addWeighted(overlay_red, 0.7, mask_colored, 0.3, 0)
        cv2.imwrite(str(output_path / "09_overlay_red.png"), overlay_red)
        
        # マスクを緑色で可視化
        overlay_green = self.original_image.copy()
        mask_colored = np.zeros_like(self.original_image)
        mask_colored[:, :, 1] = mask  # Green channel
        overlay_green = cv2.addWeighted(overlay_green, 0.7, mask_colored, 0.3, 0)
        cv2.imwrite(str(output_path / "10_overlay_green.png"), overlay_green)
        
        # 輪郭のみを可視化
        contour_vis = self.original_image.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
        
        # 検出された水滴の数を表示
        cv2.putText(contour_vis, f"Detected: {len(contours)} regions", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(output_path / "11_contours.png"), contour_vis)
        
        print(f"検出された領域数: {len(contours)}")
    
    def generate_detailed_report(self, output_dir):
        """詳細なレポートを生成"""
        report_path = Path(output_dir) / "detailed_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("改善版 水滴反射検出 前処理レポート\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("【1. 画像情報】\n")
            f.write(f"  - サイズ: {self.original_image.shape[1]} x {self.original_image.shape[0]}\n")
            f.write(f"  - ピクセル数: {self.original_image.shape[0] * self.original_image.shape[1]:,}\n\n")
            
            f.write("【2. 改善された前処理手法】\n\n")
            
            f.write("  a) 強化されたガンマ補正 (γ=0.5)\n")
            f.write("     - 標準パラメータ(γ=0.6)より強く、暗い反射も検出可能に\n")
            f.write("     - 反射のダイナミックレンジを最大限に拡大\n\n")
            
            f.write("  b) 強化されたCLAHE (clipLimit=3.0)\n")
            f.write("     - より高いclipLimitで強力なコントラスト強調\n")
            f.write("     - 微細な水滴の境界を明確化\n\n")
            
            f.write("  c) 最適化されたTop-hat変換 (kernel=7)\n")
            f.write("     - 小さいカーネルで小さな水滴にも対応\n")
            f.write("     - 明るい反射を背景から効果的に分離\n\n")
            
            f.write("  d) 適応的鏡面反射検出\n")
            f.write("     - 画像の明度統計に基づく動的な閾値設定\n")
            f.write("     - V閾値 = max(mean + 1.5*std, 150)\n")
            f.write("     - S閾値 = 80（緩和された彩度条件）\n\n")
            
            if 'v_channel' in self.results:
                v = self.results['v_channel']
                f.write(f"     統計情報:\n")
                f.write(f"       - V平均値: {np.mean(v):.1f}\n")
                f.write(f"       - V標準偏差: {np.std(v):.1f}\n")
                f.write(f"       - V最大値: {np.max(v)}\n")
                f.write(f"       - V最小値: {np.min(v)}\n\n")
            
            f.write("  e) 改善された統合処理\n")
            f.write("     - Top-hatと鏡面反射マスクをOR結合\n")
            f.write("     - 軽いモルフォロジー演算（ノイズ除去と形状補正）\n")
            f.write("     - 面積フィルタリング（最小面積=20ピクセル）\n\n")
            
            f.write("【3. 検出結果の詳細】\n\n")
            
            if 'tophat_mask' in self.results:
                tophat_pixels = np.sum(self.results['tophat_mask'] > 0)
                f.write(f"  Top-hat検出:\n")
                f.write(f"    - ピクセル数: {tophat_pixels:,}\n")
                f.write(f"    - 割合: {(tophat_pixels / (self.original_image.shape[0] * self.original_image.shape[1])) * 100:.2f}%\n\n")
            
            if 'adaptive_specular_mask' in self.results:
                spec_pixels = np.sum(self.results['adaptive_specular_mask'] > 0)
                f.write(f"  鏡面反射検出:\n")
                f.write(f"    - ピクセル数: {spec_pixels:,}\n")
                f.write(f"    - 割合: {(spec_pixels / (self.original_image.shape[0] * self.original_image.shape[1])) * 100:.2f}%\n\n")
            
            if 'filtered_mask' in self.results:
                final_pixels = np.sum(self.results['filtered_mask'] > 0)
                total_pixels = self.original_image.shape[0] * self.original_image.shape[1]
                
                # 領域数をカウント
                contours, _ = cv2.findContours(self.results['filtered_mask'], 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                
                f.write(f"  最終統合結果:\n")
                f.write(f"    - ピクセル数: {final_pixels:,}\n")
                f.write(f"    - 全体の割合: {(final_pixels / total_pixels) * 100:.2f}%\n")
                f.write(f"    - 検出領域数: {len(contours)}\n\n")
                
                if len(contours) > 0:
                    areas = [cv2.contourArea(c) for c in contours]
                    f.write(f"  領域サイズ統計:\n")
                    f.write(f"    - 最大面積: {max(areas):.0f} ピクセル\n")
                    f.write(f"    - 最小面積: {min(areas):.0f} ピクセル\n")
                    f.write(f"    - 平均面積: {np.mean(areas):.1f} ピクセル\n")
                    f.write(f"    - 中央値面積: {np.median(areas):.1f} ピクセル\n\n")
            
            f.write("【4. 出力ファイル一覧】\n")
            f.write("  - 01_original.png: オリジナル画像\n")
            f.write("  - 02_gamma.png: ガンマ補正後\n")
            f.write("  - 03_clahe.png: CLAHE適用後\n")
            f.write("  - 04_tophat.png / 04_tophat_heatmap.png: Top-hat変換\n")
            f.write("  - 05_tophat_mask.png: Top-hatベースマスク\n")
            f.write("  - 06_specular_mask.png: 適応的鏡面反射マスク\n")
            f.write("  - 07_combined_mask.png: 統合マスク\n")
            f.write("  - 08_final_mask.png: フィルタリング後の最終マスク\n")
            f.write("  - 09_overlay_red.png: 赤色オーバーレイ\n")
            f.write("  - 10_overlay_green.png: 緑色オーバーレイ\n")
            f.write("  - 11_contours.png: 輪郭可視化\n\n")
            
            f.write("【5. 推奨される次のステップ】\n")
            f.write("  1. OpenCVのinpainting機能を使用した反射除去\n")
            f.write("     - cv2.inpaint()を使用\n")
            f.write("     - INPAINT_TEICHまたはINPAINT_NSメソッド\n\n")
            f.write("  2. パラメータの微調整\n")
            f.write("     - ガンマ値: 0.4～0.6の範囲で実験\n")
            f.write("     - 最小面積: 15～30の範囲で調整\n\n")
            f.write("  3. 評価指標の選定\n")
            f.write("     - PSNR（Peak Signal-to-Noise Ratio）\n")
            f.write("     - SSIM（Structural Similarity Index）\n")
            f.write("     - 反射領域の減少率\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"詳細レポートを生成しました: {report_path}")
        return report_path


def main():
    """メイン処理"""
    input_image = "/mnt/project/2.png"
    output_dir = "/mnt/user-data/outputs/improved_preprocessing_results"
    
    preprocessor = ImprovedWaterDropletPreprocessor()
    
    try:
        print("=" * 80)
        print("改善版 水滴反射検出 前処理システム")
        print("=" * 80)
        
        # 画像を読み込む
        image = preprocessor.load_image(input_image)
        
        # 改善版パイプラインを実行
        final_mask = preprocessor.improved_comprehensive_pipeline(image)
        
        # 結果を可視化
        output_path = preprocessor.create_comparison_visualization(output_dir)
        
        # 詳細レポート生成
        report_path = preprocessor.generate_detailed_report(output_dir)
        
        print("\n" + "=" * 80)
        print("処理が完了しました！")
        print("=" * 80)
        print(f"\n出力ディレクトリ: {output_dir}")
        print(f"詳細レポート: {report_path}")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
