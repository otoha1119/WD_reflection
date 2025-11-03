"""
evaluation_controller
=====================

Controller for evaluating reflection removal quality across a batch
of images.  This module orchestrates the evaluation workflow:
reading original and processed images, generating box masks,
computing metrics via the evaluation model, aggregating statistics,
and producing visualizations.

The controller generates a comprehensive report including mean values
and distributions for all metrics.  Results are printed to the
terminal and distribution plots are saved to the ``Evaluation``
directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

from app.model.evaluation_model import EvaluationModel
from app.model.box_mask_model import BoxMaskModel


def process_images(
    original_dir: Path,
    result_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    start_image: int = 1,      # ★追加：開始画像番号
    end_image: int = 120,      # ★追加：終了画像番号
) -> None:
    """Evaluate reflection removal for all image pairs.

    Parameters
    ----------
    original_dir : pathlib.Path
        Directory containing original images (before processing).
    result_dir : pathlib.Path
        Directory containing processed images (after reflection removal).
    mask_dir : pathlib.Path
        Directory containing reflection masks.
    output_dir : pathlib.Path
        Directory where evaluation results and plots will be saved.
    start_image : int, optional
        Starting image number (inclusive). Default is 1.
    end_image : int, optional
        Ending image number (inclusive). Default is 120.
    """
    if not original_dir.is_dir():
        raise NotADirectoryError(f"Original images directory not found: {original_dir}")
    if not result_dir.is_dir():
        raise NotADirectoryError(f"Result images directory not found: {result_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {mask_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    eval_model = EvaluationModel(highlight_threshold=200, local_window_size=16)
    box_model = BoxMaskModel()
    
    # Storage for all metrics
    all_metrics: Dict[str, List[float]] = {
        'HLR': [],
        'Cov': [],
        'Brightness_Reduction': [],
        'Variance_Improvement': [],
    }
    
    processed_count = 0
    
    # ★ファイル名から数字を抽出する関数
    def extract_number(path: Path) -> int:
        match = re.search(r'(\d+)', path.stem)
        return int(match.group(1)) if match else 0
    
    # ★数値順にソートして処理
    image_files = [p for p in original_dir.iterdir() 
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
    image_files.sort(key=extract_number)
    
    # Process each image
    for img_path in image_files:
        # ★ファイル番号を取得
        img_num = extract_number(img_path)
        
        # ★指定範囲外はスキップ
        if img_num < start_image or img_num > end_image:
            continue
        
        stem = img_path.stem
        
        # Check if corresponding result exists
        result_path = result_dir / f"{stem}_reflection_removed.png"
        if not result_path.exists():
            print(f"Warning: result not found for {img_path.name}, skipping.", file=sys.stderr)
            continue
        
        # Check if corresponding mask exists
        mask_path = mask_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            print(f"Warning: mask not found for {img_path.name}, skipping.", file=sys.stderr)
            continue
        
        # Load images
        img_before = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_after = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if img_before is None or img_after is None or mask is None:
            print(f"Warning: failed to load images for {img_path.name}, skipping.", file=sys.stderr)
            continue
        
        # Generate box mask for this image
        box_mask, _ = box_model.generate_mask(img_before)
        
        # Evaluate
        metrics = eval_model.evaluate(img_before, img_after, mask, box_mask)
        
        # Store metrics
        for key, value in metrics.items():
            all_metrics[key].append(value)
        
        processed_count += 1
        #print(f"Evaluated: {img_path.name}")
    
    if processed_count == 0:
        print("No images were processed.", file=sys.stderr)
        return
    
    # Compute statistics
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS (Images {start_image}-{end_image})")  # ★範囲を表示
    print("=" * 60)
    print(f"Total images processed: {processed_count}")
    print("-" * 60)
    
    stats: Dict[str, Dict[str, float]] = {}
    
    for metric_name, values in all_metrics.items():
        if not values:
            continue
        
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        stats[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
        }
        
        # Print results
        print(f"\n{metric_name}:")
        print(f"  Mean:   {mean_val:6.2f}%")
        print(f"  Std:    {std_val:6.2f}%")
        print(f"  Min:    {min_val:6.2f}%")
        print(f"  Max:    {max_val:6.2f}%")
    
    print("\n" + "=" * 60)
    
    # Generate distribution plots
    _generate_distribution_plots(all_metrics, output_dir)
    print(f"\nDistribution plots saved to: {output_dir}")


def _generate_distribution_plots(
    metrics: Dict[str, List[float]],
    output_dir: Path,
) -> None:
    """Generate histogram plots for all metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary mapping metric names to lists of values.
    output_dir : pathlib.Path
        Directory where plots will be saved.
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create a 2x2 subplot for all four metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reflection Removal Evaluation Metrics Distribution', fontsize=16, fontweight='bold')
    
    metric_names = ['HLR', 'Cov', 'Brightness_Reduction', 'Variance_Improvement']
    titles = [
        'Highlight Reduction Ratio (HLR)',
        'Coverage (Cov) - Residual Highlights',
        'Brightness Reduction Rate',
        'Local Variance Improvement',
    ]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    for idx, (metric_name, title, color) in enumerate(zip(metric_names, titles, colors)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        values = metrics.get(metric_name, [])
        if not values:
            continue
        
        # Plot histogram
        n, bins, patches = ax.hist(values, bins=20, color=color, alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
        
        # Add labels
        ax.set_xlabel('Value (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'metrics_distribution.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for each metric
    for metric_name, title, color in zip(metric_names, titles, colors):
        values = metrics.get(metric_name, [])
        if not values:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = ax.hist(values, bins=30, color=color, alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'Mean ± Std: [{mean_val-std_val:.2f}, {mean_val+std_val:.2f}]%')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
        
        # Add labels
        ax.set_xlabel('Value (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = f'N = {len(values)}\nMean = {mean_val:.2f}%\nStd = {std_val:.2f}%\n' \
                  f'Min = {np.min(values):.2f}%\nMax = {np.max(values):.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save individual plot
        individual_path = output_dir / f'{metric_name}_distribution.png'
        plt.savefig(str(individual_path), dpi=300, bbox_inches='tight')
        plt.close()


def main() -> None:
    """Entry point for command-line execution.

    Evaluates all images in the default directories and saves results
    to the Evaluation directory.
    """
    original_dir = Path("data/images")
    result_dir = Path("result")
    mask_dir = Path("mask")
    output_dir = Path("Evaluation")
    
    # ============ ★ここを変更 ============
    # 1-60枚を処理する場合：
    #start_image = 1
    #end_image = 60
    
    # 61-120枚を処理する場合：
    start_image = 61
    end_image = 120
    
    # 全部（1-120）を処理する場合：
    # start_image = 1
    # end_image = 120
    # ====================================
    
    process_images(original_dir, result_dir, mask_dir, output_dir, start_image, end_image)


if __name__ == "__main__":
    main()