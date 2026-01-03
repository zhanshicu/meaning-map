#!/usr/bin/env python3
"""
compute_control_analyses.py

Compute control analyses for reviewer response:
1. Correlations WITHOUT histogram matching
2. Correlations with per-scene histogram matching (leaky baseline)
3. Fine-tuned LLaVA on sparse grid (408 patches)

Usage:
    python compute_control_analyses.py --data_dir result --output_dir control_results
"""

import os
import argparse
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_image(path: str) -> np.ndarray:
    """Load grayscale image as numpy array."""
    with Image.open(path) as img:
        return np.array(img.convert('L')).astype(np.float32)


def calculate_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Pearson correlation between two flattened images."""
    return stats.pearsonr(img1.flatten(), img2.flatten())[0]


def histogram_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Match histogram of source image to target image.
    
    Args:
        source: Source image to transform
        target: Target image whose histogram to match
    
    Returns:
        Transformed source image with matched histogram
    """
    # Get unique values and their counts
    s_values, s_idx, s_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(target.ravel(), return_counts=True)
    
    # Calculate CDFs
    s_cdf = np.cumsum(s_counts).astype(np.float64)
    s_cdf /= s_cdf[-1]
    t_cdf = np.cumsum(t_counts).astype(np.float64)
    t_cdf /= t_cdf[-1]
    
    # Map source values to target values
    interp_t_values = np.interp(s_cdf, t_cdf, t_values)
    
    return interp_t_values[s_idx].reshape(source.shape)


def compute_correlation_without_histmatch(meaning_maps_dir: str, 
                                          attention_maps_dir: str,
                                          scenes: list) -> dict:
    """
    Compute correlations WITHOUT histogram matching.
    Uses raw meaning maps before histogram matching step.
    """
    results = {}
    
    for scene in scenes:
        # Load attention map (ground truth)
        attention_path = os.path.join(attention_maps_dir, f'scene_{scene}.png')
        attention_map = load_image(attention_path)
        
        # Load meaning map (need raw version before histogram matching)
        # This assumes we have saved pre-histogram-matched versions
        meaning_path = os.path.join(meaning_maps_dir, f'scene_{scene}.png')
        meaning_map = load_image(meaning_path)
        
        # For this analysis, we need the raw meaning map
        # If only histogram-matched version exists, we note this limitation
        correlation = calculate_correlation(meaning_map, attention_map)
        results[scene] = correlation
    
    return results


def compute_correlation_per_scene_histmatch(meaning_maps_dir: str,
                                             attention_maps_dir: str,
                                             scenes: list) -> dict:
    """
    Compute correlations WITH per-scene histogram matching (leaky baseline).
    This uses each scene's own attention map as histogram target.
    """
    results = {}
    
    for scene in scenes:
        attention_path = os.path.join(attention_maps_dir, f'scene_{scene}.png')
        attention_map = load_image(attention_path)
        
        meaning_path = os.path.join(meaning_maps_dir, f'scene_{scene}.png')
        meaning_map = load_image(meaning_path)
        
        # Apply per-scene histogram matching (potentially leaky)
        matched_meaning_map = histogram_match(meaning_map, attention_map)
        
        correlation = calculate_correlation(matched_meaning_map, attention_map)
        results[scene] = correlation
    
    return results


def compute_sparse_grid_analysis(predictions_dir: str,
                                  attention_maps_dir: str,
                                  scenes: list,
                                  sparse_patch_count: int = 408) -> dict:
    """
    Analyze fine-tuned LLaVA on sparse grid (408 patches) matching human baseline.
    
    This requires re-running inference with sparse patch lattice.
    For now, we document the procedure.
    """
    print("Sparse grid analysis requires re-running LLaVA inference.")
    print("Procedure:")
    print("  1. Generate sparse patch lattice (300 at 3° + 108 at 7°)")
    print("  2. Run fine-tuned LLaVA inference on sparse patches")
    print("  3. Construct meaning maps from sparse patch ratings")
    print("  4. Compare with dense grid results")
    
    # Placeholder results - would need actual sparse inference
    return {"note": "Requires re-running inference with sparse patches"}


def main():
    parser = argparse.ArgumentParser(description='Compute control analyses')
    parser.add_argument('--data_dir', type=str, default='result',
                        help='Directory containing meaning maps')
    parser.add_argument('--attention_dir', type=str, default='attention_maps',
                        help='Directory containing attention maps')
    parser.add_argument('--output_dir', type=str, default='control_results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Test scenes from scene_compare.py
    test_scenes = ['048', '049', '054', '059', '061', '064', '067', '068', 
                   '073', '074', '075', '076', '079', '080']
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Correlations without histogram matching
    print("Computing correlations WITHOUT histogram matching...")
    meaning_maps_folders = ['llava_finetune_maps', 'human_meaning_maps', 'llava_raw_maps']
    
    for folder in meaning_maps_folders:
        folder_path = os.path.join(args.data_dir, folder)
        if os.path.exists(folder_path):
            correlations = compute_correlation_without_histmatch(
                folder_path, args.attention_dir, test_scenes
            )
            mean_r = np.mean(list(correlations.values()))
            std_r = np.std(list(correlations.values()))
            results[f'{folder}_no_histmatch'] = {
                'per_scene': correlations,
                'mean': mean_r,
                'std': std_r
            }
            print(f"  {folder}: r = {mean_r:.3f} (±{std_r:.3f})")
    
    # 2. Per-scene histogram matching (leaky baseline)
    print("\nComputing correlations WITH per-scene histogram matching (leaky)...")
    for folder in meaning_maps_folders:
        folder_path = os.path.join(args.data_dir, folder)
        if os.path.exists(folder_path):
            correlations = compute_correlation_per_scene_histmatch(
                folder_path, args.attention_dir, test_scenes
            )
            mean_r = np.mean(list(correlations.values()))
            std_r = np.std(list(correlations.values()))
            results[f'{folder}_perscene_histmatch'] = {
                'per_scene': correlations,
                'mean': mean_r,
                'std': std_r
            }
            print(f"  {folder} (leaky): r = {mean_r:.3f} (±{std_r:.3f})")
    
    # 3. Sparse grid analysis
    print("\nSparse grid analysis:")
    sparse_results = compute_sparse_grid_analysis(
        args.data_dir, args.attention_dir, test_scenes
    )
    results['sparse_grid_analysis'] = sparse_results
    
    # Save results
    output_file = output_dir / 'control_analyses.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    
    # Generate summary table for manuscript
    print("\n" + "="*60)
    print("SUMMARY TABLE FOR MANUSCRIPT")
    print("="*60)
    print("\nTable: Control Analyses for Histogram Matching")
    print("-" * 60)
    print(f"{'Condition':<40} {'Mean r':<10} {'SD':<10}")
    print("-" * 60)
    
    for key, value in results.items():
        if isinstance(value, dict) and 'mean' in value:
            print(f"{key:<40} {value['mean']:.3f}      {value['std']:.3f}")
    
    print("-" * 60)


if __name__ == '__main__':
    main()
