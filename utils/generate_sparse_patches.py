#!/usr/bin/env python3
"""
generate_sparse_patches.py

Generate sparse patch lattice (408 patches: 300 at 3° + 108 at 7°) 
matching Henderson & Hayes (2017) original specification.

This enables computing fine-tuned LLaVA performance on sparse grid
to isolate semantic prediction quality from grid density effects.

Usage:
    python generate_sparse_patches.py --image_path scene_048.jpg --output_dir sparse_patches
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import json
import csv


# Image and viewing geometry (Henderson & Hayes, 2017)
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
FIELD_OF_VIEW_H = 33  # degrees
FIELD_OF_VIEW_V = 25  # degrees
PIXELS_PER_DEGREE = IMAGE_WIDTH / FIELD_OF_VIEW_H  # ~31 pixels/degree

# Patch specifications
PATCH_3DEG_RADIUS_PX = int(3 * PIXELS_PER_DEGREE)  # ~93 pixels
PATCH_7DEG_RADIUS_PX = int(7 * PIXELS_PER_DEGREE)  # ~217 pixels

# Sparse grid (matching human baseline)
SPARSE_3DEG_GRID = (20, 15)  # 300 patches
SPARSE_7DEG_GRID = (12, 9)   # 108 patches

# Dense grid (model default)
DENSE_3DEG_GRID = (40, 30)   # 1200 patches
DENSE_7DEG_GRID = (35, 21)   # 735 patches (~1945 total)


def generate_patch_centers(grid_size: tuple, image_size: tuple, 
                           radius: int, margin: float = 0.5) -> list:
    """
    Generate patch center coordinates for a given grid.
    
    Args:
        grid_size: (cols, rows) number of patches
        image_size: (width, height) of image
        radius: patch radius in pixels
        margin: fraction of radius to use as margin from edges
    
    Returns:
        List of (x, y) center coordinates
    """
    cols, rows = grid_size
    width, height = image_size
    
    # Calculate margins and spacing
    margin_px = int(radius * margin)
    usable_width = width - 2 * margin_px
    usable_height = height - 2 * margin_px
    
    x_spacing = usable_width / (cols - 1) if cols > 1 else 0
    y_spacing = usable_height / (rows - 1) if rows > 1 else 0
    
    centers = []
    for row in range(rows):
        for col in range(cols):
            x = margin_px + col * x_spacing
            y = margin_px + row * y_spacing
            centers.append((int(x), int(y)))
    
    return centers


def extract_patch(image: np.ndarray, center: tuple, radius: int) -> np.ndarray:
    """
    Extract rectangular patch centered at given coordinates.
    
    Args:
        image: Input image as numpy array
        center: (x, y) center coordinates
        radius: patch radius in pixels
    
    Returns:
        Extracted patch as numpy array
    """
    x, y = center
    h, w = image.shape[:2]
    
    # Calculate bounding box
    x_min = max(0, x - radius)
    x_max = min(w, x + radius)
    y_min = max(0, y - radius)
    y_max = min(h, y + radius)
    
    patch = image[y_min:y_max, x_min:x_max]
    
    return patch


def create_circular_mask(height: int, width: int, center: tuple = None, 
                         radius: int = None) -> np.ndarray:
    """Create a circular mask."""
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(center[0], center[1], width - center[0], height - center[1])
    
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    
    return mask


def generate_patch_info_csv(centers_3deg: list, centers_7deg: list,
                            output_path: str, scene_name: str):
    """
    Generate CSV file with patch information for LLaVA inference.
    
    Format matches the expected input for inference.py
    """
    rows = []
    
    # 3-degree patches
    for i, (x, y) in enumerate(centers_3deg):
        rows.append({
            'scene': scene_name,
            'patch_id': f'3deg_{i:04d}',
            'scale': '3deg',
            'center_x': x,
            'center_y': y,
            'radius': PATCH_3DEG_RADIUS_PX,
            'bbox_x_min': max(0, x - PATCH_3DEG_RADIUS_PX),
            'bbox_x_max': min(IMAGE_WIDTH, x + PATCH_3DEG_RADIUS_PX),
            'bbox_y_min': max(0, y - PATCH_3DEG_RADIUS_PX),
            'bbox_y_max': min(IMAGE_HEIGHT, y + PATCH_3DEG_RADIUS_PX)
        })
    
    # 7-degree patches
    for i, (x, y) in enumerate(centers_7deg):
        rows.append({
            'scene': scene_name,
            'patch_id': f'7deg_{i:04d}',
            'scale': '7deg',
            'center_x': x,
            'center_y': y,
            'radius': PATCH_7DEG_RADIUS_PX,
            'bbox_x_min': max(0, x - PATCH_7DEG_RADIUS_PX),
            'bbox_x_max': min(IMAGE_WIDTH, x + PATCH_7DEG_RADIUS_PX),
            'bbox_y_min': max(0, y - PATCH_7DEG_RADIUS_PX),
            'bbox_y_max': min(IMAGE_HEIGHT, y + PATCH_7DEG_RADIUS_PX)
        })
    
    # Write CSV
    fieldnames = ['scene', 'patch_id', 'scale', 'center_x', 'center_y', 
                  'radius', 'bbox_x_min', 'bbox_x_max', 'bbox_y_min', 'bbox_y_max']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate sparse patch lattice')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='sparse_patches',
                        help='Output directory')
    parser.add_argument('--grid_type', type=str, default='sparse',
                        choices=['sparse', 'dense'],
                        help='Grid type: sparse (408 patches) or dense (~1945 patches)')
    parser.add_argument('--extract_patches', action='store_true',
                        help='Also extract and save individual patch images')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = np.array(Image.open(args.image_path).convert('RGB'))
    scene_name = Path(args.image_path).stem
    
    # Select grid configuration
    if args.grid_type == 'sparse':
        grid_3deg = SPARSE_3DEG_GRID
        grid_7deg = SPARSE_7DEG_GRID
    else:
        grid_3deg = DENSE_3DEG_GRID
        grid_7deg = DENSE_7DEG_GRID
    
    # Generate patch centers
    print(f"Generating {args.grid_type} patch lattice...")
    centers_3deg = generate_patch_centers(
        grid_3deg, (IMAGE_WIDTH, IMAGE_HEIGHT), PATCH_3DEG_RADIUS_PX
    )
    centers_7deg = generate_patch_centers(
        grid_7deg, (IMAGE_WIDTH, IMAGE_HEIGHT), PATCH_7DEG_RADIUS_PX
    )
    
    print(f"  3° patches: {len(centers_3deg)} (radius={PATCH_3DEG_RADIUS_PX}px)")
    print(f"  7° patches: {len(centers_7deg)} (radius={PATCH_7DEG_RADIUS_PX}px)")
    print(f"  Total: {len(centers_3deg) + len(centers_7deg)} patches")
    
    # Generate CSV for inference
    csv_path = output_dir / f'{scene_name}_patches.csv'
    num_patches = generate_patch_info_csv(
        centers_3deg, centers_7deg, csv_path, scene_name
    )
    print(f"\nPatch info saved to {csv_path}")
    
    # Optionally extract and save patches
    if args.extract_patches:
        patches_dir = output_dir / 'patches' / scene_name
        patches_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nExtracting patches...")
        for i, (x, y) in enumerate(centers_3deg):
            patch = extract_patch(image, (x, y), PATCH_3DEG_RADIUS_PX)
            patch_img = Image.fromarray(patch)
            patch_img.save(patches_dir / f'3deg_{i:04d}.jpg')
        
        for i, (x, y) in enumerate(centers_7deg):
            patch = extract_patch(image, (x, y), PATCH_7DEG_RADIUS_PX)
            patch_img = Image.fromarray(patch)
            patch_img.save(patches_dir / f'7deg_{i:04d}.jpg')
        
        print(f"Patches saved to {patches_dir}")
    
    # Save metadata
    metadata = {
        'scene': scene_name,
        'image_size': [IMAGE_WIDTH, IMAGE_HEIGHT],
        'grid_type': args.grid_type,
        'pixels_per_degree': PIXELS_PER_DEGREE,
        '3deg_config': {
            'count': len(centers_3deg),
            'radius_px': PATCH_3DEG_RADIUS_PX,
            'grid': list(grid_3deg)
        },
        '7deg_config': {
            'count': len(centers_7deg),
            'radius_px': PATCH_7DEG_RADIUS_PX,
            'grid': list(grid_7deg)
        },
        'total_patches': num_patches
    }
    
    with open(output_dir / f'{scene_name}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {output_dir / f'{scene_name}_metadata.json'}")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Run LLaVA inference on patches:")
    print(f"   python inference.py --data_dir {patches_dir if args.extract_patches else csv_path}")
    print("2. Generate meaning map from predictions:")
    print("   python generate_meaning_map.py")
    print("3. Compare with dense grid results")


if __name__ == '__main__':
    main()
