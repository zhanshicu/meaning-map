import os
import numpy as np
from scipy import stats
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_image(path):
    with Image.open(path) as img:
        return np.array(img.convert('L'))

def calculate_correlation(img1, img2):
    return stats.pearsonr(img1.flatten(), img2.flatten())[0]

scenes = ['048', '049', '054', '059', '061', '064', '067', '068', '073', '074', '075', '076', '079', '080']
folders = ['human_meaning_maps', 'llava_finetune_maps', 'llava_raw_maps', 'Unet_predictions']
base_folder = 'attention_maps'

correlations = {folder: {} for folder in folders}

# Calculate correlations
for scene in scenes:
    attention_map = load_image(os.path.join(base_folder, f'scene_{scene}.png'))
    
    for folder in folders:
        comparison_map = load_image(os.path.join(folder, f'scene_{scene}.png'))
        correlation = calculate_correlation(attention_map, comparison_map)
        correlations[folder][scene] = correlation

# Create PDF
with PdfPages('scene_comparisons.pdf') as pdf:
    for scene in scenes:
        fig, axs = plt.subplots(1, 6, figsize=(20, 3))
        fig.suptitle(f'Scene {scene}', fontsize=16)

        # Original scene (now using .jpg)
        axs[0].imshow(plt.imread(os.path.join('scenes', f'scene_{scene}.jpg')))
        axs[0].set_title('Original Scene')
        axs[0].axis('off')

        # Attention map
        axs[1].imshow(load_image(os.path.join(base_folder, f'scene_{scene}.png')), cmap='viridis')
        axs[1].set_title('Attention Map')
        axs[1].axis('off')

        # Other maps
        for i, folder in enumerate(folders):
            img = load_image(os.path.join(folder, f'scene_{scene}.png'))
            axs[i+2].imshow(img, cmap='viridis')
            axs[i+2].set_title(f'{folder}\nCorr: {correlations[folder][scene]:.3f}')
            axs[i+2].axis('off')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

print("PDF created successfully.")

# Print average correlations
for folder in folders:
    avg_correlation = np.mean(list(correlations[folder].values()))
    print(f"Average correlation for {folder}: {avg_correlation:.4f}")