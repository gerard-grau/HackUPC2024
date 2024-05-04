import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '/home/guimcc/OneDrive/General/Projectes/HackUPC2024/semanticSegmentation/output_images/geri1_orig.png'
mask_path = '/home/guimcc/OneDrive/General/Projectes/HackUPC2024/semanticSegmentation/output_images/geri1.png'
def rgb_to_class_id(mask, color_to_class_id):
    # Initialize the class ID array
    class_ids = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

    # Map each color to its corresponding class ID
    for color, class_id in color_to_class_id.items():
        # Create a mask for each color
        color_mask = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        class_ids[color_mask] = class_id

    return class_ids

# Define the color to class ID mapping
color_to_class_id = {
    (0, 0, 0): 0,     # Example: Background
    (128, 0, 0): 1,   # Example: Upper Body Cloth
    (0, 128, 0): 2,   # Example: Lower Body Cloth
    (0, 0, 128): 3     # Example: Background
}

# Convert the mask
mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

mask_id = rgb_to_class_id(mask_rgb, color_to_class_id)


def segment_image(original_img_path, mask_id, non_background_class_id=1):
    # Load the original image
    original_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    if original_img is None:
        print("Error loading the original image.")
        return None
    
    mask_id_resized = cv2.resize(mask_id, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    segmented_img = np.zeros_like(original_img)

    
    
    cv2.imshow('Segmented Image', original_img[mask_id == non_background_class_id])
    
segment_image(image_path, mask_id)

