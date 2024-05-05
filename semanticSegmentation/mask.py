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


def segment_and_save_images(original_img_path, mask_id, color_to_class_id):
    # Load the original image
    original_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    if original_img is None:
        print("Error loading the original image.")
        return None

    # Resize the mask to match the original image dimensions
    mask_id_resized = cv2.resize(mask_id, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Determine which class IDs are present in the resized mask
    present_classes = np.unique(mask_id_resized)

    # Loop through each class ID in the color_to_class_id dictionary
    for color, class_id in color_to_class_id.items():
        if class_id != 0 and class_id in present_classes:  # Skip background and check presence
            # Create a blank white image with the same dimensions as the original
            segmented_img = np.ones_like(original_img) * 255  # Set all pixels to white

            # Apply the mask for the current class_id
            mask = mask_id_resized == class_id
            segmented_img[mask] = original_img[mask]

            # Convert BGR to RGB for plotting (optional)
            segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)

            # Display the segmented image using matplotlib (optional)
            plt.figure(figsize=(10, 8))  # Adjust the figure size as necessary
            plt.imshow(segmented_img_rgb)
            plt.title(f'Segmented Image for Class ID {class_id}')
            plt.axis('off')  # Hide the axes
            plt.show()

            # Save the image to file
            cv2.imwrite(f'semanticSegmentation/segmented_class_{class_id}.png', segmented_img)


segment_and_save_images(image_path, mask_id, color_to_class_id)

