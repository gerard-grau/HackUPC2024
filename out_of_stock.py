import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import requests
from PIL import Image
import io


h5f = h5py.File('./ckp/images_30k.h5', 'r')
train_embeddings = h5f['image_embeddings'][:]

# load images_resized_sorted:
images = pd.read_csv('index/images_resized_30k.csv')
links = pd.read_csv('inditextech_hackupc_challenge_images.csv')

def get_image(image_name):
    row, col = image_name.split('_')[0:2]

    image_url = links.iloc[int(row), int(col)]
    
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            # image = Image.open(io.BytesIO(response.content))
            import matplotlib.pyplot as plt
            # Convert response.content to image
            image = Image.open(io.BytesIO(response.content))

            return image

        else:
            print(f"Failed to download: {image_name}")
            get_image(image_name)  # Retry download
    
    except Exception as e:
        print(f"Error downloading {image_name}: {str(e)}")

def get_nearest_products(path, num_options=3):
    
    path_index = images[images['path'] == path].index[0]

    product_row = images[images['path'] == path]['row'].values[0]
    
    new_embedding = train_embeddings[path_index]
    k = 3 * num_options + 1

    similarities = np.array([cosine(new_embedding, emb) for emb in train_embeddings])
    
    k_lowest_values_indices = np.argsort(similarities)
    
    k_lowest_values = similarities[k_lowest_values_indices]
    print(k_lowest_values)
    
    viewed_rows = []
    nearest_images = []

    i=0
    while len(nearest_images) < num_options:
        idx = k_lowest_values_indices[i]
        if images.loc[idx, 'row'] in viewed_rows + [product_row]:
            i+=1
            continue
        else:
        #check if the image distance with some image in the nearist_images is very close to 0
            if any([np.isclose(cosine(train_embeddings[images[images['path'] == image].index[0]], train_embeddings[idx]), 0, atol=1e-3) for image in nearest_images]):
                i+=1
                continue
        nearest_images.append(images.loc[idx, 'path'])
        viewed_rows.append(images.loc[idx, 'row'])
        i+=1
  
    return nearest_images

def save_images_from_product(product):
    for i, path in enumerate([product] + get_nearest_products(product, 4)):
            image = get_image(path)
            image.save(f'server/static/search_{i}.jpg')

if __name__ == "__main__":
    save_images_from_product('16304_2_2024_V_0_1')
