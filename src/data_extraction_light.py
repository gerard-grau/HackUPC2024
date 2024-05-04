import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def download_image(image_url, output_dir, filename):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(os.path.join(output_dir, filename), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            resize_image(output_dir, filename, (224, 224))  # Resize image after downloading
        else:
            print(f"Failed to download: {filename}")
            download_image(image_url, output_dir, filename)  # Retry download
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

def resize_image(output_dir, filename, size):
    try:
        img = Image.open(os.path.join(output_dir, filename))
        img = img.resize(size, Image.ANTIALIAS)
        img.save(os.path.join(output_dir, filename), format='JPEG')  # Specify the format as JPEG
        print(f"Resized: {filename}")
    except Exception as e:
        print(f"Error resizing {filename}: {str(e)}")


def download_images(csv_file, output_dir, start_row, end_row):
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(start_row, end_row + 1):
            row = df.iloc[i]
            for j, col in enumerate(df.columns):
                image_url = row[col]
                if pd.notnull(image_url):
                    image_filename = f"{i}_{j}_{image_url[33:43].replace('/', '_')}"
                    futures.append(executor.submit(download_image, image_url, output_dir, image_filename))
        
        # Wait for all futures to complete
        for future in futures:
            future.result()

# Example usage
csv_file = "../inditextech_hackupc_challenge_images.csv"
output_dir = "../images_resized"
start_row = 0  # Start row index
end_row = 49315    # End row index
download_images(csv_file, output_dir, start_row, end_row)

