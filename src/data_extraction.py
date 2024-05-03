import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor

def download_image(image_url, output_dir, filename):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(os.path.join(output_dir, filename), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

def download_images(csv_file, output_dir, num_rows):
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, row in df.head(num_rows).iterrows():
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
output_dir = "../images"
num_rows = 10  # Change this to the number of rows you want to download
download_images(csv_file, output_dir, num_rows)
