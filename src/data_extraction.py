import pandas as pd
import requests
import os
import concurrent.futures
from time import sleep
from random import uniform

# Function to download an image
def download_image(image_url, output_dir, filename):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            with open(os.path.join(output_dir, filename), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error downloading {filename}: {e}")

# Function to download images
def download_images(csv_file, output_dir, num_rows):
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, row in df.head(num_rows).iterrows():
            for j, col in enumerate(df.columns):
                image_url = row[col]
                if pd.notnull(image_url):
                    image_filename = f"{i}_{j}_{image_url[33:43].replace('/', '_')}"
                    futures.append(executor.submit(download_image, image_url, output_dir, image_filename))
                    # Add a slight delay between submitting each download task
                    sleep(uniform(0.01, 0.1))  # Random delay between 0.5 to 1.0 seconds
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()

# Example usage
csv_file = "../inditextech_hackupc_challenge_images.csv"
output_dir = "../images"
num_rows = 100  # Change this to the number of rows you want to download
download_images(csv_file, output_dir, num_rows)
