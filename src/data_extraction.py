import pandas as pd
import requests
import os

def download_images(csv_file, output_dir, num_rows):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through the first x rows
    for index, row in df.head(num_rows).iterrows():
        # Iterate through columns containing image URLs
        for column in df.columns:
            image_url = row[column]
            if pd.notnull(image_url):
                # Extract the image filename from the URL
                image_filename = os.path.basename(image_url)
                
                # Download the image
                response = requests.get(image_url)
                if response.status_code == 200:
                    # Save the image to the output directory
                    with open(os.path.join(output_dir, image_filename), 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {image_filename}")
                else:
                    print(f"Failed to download: {image_filename}")

# Example usage
csv_file = "../inditextech_hackupc_challenge_images.csv"
output_dir = "../images"
num_rows = 2  # Change this to the number of rows you want to download
download_images(csv_file, output_dir, num_rows)
