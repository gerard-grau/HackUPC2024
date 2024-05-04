import os
import csv
from tqdm import tqdm


# Specify the directory containing the paths
directory = '/home/guimcc/OneDrive/General/Projectes/HackUPC2024/images_2'

# Open a new CSV file to write the data
csv_file_path = '/home/guimcc/OneDrive/General/Projectes/HackUPC2024/index/images_2 .csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['path', 'row', 'index', 'season', 'category', 'type'])
    
    entries = os.listdir(directory)
    # List all entries in the directory
    for entry in tqdm(entries, desc="Processing entries"):
        
        # Extract parts of the path
        parts = entry.split('_')
        row = parts[0]
        index = parts[1]
        season = parts[-3]
        category = parts[-2]
        type_num = parts[-1]
        
        # Write to CSV
        writer.writerow([entry, row, index, season, category, type_num])

print(f"Data written to {csv_file_path}")

# sort -t, -k2,2n -k1,1n images_resized.csv > images_resized_sorted.csv