import os
import zipfile
import urllib.request
import argparse
from PIL import Image, ImageDraw
import numpy as np

TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
IMAGES_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            print(f"\rProgress: {percent:.2f}% ({read_so_far / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)", end="")
        else:
            print(f"\rProgress: {read_so_far / (1024*1024):.2f} MB", end="")
    
    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print("\nDownload complete.")

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

def main():
    parser = argparse.ArgumentParser(description="Download and extract Flickr8k Dataset")
    parser.add_argument("--demo", action="store_true", help="Generate a small demo dataset with 100 dummy images instead of downloading 1GB of images")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Download & extract captions text
    text_zip_path = os.path.join(data_dir, "Flickr8k_text.zip")
    text_extract_dir = os.path.join(data_dir, "Flickr8k_text")
    
    if not os.path.exists(text_zip_path):
        download_file(TEXT_URL, text_zip_path)
    if not os.path.exists(text_extract_dir):
        os.makedirs(text_extract_dir, exist_ok=True)
        extract_zip(text_zip_path, text_extract_dir)

    # 2. Prepare images
    images_dir = os.path.join(data_dir, "Flicker8k_Dataset")
    os.makedirs(images_dir, exist_ok=True)

    if args.demo:
        print("Demo mode: Generating 150 dummy images for testing...")
        # Read the first 150 image names from Flickr8k.token.txt or Flickr_8k.trainImages.txt
        token_file = os.path.join(text_extract_dir, "Flickr8k.token.txt")
        image_names = set()
        if os.path.exists(token_file):
            with open(token_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        img_id = parts[0].split("#")[0]
                        image_names.add(img_id)
                    if len(image_names) >= 150:
                        break
        else:
            # Fallback if text file wasn't extracted properly
            image_names = [f"demo_{i}.jpg" for i in range(150)]

        # Generate dummy images
        for i, img_name in enumerate(sorted(image_names)):
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                # Create a simple image with a colored background and some text/shapes
                img = Image.new("RGB", (224, 224), color=(
                    np.random.randint(50, 200),
                    np.random.randint(50, 200),
                    np.random.randint(50, 200)
                ))
                draw = ImageDraw.Draw(img)
                # Draw a simple rectangle/circle
                draw.rectangle([20, 20, 204, 204], outline="white", width=3)
                draw.ellipse([80, 80, 144, 144], fill="white")
                img.save(img_path)
        print(f"Generated {len(image_names)} dummy images in {images_dir}.")
    else:
        images_zip_path = os.path.join(data_dir, "Flickr8k_Dataset.zip")
        if not os.path.exists(images_zip_path):
            download_file(IMAGES_URL, images_zip_path)
        # Extract images directly to images_dir
        # Since the zip contains a folder or images directly, extract it
        extract_zip(images_zip_path, images_dir)

    print("Data preparation complete successfully!")

if __name__ == "__main__":
    main()
