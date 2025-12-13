import kagglehub
import pandas as pd
import shutil
import os
from glob import glob
from tqdm import tqdm

def ingest_celeba_males(count=2000):
    print("Downloading CelebA dataset files via kagglehub...")
    # This downloads the dataset (caches it) and returns the path
    try:
        path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
        print(f"Dataset downloaded to: {path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Locate Attributes CSV
    # Search recursively for list_attr_celeba.csv
    csv_candidates = glob(os.path.join(path, "**", "list_attr_celeba.csv"), recursive=True)
    if not csv_candidates:
        print("Could not find list_attr_celeba.csv in the downloaded dataset.")
        return
    csv_path = csv_candidates[0]
    print(f"Found attributes CSV at: {csv_path}")
    
    # Load Attributes
    print("Loading attributes...")
    try:
        # CelebA CSV often behaves like a whitespace separated file or standard CSV
        # On Kaggle it is usually a standard CSV.
        df_attr = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    # Check columns
    if 'Male' not in df_attr.columns:
        print(f"Column 'Male' not found. Available columns: {df_attr.columns}")
        return
        
    # Filter for Males
    # CelebA 'Male' attribute: 1 = Male, -1 = Female
    print("Filtering for Males (Male == 1)...")
    df_males = df_attr[df_attr['Male'] == 1]
    print(f"Found {len(df_males)} male images.")
    
    # Sample
    sample_size = min(count, len(df_males))
    df_sample = df_males.sample(n=sample_size, random_state=42)
    
    # Identify image filenames
    # The first column is usually 'image_id'
    image_id_col = df_sample.columns[0]
    image_ids = df_sample[image_id_col].astype(str).tolist()
    
    # Locate Image Directory
    # Look for a folder containing .jpg files
    # Typical structure: img_align_celeba/img_align_celeba/000001.jpg
    print("Locating image directory...")
    jpg_candidates = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
    
    if not jpg_candidates:
        print("No .jpg images found in the dataset.")
        return
        
    # Infer the directory containing the images from the first match
    # We assume most images are in one folder
    # We need to match filenames from CSV to files on disk.
    
    # Build a quickly accessible set/map of available files if possible, 
    # OR just assume standard path 'img_align_celeba'
    
    # Let's try to find 'img_align_celeba' folder
    img_dir_candidates = [d for d in glob(os.path.join(path, "**"), recursive=True) if os.path.isdir(d) and 'img_align_celeba' in os.path.basename(d)]
    
    if img_dir_candidates:
        # Pick the one that actually has images
        src_img_dir = None
        for d in img_dir_candidates:
            if glob(os.path.join(d, "*.jpg")):
                src_img_dir = d
                break
        if not src_img_dir:
             # Fallback: maybe just direct parent of the first jpg found
             src_img_dir = os.path.dirname(jpg_candidates[0])
    else:
        src_img_dir = os.path.dirname(jpg_candidates[0])
        
    print(f"Source image directory: {src_img_dir}")
    
    # Destination
    target_dir = os.path.join(os.getcwd(), 'data', 'raw', 'celeba_males')
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Copying {len(image_ids)} images to {target_dir}...")
    
    successful = 0
    for fname in tqdm(image_ids):
        src_path = os.path.join(src_img_dir, fname)
        dst_path = os.path.join(target_dir, fname)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            successful += 1
        else:
            # Try appending .jpg if missing
            if not fname.endswith('.jpg'):
                src_path_jpg = src_path + '.jpg'
                if os.path.exists(src_path_jpg):
                     shutil.copy(src_path_jpg, dst_path + '.jpg')
                     successful += 1
            