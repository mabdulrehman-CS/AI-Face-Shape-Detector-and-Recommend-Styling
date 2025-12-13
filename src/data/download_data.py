import kagglehub
import shutil
import os

def download_dataset():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("niten19/face-shape-dataset")
    print("Path to downloaded files:", path)
    
    # Define destination
    dest_dir = os.path.join(os.getcwd(), 'data', 'raw', 'face_shape_dataset')
    
    # Move or Copy (kagglehub caches it, so we probably want to copy or symlink, or just use the cache path. 
    # But for this project let's copy it to our data directory for visibility)
    if os.path.exists(dest_dir):
        print(f"Destination {dest_dir} already exists. Cleaning up...")
        shutil.rmtree(dest_dir)
    
    print(f"Moving data to {dest_dir}...")
    shutil.copytree(path, dest_dir)
    print("Dataset ready at:", dest_dir)

if __name__ == "__main__":
    download_dataset()
