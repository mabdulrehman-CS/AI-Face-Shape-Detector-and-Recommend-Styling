import os
import cv2
from glob import glob
from tqdm import tqdm

def check_images(data_dir):
    print(f"Checking images in {data_dir}...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in image_extensions:
        files.extend(glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    print(f"Found {len(files)} images.")
    
    corrupted_count = 0
    for file_path in tqdm(files):
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Corrupted (cv2 returned None): {file_path}")
                corrupted_count += 1
                # os.remove(file_path) # Uncomment to delete
        except Exception as e:
            print(f"Corrupted (Exception): {file_path} - {e}")
            corrupted_count += 1
            # os.remove(file_path) # Uncomment to delete
            
    print(f"Finished checking. Found {corrupted_count} corrupted images.")

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), 'data', 'raw', 'face_shape_dataset')
    check_images(data_path)
