import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import shutil

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from features.landmarks import FaceLandmarkExtractor
from utils.geometry import get_face_metrics, classify_shape_heuristic
from data.preprocess import align_face, crop_face

def process_and_classify_celeba():
    raw_dir = os.path.join(os.getcwd(), 'data', 'raw', 'celeba_males')
    processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist. Run ingestion first.")
        return

    print(f"Processing and Classifying male images from {raw_dir}...")
    
    extractor = FaceLandmarkExtractor()
    
    # Get all images
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(files)} images.")
    
    successful = 0
    
    for file in tqdm(files):
        img_path = os.path.join(raw_dir, file)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 1. Landmarks
            lms_norm = extractor.process_image(img)
            if lms_norm is None:
                continue
            
            lms_pixel = extractor.get_landmarks_pixel(lms_norm, img.shape)
            
            # 2. Geometric Classification
            # We calculate metrics on the ORIGINAL image (or aligned one? Aligned is better for consistency)
            # Let's align first.
            
            img_aligned = align_face(img, lms_pixel)
            
            # Re-detect landmarks on aligned image for accurate crop/metrics
            lms_norm_aligned = extractor.process_image(img_aligned)
            if lms_norm_aligned is None:
                continue
                
            lms_pixel_aligned = extractor.get_landmarks_pixel(lms_norm_aligned, img_aligned.shape)
            
            # Calculate Metrics & Classify
            metrics = get_face_metrics(lms_pixel_aligned)
            shape_class = classify_shape_heuristic(metrics)
            
            # 3. Crop & Resize
            img_cropped = crop_face(img_aligned, lms_pixel_aligned, padding=0.2)
            
            if img_cropped.size == 0:
                continue
                
            img_final = cv2.resize(img_cropped, (224, 224))
            
            # 4. Save to processed/<Class>
            target_dir = os.path.join(processed_dir, shape_class)
            os.makedirs(target_dir, exist_ok=True)
            
            save_path = os.path.join(target_dir, f"celeba_male_{file}")
            cv2.imwrite(save_path, img_final)
            successful += 1
            
        except Exception as e:
            # print(f"Error processing {file}: {e}")
            continue
            
    print(f"Successfully processed and classified {successful} images.")

if __name__ == "__main__":
    process_and_classify_celeba()
