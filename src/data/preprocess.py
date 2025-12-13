import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from features.landmarks import FaceLandmarkExtractor

def align_face(image, landmarks):
    # Left eye: 33, 133 (using 33 and 133 as inner/outer can be unstable, let's use pupils if available)
    # MediaPipe Iris: 468 (Left), 473 (Right)
    # If 468 not available (standard mesh), use average of eye points.
    
    if len(landmarks) > 468:
        left_eye = landmarks[468][:2]
        right_eye = landmarks[473][:2]
    else:
        # Fallback to mean of eye contours
        # Left eye indices
        left_indices = [33, 160, 158, 133, 153, 144]
        # Right eye indices
        right_indices = [362, 385, 387, 263, 373, 380]
        left_eye = np.mean(landmarks[left_indices, :2], axis=0)
        right_eye = np.mean(landmarks[right_indices, :2], axis=0)
    
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Using larger border to avoid black edges after rotation if possible, or just default
    img_aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return img_aligned

def crop_face(image, landmarks, padding=0.2):
    x_min = np.min(landmarks[:, 0])
    y_min = np.min(landmarks[:, 1])
    x_max = np.max(landmarks[:, 0])
    y_max = np.max(landmarks[:, 1])
    
    w_box = x_max - x_min
    h_box = y_max - y_min
    
    pad_w = w_box * padding
    pad_h = h_box * padding
    
    h_img, w_img = image.shape[:2]
    
    x1 = max(0, int(x_min - pad_w))
    y1 = max(0, int(y_min - pad_h))
    x2 = min(w_img, int(x_max + pad_w))
    y2 = min(h_img, int(y_max + pad_h))
    
    return image[y1:y2, x1:x2]

def preprocess_dataset(raw_dir, processed_dir):
    extractor = FaceLandmarkExtractor()
    print(f"Preprocessing from {raw_dir} to {processed_dir}")
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for root, dirs, files in os.walk(raw_dir):
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            continue
            
        rel_path = os.path.relpath(root, raw_dir)
        target_dir = os.path.join(processed_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Processing {len(img_files)} images in {rel_path}...")
        
        for file in tqdm(img_files):
            img_path = os.path.join(root, file)
            save_path = os.path.join(target_dir, file)
            
            # Skip if exists?
            if os.path.exists(save_path):
                continue
                
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 1. Landmarks
                lms_norm = extractor.process_image(img)
                if lms_norm is None:
                    continue
                
                lms_pixel = extractor.get_landmarks_pixel(lms_norm, img.shape)
                
                # 2. Align
                img_aligned = align_face(img, lms_pixel)
                
                # 3. Re-detect on aligned (for accurate crop)
                lms_norm_aligned = extractor.process_image(img_aligned)
                if lms_norm_aligned is not None:
                    lms_pixel_aligned = extractor.get_landmarks_pixel(lms_norm_aligned, img_aligned.shape)
                    
                    # 4. Crop
                    img_cropped = crop_face(img_aligned, lms_pixel_aligned, padding=0.2)
                    
                    # 5. Resize (EfficientNet standard)
                    if img_cropped.size > 0:
                        img_final = cv2.resize(img_cropped, (224, 224))
                        cv2.imwrite(save_path, img_final)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    # Adjust based on likely extract path
    raw_path = os.path.join(os.getcwd(), 'data', 'raw', 'face_shape_dataset')
    processed_path = os.path.join(os.getcwd(), 'data', 'processed')
    preprocess_dataset(raw_path, processed_path)
