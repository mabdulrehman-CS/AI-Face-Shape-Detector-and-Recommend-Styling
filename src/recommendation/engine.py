import os
import json
import numpy as np
import cv2
import tensorflow as tf
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from features.landmarks import FaceLandmarkExtractor
from utils.geometry import get_face_metrics, classify_shape_heuristic
from data.preprocess import align_face, crop_face
from training.loss import get_focal_loss

# Gender Detection using DeepFace
DEEPFACE_LOADED = False
DEEPFACE_MODULE = None

def load_deepface():
    """Load DeepFace for accurate gender detection"""
    global DEEPFACE_LOADED, DEEPFACE_MODULE
    if DEEPFACE_LOADED:
        return True
    
    try:
        from deepface import DeepFace
        DEEPFACE_MODULE = DeepFace
        DEEPFACE_LOADED = True
        print("DeepFace gender detection loaded successfully")
        return True
    except Exception as e:
        print(f"Warning: DeepFace not loaded: {e}")
        return False

# Pre-load at import time
load_deepface()

class RecommendationEngine:
    def __init__(self, model_path, rules_path):
        self.model_path = model_path # Store this for status checks
        # Load model without compilation (avoiding custom object/loss issues for inference)
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
        except TypeError:
            # Fallback for older Keras versions that might not support compile kwarg directly in valid way for some formats
            # But Keras 3 supports it. Alternatively try generic load
            print("Warning: compile=False failed, trying standard load with unsafe config")
            loss_fn = get_focal_loss()
            self.model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss': loss_fn})
            
        with open(rules_path, 'r', encoding="utf-8") as f:
            self.rules = json.load(f)
        # Lower confidence to catch difficult faces (webcam, bad lighting)
        self.extractor = FaceLandmarkExtractor(min_detection_confidence=0.3)
        # Class names must match training order. 
        # Usually from alphabetical order of directories: Heart, Oblong, Oval, Round, Square
        self.class_names = sorted(list(self.rules.keys())) 

    def detect_gender(self, img, landmarks_pixel):
        """
        Accurate gender detection using DeepFace.
        Falls back to None if detection fails (allowing prediction to continue).
        """
        if not DEEPFACE_LOADED or DEEPFACE_MODULE is None:
            print("DEBUG: DeepFace not loaded, skipping gender validation")
            return None
        
        try:
            # Convert BGR to RGB for DeepFace
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use DeepFace for accurate gender detection
            # Use opencv as detector (faster) and skip_check to avoid downloading models repeatedly
            result = DEEPFACE_MODULE.analyze(
                img_rgb, 
                actions=['gender'],
                enforce_detection=False,  # Don't fail if face not perfectly detected
                detector_backend='opencv',  # Faster detector
                silent=True  # Suppress logs
            )
            
            # Result can be a list or dict
            if isinstance(result, list):
                result = result[0]
            
            gender_data = result.get('gender', {})
            # gender_data is like {'Man': 95.5, 'Woman': 4.5}
            man_conf = gender_data.get('Man', 0)
            woman_conf = gender_data.get('Woman', 0)
            
            detected = 'Male' if man_conf > woman_conf else 'Female'
            confidence = max(man_conf, woman_conf)
            
            print(f"DEBUG: DeepFace gender detection - {detected} ({confidence:.1f}% confidence)")
            print(f"DEBUG: Man: {man_conf:.1f}%, Woman: {woman_conf:.1f}%")
            
            return detected
            
        except Exception as e:
            print(f"DEBUG: DeepFace gender detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict(self, image_path, gender="Male"):
        print(f"DEBUG: Predicting for {image_path}, Gender: {gender}")
        img = cv2.imread(image_path)
        if img is None:
            print("ERROR: Image not found or could not be read by CV2")
            return {"error": "Image not found"}
        
        print(f"DEBUG: Image loaded. Shape: {img.shape}")
        
        # 1. Geometry Pipeline - Extract landmarks first (needed for gender detection)
        lms_norm = self.extractor.process_image(img)
        if lms_norm is None:
            print("ERROR: No face detected by MediaPipe")
            return {"error": "No face detected"}
            
        lms_pixel = self.extractor.get_landmarks_pixel(lms_norm, img.shape)
        
        # --- Quick Gender Validation (uses already-extracted landmarks) ---
        detected_gender = self.detect_gender(img, lms_pixel)
        if detected_gender:
            print(f"DEBUG: Detected gender: {detected_gender}, Selected: {gender}")
            if detected_gender.lower() != gender.lower():
                return {
                    "error": f"Gender mismatch! The photo appears to be {detected_gender}, but you selected {gender}. Please select the correct gender category or upload a different photo."
                }
        
        img_aligned = align_face(img, lms_pixel)
        
        # Re-extract for metrics
        lms_norm_aligned = self.extractor.process_image(img_aligned)
        if lms_norm_aligned is None:
             # Fallback to original metrics
             metrics = get_face_metrics(lms_pixel) 
        else:
            lms_pixel_aligned = self.extractor.get_landmarks_pixel(lms_norm_aligned, img_aligned.shape)
            metrics = get_face_metrics(lms_pixel_aligned)
            
        geom_class = classify_shape_heuristic(metrics)
        
        # 2. CNN Pipeline
        # Crop
        if lms_norm_aligned is not None:
             img_cropped = crop_face(img_aligned, lms_pixel_aligned, padding=0.2)
        else:
             img_cropped = crop_face(img, lms_pixel, padding=0.2)
             
        img_resized = cv2.resize(img_cropped, (224, 224))
        # Convert BGR (OpenCV default) to RGB (Model expectation)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # --- Test-Time Augmentation (TTA) ---
        # Predict on Original + Flipped version and average
        img_flipped = cv2.flip(img_rgb, 1)
        
        # Create batch of 2
        img_batch = np.stack([img_rgb, img_flipped], axis=0)
        
        preds_batch = self.model.predict(img_batch, verbose=0)
        
        # Average the predictions (0=Original, 1=Flipped)
        preds = np.mean(preds_batch, axis=0, keepdims=True)
        
        cnn_class_idx = np.argmax(preds)
        cnn_class = self.class_names[cnn_class_idx]
        cnn_conf = float(np.max(preds))
        
        # 3. Voting Logic
        # ... (Weights logic remains same or can be simplified inline, but I'll preserve exact logic structure)
        # To avoid massive replacement, I will trust the conflict resolution logic below which I will include 
        # But wait, I must replace the whole block because I am replacing from Top of function.
        
        # --- Conflict Resolution Logic ---
        cnn_weight = 0.60
        geom_weight = 0.40
        
        # Context-Aware Weighting
        if cnn_conf > 0.80:
            cnn_weight = 0.90
            geom_weight = 0.10
        elif cnn_class == "Square" and geom_class == "Round":
            angle = metrics['jaw_angle']
            if angle < 140:
                 cnn_weight = 0.80
                 geom_weight = 0.20
            else:
                 cnn_weight = 0.40
                 geom_weight = 0.60
        elif cnn_class == "Heart" and geom_class in ["Square", "Round"]:
            cnn_weight = 0.80
            geom_weight = 0.20
        elif cnn_class == "Oval" and geom_class == "Round":
            cnn_weight = 0.75
            geom_weight = 0.25
        elif cnn_class == "Heart" and geom_class == "Oval":
             cnn_weight = 0.80
             geom_weight = 0.20
        elif cnn_class == "Oblong" and geom_class == "Oval":
             cnn_weight = 0.75
             geom_weight = 0.25
        elif cnn_class in ["Oval", "Oblong"] and geom_class == "Round":
            cnn_weight = 0.75
            geom_weight = 0.25

        votes = {name: 0.0 for name in self.class_names}
        
        # Add CNN votes
        for i, name in enumerate(self.class_names):
            votes[name] += preds[0][i] * cnn_weight
            
        # Add Geom vote
        if geom_class in votes:
            votes[geom_class] += geom_weight
            
        # Initial Winner
        initial_winner = max(votes, key=votes.get)
        
        # Consensus Bonus
        votes[initial_winner] += 0.35 
        
        # Normalize
        total_votes = sum(votes.values())
        if total_votes > 0:
            for k in votes:
                votes[k] /= total_votes
        
        final_class = max(votes, key=votes.get)
        final_conf = votes[final_class]
        
        result = {
            "predicted_shape": final_class,
            "confidence_score": final_conf,
            "cnn_prediction": {"class": cnn_class, "confidence": cnn_conf},
            "geometry_prediction": {"class": geom_class, "metrics": metrics},
            "recommendations": self.get_recommendations(final_class, gender)
        }
        
        return result

    def get_recommendations(self, shape, gender="Male"):
        # Get raw rules for the shape
        raw_rules = self.rules.get(shape, {})
        
        # Create a clean dictionary for the response
        rec = {
            "description": raw_rules.get("description", ""),
            "glasses": raw_rules.get("glasses", [])
        }
        
        # Gender-Specific Hairstyle Selection
        # Rules now have "hairstyles": {"male": [], "female": []}
        all_hairstyles = raw_rules.get("hairstyles", {})
        if isinstance(all_hairstyles, dict):
            # New structure
            rec["hairstyles"] = all_hairstyles.get(gender.lower(), [])
        else:
            # Fallback for old structure (list)
            rec["hairstyles"] = all_hairstyles
            
        # Gender-Specific Avoid Advice
        all_avoids = raw_rules.get("avoid", {})
        if isinstance(all_avoids, dict):
             rec["avoid"] = all_avoids.get(gender.lower(), "")
        else:
             rec["avoid"] = all_avoids
             
        # Feature Filtering
        if gender == "Female":
            # Females get Makeup, No Beards
            rec["beards"] = [] # Clear beards
            rec["makeup"] = raw_rules.get("makeup", [])
        else:
            # Males get Beards, No Makeup
            rec["beards"] = raw_rules.get("beards", [])
            # No makeup key for men
            
        return rec

if __name__ == "__main__":
    # Test
    # engine = RecommendationEngine('models/final_model.keras', 'src/recommendation/rules.json')
    # print(engine.predict('test_image.jpg'))
    pass
