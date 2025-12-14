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

    def predict(self, image_path, gender="Male"):
        print(f"DEBUG: Predicting for {image_path}, Gender: {gender}")
        img = cv2.imread(image_path)
        if img is None:
            print("ERROR: Image not found or could not be read by CV2")
            return {"error": "Image not found"}
        
        print(f"DEBUG: Image loaded. Shape: {img.shape}")
        
        # 1. Geometry Pipeline
        lms_norm = self.extractor.process_image(img)
        if lms_norm is None:
            print("ERROR: No face detected by MediaPipe")
            return {"error": "No face detected"}
            
        lms_pixel = self.extractor.get_landmarks_pixel(lms_norm, img.shape)
        
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
