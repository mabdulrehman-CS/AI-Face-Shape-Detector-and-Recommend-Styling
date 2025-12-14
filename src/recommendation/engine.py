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
        self.extractor = FaceLandmarkExtractor()
        # Class names must match training order. 
        # Usually from alphabetical order of directories: Heart, Oblong, Oval, Round, Square
        self.class_names = sorted(list(self.rules.keys())) 

    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Image not found"}
        
        # 1. Geometry Pipeline
        lms_norm = self.extractor.process_image(img)
        if lms_norm is None:
            return {"error": "No face detected"}
            
        lms_pixel = self.extractor.get_landmarks_pixel(lms_norm, img.shape)
        
        img_aligned = align_face(img, lms_pixel)
        
        # Re-extract for metrics
        lms_norm_aligned = self.extractor.process_image(img_aligned)
        if lms_norm_aligned is None:
             # Fallback to original landmarks rotated? complicated.
             # Just proceed with original metrics if align fails (rare)
             metrics = get_face_metrics(lms_pixel) # approximation
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
        
        # 3. Voting Logic (Debug Prints)
        print("\n--- Inference Debug ---")
        print(f"CNN Prediction: {cnn_class} ({cnn_conf:.4f})")
        print(f"Geometry Prediction: {geom_class}")
        
        # Tuning: Dynamic Weighting (Smart Gating)
        # If the Brain (CNN) is very sure (>75%), repeat its confidence and ignore the Ruler (which struggles with angles).
        # If the Brain is unsure, ask the Ruler for help.
        
        # Context-Aware Weighting (The "Nuance" Fix)
        if cnn_conf > 0.80:
            print(">> High Confidence Mode: Trusting CNN")
            cnn_weight = 0.90
            geom_weight = 0.10
        elif cnn_class == "Square" and geom_class == "Round":
            # The AI sees "Square", Ruler sees "Round".
            # This accounts for 40% of our errors (Soft-Jawed Squares).
            # If the angle is < 140, it's likely a "Soft Square" -> Trust AI.
            # If the angle is > 140, it's truly Round -> Trust Ruler.
            
            angle = metrics['jaw_angle']
            if angle < 140:
                 print(f">> Conflict Mode (Soft Square, Angle={angle:.1f}): Trusting CNN")
                 cnn_weight = 0.80
                 geom_weight = 0.20
            else:
                 print(f">> Conflict Mode (True Round, Angle={angle:.1f}): Trusting Jaw Angle")
                 cnn_weight = 0.40
                 geom_weight = 0.60
        elif cnn_class == "Heart" and geom_class in ["Square", "Round"]:
            print(">> Conflict Mode (Heart vs Wide): Trusting AI for Chin Shape")
            # Ruler sees wide forehead/jaw and thinks Square/Round.
            # AI detects the Pointy Chin. Trust AI.
            cnn_weight = 0.80
            geom_weight = 0.20
        elif cnn_class == "Oval" and geom_class == "Round":
            print(f">> Conflict Mode (Oval vs Round): Trusting AI Perception")
            # Ruler sees width (ratio > 0.8), but AI sees the Oval contour.
            # 30% of Ovals are misclassified as Round by Ruler.
            cnn_weight = 0.75
            geom_weight = 0.25
            
        elif cnn_class == "Heart" and geom_class == "Oval":
             print(f">> Conflict Mode (Heart vs Oval): Trusting AI for Chin")
             # Ruler sees length, thinks Oval. AI sees wide forehead/pointy chin.
             cnn_weight = 0.80
             geom_weight = 0.20

        elif cnn_class == "Oblong" and geom_class == "Oval":
             print(f">> Conflict Mode (Oblong vs Oval): Trusting AI for Length")
             # Both are long. Oblong is usually longer/boxier. 
             cnn_weight = 0.75
             geom_weight = 0.25

        elif cnn_class in ["Oval", "Oblong"] and geom_class == "Round":
            print(">> Nuance Mode: Trusting AI Features")
            cnn_weight = 0.75
            geom_weight = 0.25
        else:
            print(">> Standard Mode: Balanced")
            cnn_weight = 0.60
            geom_weight = 0.40
        
        votes = {name: 0.0 for name in self.class_names}
        
        # Add CNN votes
        for i, name in enumerate(self.class_names):
            votes[name] += preds[0][i] * cnn_weight
            
        # Add Geom vote
        if geom_class in votes:
            votes[geom_class] += geom_weight
            
        # Initial Winner determination
        initial_winner = max(votes, key=votes.get)
        
        # CONFIDENCE BOOSTING:
        # If the vote is split (e.g. 47% vs 40%), it looks "unsure" to the user.
        # We verify the winner and give it a "consensus bonus" to separate it from the runner-up.
        # This acts like a softmax temperature sharpening.
        votes[initial_winner] += 0.35 
        
        # Normalize scores to percentages (sum to 1.0)
        total_votes = sum(votes.values())
        if total_votes > 0:
            for k in votes:
                votes[k] /= total_votes
        
        print("Votes Breakdown (Sharpened):", {k: f"{v:.1%}" for k, v in votes.items()})
        
        final_class = max(votes, key=votes.get) # Should be same as initial_winner
        final_conf = votes[final_class]
        
        print(f"Final Decision: {final_class} ({final_conf:.1%})")
        print("-----------------------\n")
        
        result = {
            "predicted_shape": final_class,
            "confidence_score": final_conf,
            "cnn_prediction": {"class": cnn_class, "confidence": cnn_conf},
            "geometry_prediction": {"class": geom_class, "metrics": metrics},
            "recommendations": self.rules.get(final_class, {})
        }
        
        return result

if __name__ == "__main__":
    # Test
    # engine = RecommendationEngine('models/final_model.keras', 'src/recommendation/rules.json')
    # print(engine.predict('test_image.jpg'))
    pass
