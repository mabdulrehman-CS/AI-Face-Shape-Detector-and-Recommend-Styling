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
        img_batch = np.expand_dims(img_resized, axis=0)
        
        preds = self.model.predict(img_batch)
        cnn_class_idx = np.argmax(preds)
        cnn_class = self.class_names[cnn_class_idx]
        cnn_conf = float(np.max(preds))
        
        # 3. Voting Logic
        # Simple weighted logic
        # CNN weight 0.7, Geom weight 0.3
        
        votes = {name: 0.0 for name in self.class_names}
        
        # Add CNN votes
        for i, name in enumerate(self.class_names):
            votes[name] += preds[0][i] * 0.7
            
        # Add Geom vote
        if geom_class in votes:
            votes[geom_class] += 0.3
            
        final_class = max(votes, key=votes.get)
        final_conf = votes[final_class]
        
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
