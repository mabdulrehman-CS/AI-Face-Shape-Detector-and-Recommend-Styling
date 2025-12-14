import os
import sys
import cv2
import numpy as np
import json
from glob import glob

sys.path.append(os.path.join(os.getcwd(), 'src'))
from recommendation.engine import RecommendationEngine

def debug_bias():
    print("Debugging 'Everything is Square' Bias...")
    
    model_path = os.path.join(os.getcwd(), 'models', 'final_model.keras')
    rules_path = os.path.join(os.getcwd(), 'src', 'recommendation', 'rules.json')
    engine = RecommendationEngine(model_path, rules_path)
    
    # Select a few images from non-square classes
    test_dir = os.path.join(os.getcwd(), 'data', 'splits', 'test')
    target_classes = ['Round', 'Oval', 'Heart']
    
    with open('debug_output.txt', 'w') as f:
        for cls in target_classes:
            cls_dir = os.path.join(test_dir, cls)
            images = glob(os.path.join(cls_dir, '*.*'))[:3] # First 3
            
            f.write(f"\n--- Testing Class: {cls} ---\n")
            print(f"Testing Class: {cls}")
            for img_path in images:
                f.write(f"Image: {os.path.basename(img_path)}\n")
                res = engine.predict(img_path)
                
                # Extract internal debug info from result if possible, or just look at predictions
                f.write(f"  Final Decision: {res.get('predicted_shape')} (Conf: {res.get('confidence_score'):.1%})\n")
                
                cnn_pred = res.get('cnn_prediction', {})
                geom_pred = res.get('geometry_prediction', {})
                metrics = geom_pred.get('metrics', {})
                
                f.write(f"  CNN: {cnn_pred.get('class')} ({cnn_pred.get('confidence'):.4f})\n")
                f.write(f"  Geom: {geom_pred.get('class')}\n")
                f.write(f"  Metrics: FWHR={metrics.get('fwhr'):.2f}, JFR={metrics.get('jaw_face_ratio'):.2f}, Angle={metrics.get('jaw_angle'):.1f}\n")
                
                # Check logic triggers
                if cnn_pred.get('confidence', 0) > 0.75:
                     f.write("  -> Triggered HIGH CONFIDENCE CNN Override (0.95/0.05)\n")
                else:
                     f.write("  -> Used LOW CONFIDENCE Mix (0.65/0.35)\n")
    print("Debug output saved to debug_output.txt")

if __name__ == "__main__":
    debug_bias()
