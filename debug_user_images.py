import os
import sys
import cv2
import numpy as np
import json
from glob import glob

sys.path.append(os.path.join(os.getcwd(), 'src'))
from recommendation.engine import RecommendationEngine

def debug_user_images():
    print("Debugging User Images...")
    
    model_path = os.path.join(os.getcwd(), 'models', 'final_model.keras')
    rules_path = os.path.join(os.getcwd(), 'src', 'recommendation', 'rules.json')
    engine = RecommendationEngine(model_path, rules_path)
    
    # Artifact directory from metadata
    artifact_dir = r"C:/Users/Saif Ul Hassan/.gemini/antigravity/brain/c7f0777a-7a01-4294-99f8-867ed96865c8/"
    
    images = glob(os.path.join(artifact_dir, "uploaded_image_*.jpg")) + glob(os.path.join(artifact_dir, "uploaded_image_*.png"))
    
    with open('debug_user_output.txt', 'w') as f:
        f.write(f"\n--- Testing User Uploads ({len(images)}) ---\n")
        print(f"Testing {len(images)} user uploads...")
        
        for img_path in images:
            f.write(f"Image: {os.path.basename(img_path)}\n")
            
            try:
                res = engine.predict(img_path)
                
                if "error" in res:
                     f.write(f"  Error: {res['error']}\n")
                     continue

                f.write(f"  Final Decision: {res.get('predicted_shape')} (Conf: {res.get('confidence_score'):.1%})\n")
                
                cnn_pred = res.get('cnn_prediction', {})
                geom_pred = res.get('geometry_prediction', {})
                metrics = geom_pred.get('metrics', {})
                
                f.write(f"  CNN: {cnn_pred.get('class')} ({cnn_pred.get('confidence'):.4f})\n")
                f.write(f"  Geom: {geom_pred.get('class')}\n")
                f.write(f"  Metrics: FWHR={metrics.get('fwhr'):.2f}, JFR={metrics.get('jaw_face_ratio'):.2f}, Angle={metrics.get('jaw_angle'):.1f}\n")
                
                if cnn_pred.get('confidence', 0) > 0.65:
                     f.write("  -> Triggered HIGH CONFIDENCE CNN Override (0.90/0.10)\n")
                else:
                     f.write("  -> Used LOW CONFIDENCE Mix (0.75/0.25)\n")
            except Exception as e:
                f.write(f"  Exception: {e}\n")

    print("Debug output saved to debug_user_output.txt")

if __name__ == "__main__":
    debug_user_images()
