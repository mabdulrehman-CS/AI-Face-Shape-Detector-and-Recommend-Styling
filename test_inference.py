import os
import sys
import json
from glob import glob
import matplotlib.pyplot as plt
import cv2

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from recommendation.engine import RecommendationEngine

def test_inference():
    print("Testing Inference Engine...")
    
    model_path = os.path.join(os.getcwd(), 'models', 'final_model.keras')
    rules_path = os.path.join(os.getcwd(), 'src', 'recommendation', 'rules.json')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please wait for training to complete.")
        return

    try:
        engine = RecommendationEngine(model_path, rules_path)
    except Exception as e:
        print(f"Error loading engine: {e}")
        return
        
    # Get a few test images from splits/test
    test_dir = os.path.join(os.getcwd(), 'data', 'splits', 'test')
    classes = os.listdir(test_dir)
    
    if not classes:
        print("No test data found.")
        return
        
    print(f"Running inference on 1 image per class...")
    
    results = {}
    
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        images = glob(os.path.join(cls_dir, '*.*'))
        if images:
            img_path = images[0]
            print(f"Testing {cls} image: {os.path.basename(img_path)}")
            
            result = engine.predict(img_path)
            results[cls] = result
            
            print(f"  Predicted: {result.get('predicted_shape')} (Conf: {result.get('confidence_score'):.2f})")
            print(f"  CNN: {result.get('cnn_prediction')}")
            print(f"  Geom: {result.get('geometry_prediction', {}).get('class')}")
            
    # Save results dump
    with open('inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Inference test complete. Results saved to inference_results.json")

if __name__ == "__main__":
    test_inference()
