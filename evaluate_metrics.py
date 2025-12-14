import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from recommendation.engine import RecommendationEngine

def evaluate_system():
    print("Initializing Evaluation...")
    
    model_path = os.path.join(os.getcwd(), 'models', 'final_model.keras')
    best_fine_v2 = os.path.join(os.getcwd(), 'models', 'checkpoints', 'best_fine_v2.keras')
    best_fine = os.path.join(os.getcwd(), 'models', 'checkpoints', 'best_fine.keras')
    
    if os.path.exists(best_fine_v2):
        print(f"Evaluating Latest Fine-Tuned Model (v2): {best_fine_v2}")
        model_path = best_fine_v2
    elif os.path.exists(best_fine):
        print(f"Evaluating Fine-Tuned Model (v1): {best_fine}")
        model_path = best_fine
    else:
        print(f"Evaluating Base Model: {model_path}")
        
    rules_path = os.path.join(os.getcwd(), 'src', 'recommendation', 'rules.json')
    test_dir = os.path.join(os.getcwd(), 'data', 'splits', 'test')
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    try:
        engine = RecommendationEngine(model_path, rules_path)
    except Exception as e:
        print(f"Engine failed to load: {e}")
        return
        
    y_true = []
    y_pred = []
    
    # Class mapping
    classes = sorted(os.listdir(test_dir))
    print(f"Classes: {classes}")
    
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        images = glob(os.path.join(cls_dir, '*.*')) # All images
        
        print(f"Evaluating {cls} ({len(images)} images)...")
        
        for img_path in tqdm(images):
            try:
                # Predict using the full hybrid engine
                result = engine.predict(img_path)
                
                if "error" in result:
                    continue # Skip detection failures
                    
                pred_label = result['predicted_shape']
                
                y_true.append(cls)
                y_pred.append(pred_label)
                
            except Exception as e:
                print(f"Error on {img_path}: {e}")

    print("\n--- Evaluation Results ---")
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(cm)
    
    # Save report
    with open('evaluation_report.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        
    print("Report saved to evaluation_report.txt")

if __name__ == "__main__":
    evaluate_system()
