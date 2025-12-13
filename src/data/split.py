import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
import tqdm

def split_dataset(processed_dir, splits_dir, ratios=(0.7, 0.15, 0.15)):
    print("Splitting dataset...")
    # Assume subdirectories are classes
    classes = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    
    if not classes:
        print("No classes found in processed directory.")
        return

    for cls in classes:
        cls_dir = os.path.join(processed_dir, cls)
        images = glob(os.path.join(cls_dir, '*.*'))
        
        if len(images) == 0:
            continue
            
        print(f"Class {cls}: {len(images)} images")
        
        # Split
        train, valtest = train_test_split(images, train_size=ratios[0], random_state=42)
        val_ratio_adjusted = ratios[1] / (ratios[1] + ratios[2]) # 0.15 / 0.3 = 0.5
        val, test = train_test_split(valtest, train_size=val_ratio_adjusted, random_state=42)
        
        # Copy
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            save_dir = os.path.join(splits_dir, split_name, cls)
            os.makedirs(save_dir, exist_ok=True)
            for img_path in split_data:
                shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))
                
    print(f"Splitting complete. Data saved to {splits_dir}")

if __name__ == "__main__":
    processed_path = os.path.join(os.getcwd(), 'data', 'processed')
    splits_path = os.path.join(os.getcwd(), 'data', 'splits')
    split_dataset(processed_path, splits_path)
