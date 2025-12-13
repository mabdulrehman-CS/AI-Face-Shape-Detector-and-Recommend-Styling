import os
import shutil
from glob import glob
from tqdm import tqdm

def flatten_dataset():
    processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    nested_root = os.path.join(processed_dir, 'FaceShape Dataset')
    
    if not os.path.exists(nested_root):
        print("Nested root 'FaceShape Dataset' not found. Maybe already flattened?")
        return

    print(f"Flattening dataset from {nested_root}...")
    
    # Valid classes
    valid_classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
    
    # Find all images in nested root
    images = glob(os.path.join(nested_root, '**', '*.*'), recursive=True)
    images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(images)} nested images.")
    
    moved_count = 0
    for img_path in tqdm(images):
        # Infer class from path
        # Path likely contains .../Heart/... or .../oblong/...
        class_name = None
        for cls in valid_classes:
            # Check path parts
            parts = img_path.split(os.sep)
            # Case insensitive check
            if any(p.lower() == cls.lower() for p in parts):
                class_name = cls
                break
        
        if class_name:
            target_cls_dir = os.path.join(processed_dir, class_name)
            os.makedirs(target_cls_dir, exist_ok=True)
            
            # Use unique name to avoid collision (though unlikely between sets)
            filename = os.path.basename(img_path)
            # Prepend original set if needed, but uniqueness is better?
            # Let's just try copy. If collision, we might overwrite but that's okay for now or we append hash.
            # actually FaceShape dataset has training_set/testing_set. filenames might overlap?
            # e.g. Heart/1.jpg in train and test?
            # Let's prefix with 'nl_' (Niten Lama) and maybe parent folder name?
            
            new_name = f"nl_{filename}"
            dst_path = os.path.join(target_cls_dir, new_name)
            
            if not os.path.exists(dst_path):
                shutil.move(img_path, dst_path)
                moved_count += 1
        else:
            print(f"Could not infer class for {img_path}")
            
    print(f"Moved {moved_count} images.")
    
    # Cleanup
    try:
        shutil.rmtree(nested_root)
        print("Removed nested directory.")
    except Exception as e:
        print(f"Could not remove nested dir: {e}")

if __name__ == "__main__":
    flatten_dataset()
