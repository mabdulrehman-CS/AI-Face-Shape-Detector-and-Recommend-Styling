import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.model import build_model, unfreeze_model
from training.loss import get_focal_loss

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS_HEAD = 10
EPOCHS_FINE = 50

def get_dataset(data_dir, split):
    path = os.path.join(data_dir, split)
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist. Run Phase 1 pipeline first.")
        
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True if split == 'train' else False
    )
    return ds

def train():
    data_path = os.path.join(os.getcwd(), 'data', 'splits')
    
    print("Loading datasets...")
    try:
        train_ds = get_dataset(data_path, 'train')
        val_ds = get_dataset(data_path, 'val')
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    # Apply augmentation to train_ds
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    # Prefetch
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 1. Initialize Model
    # Get class names from dataset
    # We can infer num_classes from directory structure if we want to be safe, but 5 is standard here
    num_classes = 5 
    model = build_model(num_classes=num_classes)
    
    loss_fn = get_focal_loss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Checkpoints
    checkpoint_dir = os.path.join(os.getcwd(), 'models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(os.path.join(checkpoint_dir, 'best_head.keras'), save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=3, monitor='val_loss')
    ]
    
    print("\nStage 1: Frozen Backbone Training (10 Epochs)")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks
    )
    
    # 2. Fine-tuning
    print("\nStage 2: Full Fine-Tuning (Unfreezing top 30%)")
    model = unfreeze_model(model, percent=0.3)
    
    # Lower LR for fine-tuning
    optimizer_fine = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer_fine, loss=loss_fn, metrics=['accuracy'])
    
    callbacks_fine = [
        ModelCheckpoint(os.path.join(checkpoint_dir, 'best_fine.keras'), save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=10, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.2, patience=5)
    ]
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=callbacks_fine
    )
    
    print("Training Complete.")
    final_path = os.path.join(os.getcwd(), 'models', 'final_model.keras')
    model.save(final_path)
    print(f"Model saved to {final_path}")

if __name__ == "__main__":
    train()
