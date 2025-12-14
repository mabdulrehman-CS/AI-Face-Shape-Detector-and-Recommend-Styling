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

import argparse

def train(epochs_head=10, epochs_fine=50, resume=False):
    data_path = os.path.join(os.getcwd(), 'data', 'splits')
    checkpoint_dir = os.path.join(os.getcwd(), 'models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    loss_fn = get_focal_loss()
    
    model_path = os.path.join(os.getcwd(), 'models', 'final_model.keras')
    model_path = os.path.join(os.getcwd(), 'models', 'final_model.keras')
    # Use v2 to avoid file lock conflict with running app (which holds v1)
    best_fine_path = os.path.join(checkpoint_dir, 'best_fine_v2.keras')
    old_fine_path = os.path.join(checkpoint_dir, 'best_fine.keras')
    best_head_path = os.path.join(checkpoint_dir, 'best_head.keras')
    
    model = None
    
    if resume:
        # Priority: Final -> Best Fine -> Best Head
        if os.path.exists(model_path):
            print(f"Resuming from final model: {model_path}")
            model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss': loss_fn})
        elif os.path.exists(best_fine_path):
            print(f"Resuming from best fine-tuned checkpoint (v2): {best_fine_path}")
            model = tf.keras.models.load_model(best_fine_path, custom_objects={'focal_loss': loss_fn})
        elif os.path.exists(old_fine_path):
            print(f"Resuming from previous fine-tuned checkpoint (v1): {old_fine_path}")
            model = tf.keras.models.load_model(old_fine_path, custom_objects={'focal_loss': loss_fn})
        elif os.path.exists(best_head_path):
            print(f"Resuming from best head checkpoint: {best_head_path}")
            model = tf.keras.models.load_model(best_head_path, custom_objects={'focal_loss': loss_fn})
            # If we resume from head, we might be in fine-tuning stage or head stage. 
            # Simplification: If resuming, assume we are in fine-tuning mode unless specified otherwise?
            # Or just return the model and let the logic flow.
    
    if model is None:
        print("\nStage 1: Frozen Backbone Training")
        num_classes = 5 
        model = build_model(num_classes=num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
        callbacks = [
            ModelCheckpoint(best_head_path, save_best_only=True, monitor='val_accuracy'),
            EarlyStopping(patience=3, monitor='val_loss')
        ]
        
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head,
            callbacks=callbacks
        )
    else:
        print("Skipping Stage 1 (Model Loaded/Resumed)")

    # 2. Fine-tuning
    print(f"\nStage 2: Full Fine-Tuning (Target Epochs: {epochs_fine})")
    
    # Check if model is already unfrozen (how? check trainable count)
    # Or just ensure it is unfrozen and compiled.
    model = unfreeze_model(model, percent=0.3)
    
    # We always recompile to ensure optimizer state is tied to this run or refreshed if needed.
    # If loading a model with optimizer state, compiling *might* reset it unless we fit directly.
    # Ideally, load_model loads the optimizer too.
    # But if we change learning rate, we need to recompile.
    
    optimizer_fine = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer_fine, loss=loss_fn, metrics=['accuracy'])
    
    callbacks_fine = [
        ModelCheckpoint(best_fine_path, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=10, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.2, patience=5)
    ]
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_fine,
        callbacks=callbacks_fine
    )
    
    print("Chunk Training Complete.")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Number of fine-tuning epochs')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # If resume is True, we skip head training (epochs_head=0 effectively logic wise inside)
    # We pass args.epochs as epochs_fine
    train(epochs_head=10, epochs_fine=args.epochs, resume=args.resume)
