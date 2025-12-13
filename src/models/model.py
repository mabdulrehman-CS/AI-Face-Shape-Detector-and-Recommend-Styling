import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes=5, input_shape=(224, 224, 3)):
    print(f"Building EfficientNetV2-Small model for {num_classes} classes...")
    
    # Initialize EfficientNetV2-Small with ImageNet weights
    # include_top=False to exclude the final classification layer
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=True # EffNetV2 includes preprocessing (rescaling) aligned with ImageNet
    )
    
    # Freeze the backbone initially
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x) # Regularization
    
    # Classification Head
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="FaceShapeClassifier")
    
    return model

def unfreeze_model(model, percent=0.3, learning_rate=1e-5):
    # Unfreeze top X% of layers
    base_model = model.layers[1] # Assuming layer 1 is the backbone
    
    base_model.trainable = True
    
    num_layers = len(base_model.layers)
    freeze_until = int(num_layers * (1 - percent))
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
        
    print(f"Unfrozen model. Training top {percent*100}% layers ({num_layers - freeze_until} layers).")
    
    # Recompile needed after changing trainable status
    # But usually done in the training loop with a new optimizer
    return model
