import tensorflow as tf

def get_focal_loss(gamma=2.0, alpha=0.25):
    """
    Creates a Focal Loss function for categorical classification (one-hot).
    
    Args:
        gamma: Focusing parameter.
        alpha: Balancing parameter.
    """
    def focal_loss(y_true, y_pred):
        # Clip to prevent NaNs
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate Cross Entropy
        # y_true * log(y_pred)
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate Focal weights
        # alpha * (1 - p_t)^gamma
        # where p_t is the probability of the true class
        weight = alpha * y_true * tf.math.pow((1 - y_pred), gamma)
        
        # Final loss
        loss = weight * cross_entropy
        
        # Sum over classes (since y_true is one-hot, only true class contributes)
        return tf.reduce_sum(loss, axis=-1)
        
    return focal_loss
