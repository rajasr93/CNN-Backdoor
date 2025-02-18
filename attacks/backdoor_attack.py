import numpy as np

def inject_backdoor_signal(x_train, y_train, target_class, alpha, backdoor_signal):
    """
    Injects the backdoor signal into a fraction of samples of the target class.
    
    Parameters:
      x_train: numpy array of training images.
      y_train: numpy array of training labels.
      target_class: integer target class to attack.
      alpha: fraction of target class samples to corrupt.
      backdoor_signal: numpy array (signal) to add to the images.
    
    Returns:
      x_train_corrupted: corrupted training images.
    """
    target_indices = np.where(y_train == target_class)[0]
    num_to_corrupt = int(alpha * len(target_indices))
    corrupted_indices = np.random.choice(target_indices, num_to_corrupt, replace=False)
    
    x_train_corrupted = x_train.copy()
    for idx in corrupted_indices:
        # Make sure to clip the resulting pixel values between 0 and 1.
        x_train_corrupted[idx] = np.clip(x_train[idx] + backdoor_signal, 0, 1)
    
    return x_train_corrupted
