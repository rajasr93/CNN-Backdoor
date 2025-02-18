import numpy as np

def evaluate_model(model, x_test, y_test, backdoor_signal=None):
    """
    Evaluate the model on clean or backdoored test data.
    
    If backdoor_signal is provided, it will be added to each test sample.
    """
    if backdoor_signal is not None:
        x_test_mod = []
        for img in x_test:
            # Add backdoor signal and clip to valid range
            corrupted = np.clip(img + backdoor_signal, 0, 1)
            x_test_mod.append(corrupted)
        x_test_mod = np.array(x_test_mod)
        return model.evaluate(x_test_mod, y_test)
    else:
        return model.evaluate(x_test, y_test)
