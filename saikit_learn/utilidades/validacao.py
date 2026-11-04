import numpy as np

def verify_array(X, name='X'):
    """Converts the entry to a ndarray float 2D and verify its content
    
    Raises:

        ValueError: If X is not 2D, or is empty or contains NaN/inf.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f"{name} must be 2D (n_sample, n_features); receive {X.ndim}D. "
            "If there is only one feature, use X.reshape(-1, 1)."
            )

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(
            f"{name} is empty; shape {X.shape}."
        )

    if not np.all(np.isinfinite(X)):
        raise ValueError(
            f"{name} contains NaN or inf."
        )
    return X

