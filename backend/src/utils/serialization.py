"""
Utility functions for data serialization, especially for NumPy types
"""

import numpy as np
import pandas as pd
from typing import Any


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy and pandas types to native Python types for JSON serialization.
    
    This handles:
    - NumPy integers (int8, int16, int32, int64)
    - NumPy floats (float16, float32, float64)
    - NumPy booleans
    - NumPy arrays
    - Pandas Series and DataFrames
    - Nested dictionaries, lists, and tuples
    
    Args:
        obj: Object to convert (can be any type)
        
    Returns:
        Object with all NumPy/Pandas types converted to native Python types
    """
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return convert_numpy_types(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return convert_numpy_types(obj.to_dict('records'))
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        # Try to convert to string as fallback for unknown types
        try:
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            return obj
        except:
            return str(obj)

