from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"mae": round(float(mean_absolute_error(actual, predicted)), 2),
            "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 2)}
