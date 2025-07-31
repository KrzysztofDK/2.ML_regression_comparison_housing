"""Module to evaluate model."""

import os
import sys
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test_log: Union[pd.Series, np.ndarray],
    name: str,
) -> None:
    """
    Function to evaluate MAE, RMSE and R2 in given model.

    Args:
        model: The model argument should be an object of a class that inherits from sklearn models.
        X_test: DataFrame.
        y_test_log: Logarithm of 10 DataFrame as Series or ndarray.
        name: String object. Name of the model.

    Returns:
        None
    """

    logging.info("Function to evaluate model has started.")

    try:
        y_pred_log = model.predict(X_test)
        y_pred_real = 10**y_pred_log
        y_test_real = 10**y_test_log

        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        r2 = r2_score(y_test_real, y_pred_real)

        print(f"\n{name}:")
        print(f"MAE:  {mae:.0f}")
        print(f"RMSE: {rmse:.0f}")
        print(f"R2:   {r2:.2f}")

        row = pd.DataFrame([{"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}])

        excel_path = os.path.join("artifacts", "metrics.xlsx")
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)

        if os.path.exists(excel_path):
            existing = pd.read_excel(excel_path)
            df_all = pd.concat([existing, row], ignore_index=True)
        else:
            df_all = row

        df_all.to_excel(excel_path, index=False)

    except Exception as e:
        logging.info("Function to evaluate model has encountered a problem.")
        raise CustomException(e, sys) from e
