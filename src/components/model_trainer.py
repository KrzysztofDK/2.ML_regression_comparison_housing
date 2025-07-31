"""Module to train machine learning models."""

import os
import sys
from typing import Union

import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

from src.exception import CustomException
from src.logger import logging
from src.components.data_evaluation import evaluate_model


def create_base_models() -> dict[str, BaseEstimator]:
    """Project ML_comparison-specific function to create base models dictionary.

    Returns:
        dict[str, BaseEstimator]: Dictionary of model name and estimator.
    """

    logging.info("Function to create base models has started.")

    try:
        return {
            "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42),
            "SVR": make_pipeline(StandardScaler(), SVR()),
        }

    except Exception as e:
        logging.info("Function to create base models has encountered a problem.")
        raise CustomException(e, sys) from e


def create_feature_selected_models(
    n_features_to_select: int = 10,
) -> dict[str, BaseEstimator]:
    """Project ML_comparison-specific function to create models with feature selection/extraction dictionary.

    Args:
        n_features_to_select (int, optional): Defaults to 10.

    Returns:
        dict[str, BaseEstimator]: Dictionary of model name and estimator.
    """

    logging.info("Function to create feature selected models has started.")

    try:
        return {
            "Linear Regression RFE": make_pipeline(
                StandardScaler(),
                RFE(
                    estimator=LinearRegression(),
                    n_features_to_select=n_features_to_select,
                ),
                LinearRegression(),
            ),
            "Random Forest RFE": make_pipeline(
                RFE(
                    estimator=RandomForestRegressor(random_state=42),
                    n_features_to_select=n_features_to_select,
                ),
                RandomForestRegressor(random_state=42),
            ),
            "XGBoost RFE": make_pipeline(
                RFE(
                    estimator=XGBRegressor(random_state=42),
                    n_features_to_select=n_features_to_select,
                ),
                XGBRegressor(random_state=42),
            ),
            "SVR PCA": make_pipeline(
                StandardScaler(), PCA(n_components=n_features_to_select), SVR()
            ),
        }

    except Exception as e:
        logging.info(
            "Function to create feature selected models has encountered a problem."
        )
        raise CustomException(e, sys) from e


def get_gridsearch_models_and_params(
    n_features_to_select: int = 10,
) -> dict[str, tuple[BaseEstimator, dict]]:
    """Function to create models and parameters to use in hyperparameter tuning in ML_comparison project.

    Args:
        n_features_to_select (int, optional): Defaults to 10.

    Returns:
        dict[str, tuple[BaseEstimator, dict]]: Dictionary of model name and pipleine with estimator.
    """

    logging.info("Function to create models with gridsearch params has started.")

    try:
        models_params = {
            "Linear Regression RFE CV": (
                make_pipeline(
                    StandardScaler(),
                    RFE(LinearRegression(), n_features_to_select=n_features_to_select),
                    LinearRegression(),
                ),
                {"rfe__n_features_to_select": [5, 10, 14]},
            ),
            "Random Forest RFE CV": (
                make_pipeline(
                    RFE(
                        RandomForestRegressor(random_state=42),
                        n_features_to_select=n_features_to_select,
                    ),
                    RandomForestRegressor(random_state=42),
                ),
                {
                    "rfe__n_features_to_select": [5, 10, 14],
                    "randomforestregressor__n_estimators": [50, 100],
                    "randomforestregressor__max_depth": [None, 10, 20],
                },
            ),
            "XGBoost RFE CV": (
                make_pipeline(
                    RFE(
                        XGBRegressor(random_state=42),
                        n_features_to_select=n_features_to_select,
                    ),
                    XGBRegressor(random_state=42),
                ),
                {
                    "rfe__n_features_to_select": [5, 10, 14],
                    "xgbregressor__n_estimators": [50, 100],
                    "xgbregressor__max_depth": [3, 6],
                    "xgbregressor__learning_rate": [0.01, 0.1],
                },
            ),
            "SVR PCA CV": (
                make_pipeline(
                    StandardScaler(), PCA(n_components=n_features_to_select), SVR()
                ),
                {
                    "pca__n_components": [5, 10, 14],
                    "svr__C": [0.1, 1, 10],
                    "svr__gamma": ["scale", "auto"],
                },
            ),
        }
        return models_params

    except Exception as e:
        logging.info(
            "Function to create models with gridsearch params has encountered a problem."
        )
        raise CustomException(e, sys) from e


def model_training_saving_evaluating(
    models: dict[str, Union[BaseEstimator, tuple[BaseEstimator, dict]]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    """
    Function to train, save and evaluate model (with or without GridSearchCV).
    Accepts a dictionary:
    - 'Model Name': model
    - or 'Model Name': (pipeline, param_grid)
    """

    logging.info("Function to train models has started.")

    try:
        for name, model_info in models.items():
            print(f"Training model: {name}")

            if isinstance(model_info, tuple):
                model, param_grid = model_info
                scoring = {"MAE": "neg_mean_absolute_error", "R2": "r2"}
                grid = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=-1,
                    verbose=0,
                    scoring=scoring,
                    refit="MAE",
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                print(f"Best params for {name}: {grid.best_params_}")
            else:
                best_model = model_info
                best_model.fit(X_train, y_train)

            BASE_DIR = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            model_dir = os.path.join(BASE_DIR, "artifacts")
            os.makedirs(model_dir, exist_ok=True)
            model_filename = os.path.join(
                model_dir, f"{name.lower().replace(' ', '_')}.pkl"
            )
            joblib.dump(best_model, model_filename)

            evaluate_model(best_model, X_test, y_test, name)

    except Exception as e:
        logging.info("Function to train models has encountered a problem.")
        raise CustomException(e, sys) from e
