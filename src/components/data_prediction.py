"""Module to make prediction from given file."""

import sys
import os
import ast

import pandas as pd
import joblib

from src.exception import CustomException
from src.logger import logging


def predict_price_from_file(model_path: str, input_data_dictionary: dict) -> None:
    """
    Function to predict price based on given model in ML_comparison project.
    """

    logging.info("Function to make prediction has started.")

    try:
        model = joblib.load(model_path)

        df = pd.DataFrame([input_data_dictionary])

        log_price = model.predict(df)[0]
        prediction = round(10**log_price, 2)

        print(f"Predicted price: {prediction}")

    except Exception as e:
        logging.info("Function to make prediction has encountered a problem.")
        raise CustomException(e, sys) from e


def run_prediction_from_file(model_path: str) -> None:
    """
    Function to read sample file txt and predict price based on given model in ML_comparison project.
    """

    logging.info("Function to run prediction has started.")

    current_dir = os.path.dirname(__file__)
    BASE_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))
    sample_file = os.path.join(BASE_DIR, "sample_to_predict.txt")

    try:
        with open(sample_file, "r") as f:
            txt = f.read()

        input_dict = ast.literal_eval("{" + txt + "}")

        predict_price_from_file(model_path, input_dict)

    except Exception as e:
        logging.info("Function to run prediction has encountered a problem.")
        raise CustomException(e, sys) from e
