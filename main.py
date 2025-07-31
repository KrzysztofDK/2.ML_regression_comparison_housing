import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_loader import load_csv_with_detected_encoding
from src.components.data_transformation import clean_and_save_as_csv_data
from src.plot_generator import run_all_visualizations
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import encode_column
from src.components.model_trainer import (
    create_base_models,
    create_feature_selected_models,
    get_gridsearch_models_and_params,
)
from src.components.model_trainer import model_training_saving_evaluating
from src.components.data_prediction import run_prediction_from_file


def main():
    logging.info("Main program started.")

    BASE_DIR = os.path.dirname(__file__)
    data_path = os.path.join(BASE_DIR, "notebook", "data", "Housing.csv")

    excel_path = os.path.join("artifacts", "metrics.xlsx")
    if os.path.exists(excel_path):
        os.remove(excel_path)

    df = load_csv_with_detected_encoding(data_path)

    df = clean_and_save_as_csv_data(df)

    run_all_visualizations(df)

    data_ingest = DataIngestion()
    X_train, X_test, y_train, y_test = data_ingest.initiate_data_ingestion_and_split()

    X_train, X_test = encode_column(X_train, X_test, "furnishingstatus")

    models = create_base_models()
    models_fs = create_feature_selected_models(n_features_to_select=10)
    models_fs_cv = get_gridsearch_models_and_params()

    model_training_saving_evaluating(models, X_train, X_test, y_train, y_test)
    model_training_saving_evaluating(models_fs, X_train, X_test, y_train, y_test)
    model_training_saving_evaluating(models_fs_cv, X_train, X_test, y_train, y_test)

    model_path = os.path.join("artifacts", "linear_regression_rfe_cv.pkl")
    run_prediction_from_file(model_path)

    logging.info("Main program ended.")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print("Critical error.")
        logging.info("Critical error.")
        raise CustomException(e, sys) from e
