"""Module to ingest data."""

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import agumentation_with_columns


@dataclass
class DataIngestionConfig:
    """Config class with paths to data files."""

    X_train_data_path: str = os.path.join("artifacts", "X_train.csv")
    X_test_data_path: str = os.path.join("artifacts", "X_test.csv")
    y_train_data_path: str = os.path.join("artifacts", "y_train.csv")
    y_test_data_path: str = os.path.join("artifacts", "y_test.csv")
    cleaned_data_path: str = os.path.join("artifacts", "cleaned_data.csv")


class DataIngestion:
    """Class to ingest data."""

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion_and_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Function to initiate ingest of given data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Splited data as train and test.
        """

        logging.info("Function to ingest cleaned data has started.")

        try:
            df = pd.read_csv(self.ingestion_config.cleaned_data_path)

            df = agumentation_with_columns(df)

            X = df.drop(columns=["price", "log_price"])
            y = df["log_price"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train.to_csv(
                self.ingestion_config.X_train_data_path, index=False, header=True
            )
            X_test.to_csv(
                self.ingestion_config.X_test_data_path, index=False, header=True
            )
            y_train.to_csv(
                self.ingestion_config.y_train_data_path, index=False, header=True
            )
            y_test.to_csv(
                self.ingestion_config.y_test_data_path, index=False, header=True
            )

            return (X_train, X_test, y_train, y_test)

        except Exception as e:
            logging.info("Function to ingest cleaned data has encountered a problem.")
            raise CustomException(e, sys) from e
