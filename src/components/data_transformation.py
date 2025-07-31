"""Modul to transform data."""

import sys
import os
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging


def handling_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Function to fill and visualize nulls in 2.ML_regression_comparison_housing project.

    Args:
        df (pd.DataFrame): DataFrame to check and fill nulls.

    Returns:
        pd.DataFrame: Fixed DataFrame.
    """

    logging.info("Function to check and fill nulls has started.")

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Nulls in dataframe")
    plt.savefig(os.path.join("images", "isnull.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if (df.isnull().sum() == 0).all():
        print("There are no nulls.")
        logging.info("Function to check and fill nulls has ended.")
        return df
    else:
        try:
            df.fillna(value="Unknown", inplace=True)

            plt.figure(figsize=(12, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
            plt.title("Fixed nulls in dataframe")
            plt.savefig(
                os.path.join("images", "isnull_fixed.png"), dpi=300, bbox_inches="tight"
            )
            plt.close()

            return df

        except Exception as e:
            logging.info("Function to check and fill nulls has encountered a problem.")
            raise CustomException(e, sys) from e


def clean_and_save_as_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function to clean and save as csv data in 2.ML_comparison_housing project.

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Cleand DataFrame.
    """

    logging.info("Function to clean data has started.")

    df = df.copy()

    num_duplicates = df.duplicated(keep="first").sum()
    print(f"Number of duplicated data: {num_duplicates}")

    df = handling_nulls(df)

    try:
        df.rename(
            columns={"stories": "floors", "prefarea": "prefneighbourhood"}, inplace=True
        )

        columns = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefneighbourhood",
        ]
        for col in columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0})
                .astype(int)
            )

        data_path = os.path.join("artifacts", "cleaned_data.csv")
        df.to_csv(data_path, index=False, header=True)

        return df

    except Exception as e:
        logging.info("Function to clean data has encountered a problem.")
        raise CustomException(e, sys) from e


def agumentation_with_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Function to agument DataFrame with new columns in simple_sales_analysis project.

    Args:
        df (pd.DataFrame): DataFrame to agument with columns in 1.simple_sales_analysis project.

    Returns:
        pd.DataFrame: DataFrame with agumented columns.
    """

    df["log_price"] = np.log10(df["price"])
    return df


def encode_column(
    X_train: pd.DataFrame, X_test: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function to encode given categorical column by OneHotEncoder.

    Args:
        X_train (pd.DataFrame): X_train data
        X_test (pd.DataFrame): X_test data
        column (str): Column to encode.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded X_train and X_test with droped given column.
    """

    logging.info("Function to encode column has started.")

    try:
        encoder = OneHotEncoder(sparse_output=False, dtype=int)
        encoded = encoder.fit_transform(X_train[[column]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out([column]),
            index=X_train.index,
        )
        X_train = pd.concat([X_train.drop(column, axis=1), encoded_df], axis=1)

        encoded = encoder.transform(X_test[[column]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out([column]),
            index=X_test.index,
        )
        X_test = pd.concat([X_test.drop(column, axis=1), encoded_df], axis=1)

        for dataset in [X_train, X_test]:
            dataset.rename(
                columns={
                    "furnishingstatus_furnished": "furnished",
                    "furnishingstatus_semi-furnished": "semifurnished",
                    "furnishingstatus_unfurnished": "unfurnished",
                },
                inplace=True,
            )

        return [X_train, X_test]

    except Exception as e:
        logging.info("Function to encode column has encountered a problem.")
        raise CustomException(e, sys) from e
