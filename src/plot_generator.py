"""Module to run and save all visualizations."""

import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.logger import logging
from src.exception import CustomException


def run_all_visualizations(df: pd.DataFrame) -> None:
    """Function to visualize data in 2.ML_regression_comparison_housing project.

    Args:
        df (pd.DataFrame): Data to visualize.
    """

    logging.info("Function to run all visualizations has started.")

    try:
        formatter = FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
        plt.figure(figsize=(10, 6))
        sns.histplot(df["price"], kde=True, bins=30)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(np.arange(0, 13500001, step=1500000), rotation=45)
        plt.title("Price breakdown")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.savefig(
            os.path.join("images", "price_breakdown_hist.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.histplot(np.log10(df["price"]), bins=30, kde=True)
        plt.savefig(
            os.path.join("images", "log10_price_hist.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        plt.figure(figsize=(12, 8))
        df[["area", "bedrooms", "bathrooms", "floors"]].hist(bins=20, figsize=(12, 8))
        plt.savefig(
            os.path.join("images", "histplots.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
        plt.savefig(
            os.path.join("images", "corr_heatmap.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        binary_cols = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefneighbourhood",
        ]
        for col in binary_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[col], y=df["price"])
            plt.title(f"Price dependence on {col}")
            plt.savefig(
                os.path.join("images", f"{col}_box.png"), dpi=300, bbox_inches="tight"
            )
            plt.close()

        formatter = FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="area", y="price", data=df)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.title("Price dependence on area")
        plt.xlabel("Area")
        plt.ylabel("Price")
        plt.savefig(
            os.path.join("images", "price_area_scatter.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.pairplot(df[["price", "area", "bedrooms", "bathrooms"]])
        plt.savefig(
            os.path.join("images", "pairplot.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        logging.info("Function to run all visualizations has encountered a problem.")
        raise CustomException(e, sys) from e
