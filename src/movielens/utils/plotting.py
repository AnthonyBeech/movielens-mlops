# plotting.py
import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class Plotter:
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the plotter with a configuration.
        The config (cfg) is assumed to provide output file names for plots.
        """
        self.cfg = cfg

    def get_predictions_vs_truth_figure(self, truths: list, preds: list) -> plt:
        """
        Generate a Matplotlib figure comparing predictions against true values.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(truths, preds, alpha=0.6)
        ax.plot([min(truths), max(truths)], [min(truths), max(truths)], color="red", lw=2)
        ax.set_xlabel("True Ratings")
        ax.set_ylabel("Predicted Ratings")
        ax.set_title("Predictions vs. True Ratings")
        fig.tight_layout()
        return fig

    def get_error_distribution_figure(self, truths: list, preds: list) -> plt:
        """
        Generate a Matplotlib figure showing a histogram of prediction errors.
        """
        truths_arr = np.array(truths)
        preds_arr = np.array(preds)
        errors = preds_arr - truths_arr

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=30, alpha=0.7)
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Prediction Errors")
        fig.tight_layout()
        return fig

    def log_plots(self, truths: list, preds: list) -> None:
        """
        Generate figures and log them as MLflow artifacts using mlflow.log_figure.
        """
        fig1 = self.get_predictions_vs_truth_figure(truths, preds)
        fig2 = self.get_error_distribution_figure(truths, preds)

        mlflow.log_figure(fig1, artifact_file="plots/pred_vs_truth.png")
        mlflow.log_figure(fig2, artifact_file="plots/error_distribution.png")

        plt.close(fig1)
        plt.close(fig2)

        log.info("Plots logged to MLflow")
