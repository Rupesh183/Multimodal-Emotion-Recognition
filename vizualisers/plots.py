import os
from typing import List
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from core.config import CONFIG


class PlotVisualizer:
    @classmethod
    def plot_history(cls, history: List[float], title: str):
        plt.plot(history)
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.6)

    @classmethod
    def plot_many(cls, dims: tuple, *args, filename: str = None):
        assert dims[0] * dims[1] == len(args)
        plt.figure(figsize=(12, 5))
        for idx, arg in enumerate(args, start=1):
            plt.subplot(*dims, idx)
            arg()
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            out_file = os.path.join(CONFIG.save_plots_location(), filename)
            plt.savefig(out_file, dpi=300)
            plt.close()

    @classmethod
    def plot_confusion_matrix(
        cls, conf_matrix, labels: List[str], filename: str = None
    ):
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            out_file = os.path.join(CONFIG.save_plots_location(), filename)
            plt.savefig(out_file, dpi=300)
            plt.close()
