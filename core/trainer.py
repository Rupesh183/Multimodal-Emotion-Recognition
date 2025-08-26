import logging
import numpy as np
import tqdm
from typing import List

import torch
import torch.nn as nn
import transformers
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from vizualisers.plots import PlotVisualizer

logger = logging.getLogger(__name__)


class AbstractTrainer:
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        optimizer,
        loss: callable,
        name: str,
    ):
        self.model = model
        self._num_epochs = num_epochs
        self._optimizer = optimizer
        self._loss = loss
        self._name = name

    def _get_logits_and_real(self, batch) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError("Subclasses must implement _get_logits_and_real()")

    def train(self, train_dataloader: DataLoader):
        total_steps = len(train_dataloader) * self._num_epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        history_loss = []
        history_acc = []

        logger.info(f"Starting training for {self._name} model across {self._num_epochs} epochs...")
        self.model.train()
        for epoch in range(1, self._num_epochs + 1):
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}/{self._num_epochs}")
            for batch in loader:
                logits, real = self._get_logits_and_real(batch)
                loss = self._loss(logits, real)

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self._optimizer.step()
                scheduler.step()

                preds = torch.argmax(logits, dim=1)
                accuracy = torch.mean((preds == real).float()).item()
                loss_val = loss.item()

                history_loss.append(loss_val)
                history_acc.append(accuracy)
                loader.set_postfix(loss=f"{loss_val:.4f}", acc=f"{accuracy:.4f}")

        self.plot_histories(history_loss, history_acc)

    def plot_histories(
        self, history_losses: List[float], history_accuracies: List[float]
    ):
        PlotVisualizer.plot_many(
            (1, 2),
            lambda: PlotVisualizer.plot_history(
                history_losses, f"{self._name.capitalize()} Loss History"
            ),
            lambda: PlotVisualizer.plot_history(
                history_accuracies, f"{self._name.capitalize()} Accuracy History"
            ),
            filename=f"{self._name}-losses.png",
        )

    def eval(self, test_dataloader: DataLoader, labels: List[str] = None):
        y_actual = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            loader = tqdm.tqdm(test_dataloader, desc=f"Evaluating {self._name}")
            for batch in loader:
                logits, real = self._get_logits_and_real(batch)
                preds = torch.argmax(logits, dim=1)
                y_actual.extend(real.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        possible_values = sorted(list(set(y_actual) | set(y_pred)))
        possible_labels = [
            labels[i] if i < len(labels) else f"Class_{i}" for i in possible_values
        ]

        conf_matrix = confusion_matrix(y_actual, y_pred)
        acc = float(np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix))
        print(f"\n{self._name.upper()} Evaluation Accuracy: {acc * 100:.2f}%\n")
        PlotVisualizer.plot_confusion_matrix(
            conf_matrix, possible_labels, filename=f"{self._name}-conf_matrix.png"
        )
        return acc
