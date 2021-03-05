from . import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class MulticlassEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, average: str = 'micro',
                 main_metric: str = 'f1_micro', write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        assert average in {'micro', 'macro', 'weighted'}
        assert average in {"accuracy", f"precision_{average}", f"recall_{average}", f"f1_{average}"}

        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model
        self.average = average
        self.main_metric = main_metric

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.csv_file = "multiclass_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", f"precision_{average}", f"recall_{average}", f"f1_{average}"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1,
                 global_step: int = -1) -> float:
        model.eval()

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        predictions, labels = [], []
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)

            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            predictions.append(torch.argmax(prediction, dim=1).detach().cpu().numpy())
            labels.append(label_ids.detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average=self.average)
        recall = recall_score(labels, predictions, average=self.average)
        f1 = f1_score(labels, predictions, average=self.average)

        computed_metrics = dict(zip(self.csv_headers[2:], [accuracy, precision, recall, f1]))

        logger.info("Accuracy: {:.4f}\n".format(accuracy))

        if wandb_available and wandb.run is not None:
            wandb.log(computed_metrics, step=global_step)

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy, precision, recall, f1])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy, precision, recall, f1])

        return computed_metrics[self.main_metric]
