"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""
from sentence_transformers import SentenceTransformer, evaluation, models, losses, InputExample
import unittest
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

from sentence_transformers.evaluation import MulticlassEvaluator


class MulticlassEvaluatorTest(unittest.TestCase):
    def test_multiclass(self):
        transformer = models.Transformer('prajjwal1/bert-tiny')
        model = SentenceTransformer(modules=[
            transformer,
            models.Pooling(transformer.get_word_embedding_dimension())
        ])
        softmax_loss = losses.SoftmaxLoss(model, transformer.get_word_embedding_dimension(), num_labels=3)

        samples = [
            InputExample(texts=["Hello Word, a first test sentence", "Hello Word, a other test sentence"], label=0),
            InputExample(texts=["Hello Word, a second test sentence", "Hello Word, a other test sentence"], label=1),
            InputExample(texts=["Hello Word, a third test sentence", "Hello Word, a other test sentence"], label=2)
        ]
        dataloader = DataLoader(samples, batch_size=1)
        evaluator = MulticlassEvaluator(dataloader, softmax_model=softmax_loss)
        result = evaluator(model)

        i = 0
        # assert emb.shape == (transformer.get_word_embedding_dimension() * num_heads,)
        #
        # # Single sentence as list
        # emb = model.encode(["Hello Word, a test sentence"])
        # assert emb.shape == (1, transformer.get_word_embedding_dimension() * num_heads

    def test_find_best_f1_and_threshold(self):
        """Tests that the F1 score for the computed threshold is correct"""
        y_true = np.random.randint(0, 2, 1000)
        y_pred_cosine = np.random.randn(1000)
        best_f1, best_precision, best_recall, threshold = evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(
            y_pred_cosine, y_true, high_score_more_similar=True)
        y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
        sklearn_f1score = f1_score(y_true, y_pred_labels)
        assert np.abs(best_f1 - sklearn_f1score) < 1e-6

    def test_find_best_accuracy_and_threshold(self):
        """Tests that the Acc score for the computed threshold is correct"""
        y_true = np.random.randint(0, 2, 1000)
        y_pred_cosine = np.random.randn(1000)
        max_acc, threshold = evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(y_pred_cosine, y_true,
                                                                                                  high_score_more_similar=True)
        y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
        sklearn_acc = accuracy_score(y_true, y_pred_labels)
        assert np.abs(max_acc - sklearn_acc) < 1e-6
