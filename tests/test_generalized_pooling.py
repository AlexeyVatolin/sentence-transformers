import unittest

from sentence_transformers import models
from sentence_transformers import SentenceTransformer


class MyTestCase(unittest.TestCase):
    def test_generalized_pooling(self):
        transformer = models.Transformer('prajjwal1/bert-tiny')
        model = SentenceTransformer(modules=[
            transformer,
            models.GeneralizedPooling(transformer.get_word_embedding_dimension())
        ])

        emb = model.encode("Hello Word, a test sentence")
        assert emb.shape == (transformer.get_word_embedding_dimension(), )

        # Single sentence as list
        emb = model.encode(["Hello Word, a test sentence"])
        assert emb.shape == (1, transformer.get_word_embedding_dimension())

    def test_multiheadgeneralized_pooling(self):
        transformer = models.Transformer('prajjwal1/bert-tiny')
        num_heads = 5
        model = SentenceTransformer(modules=[
            transformer,
            models.GeneralizedMultiheadPooling(transformer.get_word_embedding_dimension(), num_heads=num_heads)
        ])

        emb = model.encode("Hello Word, a test sentence")
        assert emb.shape == (transformer.get_word_embedding_dimension() * num_heads, )

        # Single sentence as list
        emb = model.encode(["Hello Word, a test sentence"])
        assert emb.shape == (1, transformer.get_word_embedding_dimension() * num_heads)


if __name__ == '__main__':
    unittest.main()
