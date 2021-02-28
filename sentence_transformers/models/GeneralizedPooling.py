from typing import Dict

import torch
from torch import nn, Tensor
from torch.functional import F


class GeneralizedPooling(nn.Module):
    def __init__(self, word_embedding_dimension: int):
        super().__init__()
        self.fc1 = nn.Linear(word_embedding_dimension, word_embedding_dimension)
        self.fc2 = nn.Linear(word_embedding_dimension, word_embedding_dimension)
        self.word_embedding_dimension = word_embedding_dimension

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        # (batch, seq_len, word_dim)
        f = F.relu(self.fc1(token_embeddings))
        f = F.softmax(self.fc2(f), dim=-2)

        # (batch, word_dim)
        output_vector = torch.sum(f * token_embeddings, dim=-2)
        features['sentence_embedding'] = output_vector
        return features


class GeneralizedMultiheadPooling(nn.Module):
    def __init__(self, word_embedding_dimension: int, num_heads: int=2):
        super().__init__()
        self.poolings = nn.ModuleList([GeneralizedPooling(word_embedding_dimension) for _ in range(num_heads)])

    def forward(self, features: Dict[str, Tensor]):
        outputs = []

        for pooling in self.poolings:
            outputs.append(pooling(features)['sentence_embedding'])

        features['sentence_embedding'] = torch.cat(outputs, axis=-1)
        return features