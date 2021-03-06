import json
import os
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
        self.config_keys = ['word_embedding_dimension']

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        # (batch, seq_len, word_dim)
        f = F.relu(self.fc1(token_embeddings))
        sentence_attention = F.softmax(self.fc2(f), dim=-2)

        # (batch, word_dim)
        output_vector = torch.sum(sentence_attention * token_embeddings, dim=-2)
        features['sentence_embedding'] = output_vector
        features['sentence_attention'] = f
        return features

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'model.pth'))

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = GeneralizedPooling(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'model.pth'), map_location=torch.device('cpu')))
        return model


class GeneralizedMultiheadPooling(nn.Module):
    """
    Based on https://arxiv.org/pdf/1806.09828.pdf
    Enhancing Sentence Embedding with Generalized Pooling
    """
    def __init__(self, word_embedding_dimension: int, num_heads: int = 2):
        super().__init__()
        self.poolings = nn.ModuleList([GeneralizedPooling(word_embedding_dimension) for _ in range(num_heads)])
        self.config_keys = ['word_embedding_dimension', 'num_heads']
        self.word_embedding_dimension = word_embedding_dimension
        self.num_heads = num_heads

    def forward(self, features: Dict[str, Tensor]):
        outputs = []

        for pooling in self.poolings:
            outputs.append(pooling(features)['sentence_embedding'])

        features['sentence_embedding'] = torch.cat(outputs, dim=-1)
        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'model.pth'))

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension * self.num_heads

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = GeneralizedMultiheadPooling(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'model.pth'), map_location=torch.device('cpu')))
        return model
