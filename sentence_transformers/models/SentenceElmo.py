from typing import List

import torch
from torch import nn
from allennlp.modules.elmo import Elmo, batch_to_ids

from sentence_transformers.models.tokenizer import WordTokenizer


class SentenceElmo(nn.Module):
    def __init__(self, options_file, weight_file, tokenizer, average_mod='mean'):
        super().__init__()
        assert average_mod in {'mean', 'max', 'last'}

        self.elmo = Elmo(options_file=options_file, weight_file=weight_file, num_output_representations=1,
                         requires_grad=True)

        self.tokenizer = tokenizer
        self.average_mod = average_mod

    def get_word_embedding_dimension(self) -> int:
        return self.elmo.get_output_dim()

    def forward(self, features):
        output = self.elmo(features['input_ids'])
        token_embeddings = output['elmo_representations'][0]

        features = {}
        if self.average_mod == 'mean':
            features['sentence_embedding'] = token_embeddings.mean(axis=1)
        elif self.average_mod == 'max':
            features['sentence_embedding'] = token_embeddings.max(axis=1).values
        else:
            last_token_indices = output['mask'].sum(axis=1) - 1
            features['sentence_embedding'] = token_embeddings[torch.arange(token_embeddings.shape[0]),
                                             last_token_indices, :]

        return features

    def tokenize(self, texts: List[str]):
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        input_ids = batch_to_ids(tokenized_texts)

        output = {'input_ids': input_ids}
        return output
