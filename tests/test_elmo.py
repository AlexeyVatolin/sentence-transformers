import unittest

import razdel

from sentence_transformers.models.SentenceElmo import SentenceElmo


class RazdelTokenizer:
    def tokenize(self, text):
        return [x.text for x in razdel.tokenize(text)]


class MyTestCase(unittest.TestCase):
    def test_something(self):
        options_file = "/Users/a17881256/Downloads/195/options.json"
        weight_file = "/Users/a17881256/Downloads/195/model.hdf5"
        sentences = ['тест текст .', 'другой .']
        tokenizer = RazdelTokenizer()
        elmo = SentenceElmo(options_file, weight_file, tokenizer)

        tokenized = elmo.tokenize(sentences)
        output = elmo(tokenized)
        i = 0


if __name__ == '__main__':
    unittest.main()
