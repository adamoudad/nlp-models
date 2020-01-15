import random

import numpy as np
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequence

class NextTokenGenerator(Sequence):
    """
    Generator for training neural network on next token prediction
    Takes a file as input data and iterates over its characters.
    Inherits from keras.utils.Sequence class (see https://keras.io/utils/#sequence)
    Therefore it can be passed directly to fit_generator method of a Keras model.
    Not implemented: manual setting of character_set, max_length via __init__ parameters (they are currently inferred from the dataset)
    """
    def __init__(self,
                 data_file,
                 batch_size,
                 character_set="",
                 max_length=0,
                 shuffle=True):
        if character_set or max_length: raise NotImplementedError("Manual setting of character set and max length is not available. Will be inferred from the dataset.")
        self.data = []
        self.character_set = set()
        self.batch_size = batch_size
        self.max_length = 0
        self.load(data_file)
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(self.data)
        if shuffle: random.shuffle(self.data)
    
    def load_data(self, data_path):
        """
        Load data from file, process text and and extract the character set
        Processing involves
        - all characters to lowercase
        """
        with data_file.open("r") as f:
            datum = ""
            for line in f:
                if line[:5] == 5*"=":
                    self.character_set.update(set(datum))
                    self.data.append(datum)
                    self.max_length = max(self.max_length, len(datum))
                    datum = ""
                else:
                    datum += line.lower()
            if datum:           # Last datum of the file
                self.character_set.union(set(datum))
                self.data.append(datum)
                self.max_length = max(self.max_length, len(datum))

    def preprocess(self):
        self.data = pad_sequence(self.data)
        
    def __len__(self):
        """
        Inspired by https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L331
        """
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, index):
        """
        https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L334
        """
        batch = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        return list(map(self.tokenizer.texts_to_matrix, batch))  # Convert to a batch of sequences of one-hot vectors

