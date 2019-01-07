import numpy as np
import tensorflow as tf
from textify.data import CharacterBasedDataLayer

class CharCNNDataLayer(CharacterBasedDataLayer):

    def _build_features_dataset(self, features_source):
        
        max_len = self._max_len
        vocab = self._vocab
        tokenizer = self._tokenizer
        num_parallel_calls = self._num_parallel_calls

        dataset = tf.data.TextLineDataset(features_source)
        dataset = dataset.map(lambda text: tokenizer(text),
            num_parallel_calls=num_parallel_calls)
        
        dataset = dataset.map(lambda tokens: tokens[:max_len],
            num_parallel_calls=num_parallel_calls)     

        dataset = dataset.map(lambda tokens: tf.cast(vocab.lookup(tokens), tf.int32),
            num_parallel_calls=num_parallel_calls) 

        def pad_(x):

            ids = np.zeros(max_len, dtype=np.int32)
            ids[:x.shape[0]] = x
            return ids
        
        dataset = dataset.map(lambda x: tf.py_func(pad_, [x], [x.dtype]), num_parallel_calls)


        dataset = dataset.map(lambda token_ids: {'ids': token_ids, 'length': tf.size(token_ids)},
            num_parallel_calls=num_parallel_calls) 
        
        dataset = dataset.map(lambda x: {'ids': tf.reshape(x['ids'], [self._max_len]), 'length': x['length']})
        
        return dataset

