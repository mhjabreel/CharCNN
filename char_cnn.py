import tensorflow as tf

from textify.models import _Classifier
from textify.layers.embeddings import Embedding



class CharCNN(_Classifier):

    def _get_embeddings(self, features, mode=tf.estimator.ModeKeys.TRAIN):
        
        embedding = Embedding(self._params['embedding_specs'])
        embedding.build(None)
        return embedding.call(features)          
            
        
    def _encode(self, embeddings, lengths=None, mode=tf.estimator.ModeKeys.TRAIN):

        conv_layers = self._params.get('conv_layers', None)
        
        if conv_layers is None:
            conv_layers = [
                [256, 7, 3],
                [256, 7, 3],
                [256, 3, None],
                [256, 3, None],
                [256, 3, None],
                [256, 3, 3]
                ]

        x = embeddings

        vec_dim = self._params.get('seq_len', 1014)

        for i, cl in enumerate(conv_layers):
            
            vec_dim -= (cl[1] - 1)

            x = tf.layers.conv1d(x, cl[0], cl[1], activation=tf.nn.relu, name='Conv_%d' % i)
            if not cl[2] is None:
                
                vec_dim -= cl[2]
                vec_dim //= cl[2]
                vec_dim += 1
                
                x =  tf.layers.max_pooling1d(x, cl[2], cl[2], name='Pool_%d' % i)
        
        vec_dim *= cl[0]
        
        x = tf.reshape(x, [-1, vec_dim])

        return x