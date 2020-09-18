import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import dot, transpose, categorical_crossentropy, stack, shape
import numpy as np
import abc

class FairCorrectedModel(tf.keras.Model, metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'correction_method') and 
                callable(subclass.correction_method) and 
                NotImplemented)

    @abc.abstractmethod
    def correction_method(self, y_true, y_pred, y_sensitive):
        """create a corrected version of base_loss"""
        raise NotImplementedError
    
    def __init__(self, model):
        super(FairCorrectedModel, self).__init__()
        self.__model = model

    def call(self, inputs):
        return self.__model.call(inputs)

    def evaluate(self, X, y, **kwargs):
        return self.__model.evaluate(X, y, **kwargs)
        
    def fit(self, X, y, sensitive, **kwargs):
        self.sensitive_size = shape(sensitive)[1]
        y = np.hstack([sensitive, y])
        return self.__model.fit(X, y, **kwargs)

    def compile(self, **kwargs):
        kwargs['loss'] = self.correct_loss(kwargs['loss'])
        return self.__model.compile(**kwargs)
    
    def correct_loss(self, base_loss):
        def loss(y_true,y_pred):
            y_sensitive = y_true[:,0:self.sensitive_size]
            y_true_target = y_true[:,-self.sensitive_size:]
            y_pred_corrected = self.correction_method(y_true_target, y_pred, y_sensitive)  
            return base_loss(y_true_target, y_pred_corrected)
        return loss


class ForwardCorrectedModel(FairCorrectedModel):

    def __init__(self, model, transition_matrixes):
        super(ForwardCorrectedModel, self).__init__(model)
        self.transition_matrixes = transition_matrixes

    def correction_method(self, y_true, y_pred, y_sensitive):
        T = K.stack([ transition_matrix.T for transition_matrix in self.transition_matrixes])
        T_volume = tf.tensordot(y_sensitive, T, axes=1)
        y_pred_corrected = K.batch_dot(T_volume, y_pred)
        return y_pred_corrected        