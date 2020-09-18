import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import dot, transpose, categorical_crossentropy, stack, shape

class ForwardCorrectedModel(tf.keras.Model):

    def __init__(self, model, transition_matrixes):
        super(ForwardCorrectedModel, self).__init__()
        self.__model = model
        self.transition_matrixes = transition_matrixes

    def call(self, inputs):
        return self.__model.call(inputs)

    def evaluate(self, X, y, **kwargs):
        return self.__model.evaluate(X, y, **kwargs)
        
    def fit(self, X, y, sensitive, **kwargs):
        self.sensitive_size = shape(sensitive)[1]
        y = np.hstack([sensitive, y])
        return self.__model.fit(X, y, **kwargs)

    def compile(self, **kwargs):
        kwargs['loss'] = self.forward_loss(kwargs['loss'])
        return self.__model.compile(**kwargs)

    def forward_loss(self, base_loss):

        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true,y_pred):
            T = K.stack([ transition_matrix.T for transition_matrix in self.transition_matrixes])
            y_sensitive = y_true[:,0:self.sensitive_size]
            y_true_target = y_true[:,-self.sensitive_size:]
            T_volume = tf.tensordot(y_sensitive, T, axes=1)
        
            y_pred_target_corrected = K.batch_dot(T_volume, y_pred)
            
            return base_loss(y_true_target, y_pred_target_corrected)
    
        # Return a function
        return loss
