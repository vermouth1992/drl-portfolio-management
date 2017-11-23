"""
Contain an abstract base model that all the subclass need to follow the API
"""


class BaseModel(object):
    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length, num_features).
            Feature contains (open, high, low, close)

        Returns: action to take at next timestamp. A numpy array shape (num_stocks + 1,)

        """
        raise NotImplementedError('This method must be implemented by subclass')
