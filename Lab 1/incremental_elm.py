import numpy as np

class IncrementalELM:
    def __init__(self, n_hidden=20, random_state=None):
        self.n_hidden = n_hidden
        self.random_state = np.random.RandomState(random_state)
        self.input_weights = None
        self.biases = None
        self.output_weights = None
        self.H_pseudo_inv = None

    def _activation(self, x):
        return 1 / (1 + np.exp(-x))

    def partial_fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y).T

        if self.input_weights is None:
            n_features = X.shape[1]
            self.input_weights = self.random_state.uniform(-1, 1, (self.n_hidden, n_features))
            self.biases = self.random_state.uniform(-1, 1, (self.n_hidden, 1))
            self.output_weights = np.zeros((self.n_hidden, 1))
            self.H_pseudo_inv = None

        H = self._activation(np.dot(self.input_weights, X.T) + self.biases)

        if self.H_pseudo_inv is None:
            H = H.T
            HTH = H.T @ H
            self.H_pseudo_inv = np.linalg.pinv(H)
            self.output_weights = self.H_pseudo_inv @ y
        else:
            y_pred = (self.output_weights.T @ H).flatten()
            error = y.flatten() - y_pred
            lr = 0.01
            self.output_weights += lr * H @ error[:, None]

    def predict(self, X):
        X = np.atleast_2d(X)
        H = self._activation(np.dot(self.input_weights, X.T) + self.biases)
        y_pred = (self.output_weights.T @ H).flatten()
        y_pred_class = np.clip(np.round(y_pred), 1, 5).astype(int)
        return y_pred_class
