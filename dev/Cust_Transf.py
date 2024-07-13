import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

# Logarithmic Transformer
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)  # log1p is log(1 + X) to handle zero values

# Square Root Transformer
class SqrtTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.sqrt(X)

# Box-Cox Transformer
# Box-Cox Transformer
class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lambdas_ = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            _, self.lambdas_[i] = stats.boxcox(X[:, i] + 1)  # shift data to be positive
        return self
    
    def transform(self, X):
        Xt = np.zeros(X.shape)
        for i in range(X.shape[1]):
            Xt[:, i] = stats.boxcox(X[:, i] + 1, lmbda=self.lambdas_[i])  # apply stored lambda
        return Xt