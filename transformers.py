"""
This Module is used to store the custom transformers built for this project.
It consists of 3 transformers

1) FeatureSelector: Selects the columns of the DataFrame from the input arguments
2) NumericalTransformer: Generates columns (datatype: float64) and returns array of numerical Data
3) CategoricalTransformer: Generates columns (datatype: obj(str)) and returns array of objects(str)
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline


class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    # Return self, nothing else to do here
    def fit(self, X, y=None) -> None:
        return self

    # Method tthat describes what the transformer needs to do
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Returns the initialized columns in feature_names
        """
        return X[self.feature_names]


class NumericalTransformer(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None) -> None:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Overall transformation steps (e.g. any columns to be dropped / modified)
        """
        X_ = X.copy()
        return X_.values


class CategoricalTransformer(TransformerMixin, BaseEstimator):

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None) -> None:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Overall transformation steps (e.g. any columns to be dropped / modified)
        """

        X_ = X.copy()
        return X_.values
