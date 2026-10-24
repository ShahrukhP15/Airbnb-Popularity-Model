"""Fold-local Airbnb features shared by the notebook and saved models."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ListingFeatures(BaseEstimator, TransformerMixin):
    """Learn neighborhood review totals exclusively from the fit partition.

    The transform partition's own review counts are never used. Neighborhoods
    absent from the fit partition have no observed review history (total zero).
    """

    def fit(self, X, y=None):
        self.neighborhood_totals_ = X.groupby('neighbourhood')['number_of_reviews'].sum()
        return self

    def transform(self, X):
        check_is_fitted(self, 'neighborhood_totals_')
        result = X.copy()
        result['minimum_stay_price'] = result['price'] * result['minimum_nights']
        result['neighbourhood_popularity'] = (
            result['neighbourhood'].map(self.neighborhood_totals_).fillna(0.0)
        )
        return result
