"""
Basic utility algorithms and combiners.
"""

import logging
from collections.abc import Iterable, Sequence

import pandas as pd
import numpy as np

from ..data import sparse_ratings
from . import Predictor, Recommender, CandidateSelector
from ..util import derivable_rng

from .bias import Bias  # noqa: F401
from .ranking import TopN  # noqa: F401

_logger = logging.getLogger(__name__)


class Popular(Recommender):
    """
    Recommend the most popular items.

    The :py:class:`PopScore` class is more flexible, and recommended for new code.

    Args:
        selector(CandidateSelector):
            The candidate selector to use. If ``None``, uses a new
            :class:`UnratedItemCandidateSelector`.

    Attributes:
        item_pop_(pandas.Series):
            Item rating counts (popularity)
    """

    def __init__(self, selector=None):
        self.selector = selector

    def fit(self, ratings, **kwargs):
        pop = ratings.groupby('item').user.count()
        pop.name = 'score'
        self.item_pop_ = pop.astype('float64')

        if self.selector is None:
            self.selector = UnratedItemCandidateSelector()
        self.selector.fit(ratings)

        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        scores = self.item_pop_
        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

        idx = scores.index.get_indexer(candidates)
        idx = idx[idx >= 0]
        scores = scores.iloc[idx]

        if n is None:
            return scores.sort_values(ascending=False).reset_index()
        else:
            return scores.nlargest(n).reset_index()

    def __str__(self):
        return 'Popular'


class PopScore(Predictor):
    """
    Score items by their popularity.  Use with :py:class:`TopN` to get a
    most-popular-items recommender.

    Args:
        score_type(str):
            The method for computing popularity scores.  Can be one of the following:

            - ``'quantile'`` (the default)
            - ``'rank'``
            - ``'count'``

    Attributes:
        item_pop_(pandas.Series):
            Item popularity scores.
    """

    def __init__(self, score_method='quantile'):
        self.score_method = score_method

    def fit(self, ratings, **kwargs):
        _logger.info('counting item popularity')
        scores = ratings['item'].value_counts()
        if self.score_method == 'rank':
            _logger.info('ranking %d items', len(scores))
            scores = scores.rank().sort_index()
        elif self.score_method == 'quantile':
            _logger.info('computing quantiles for %d items', len(scores))
            cmass = scores.sort_values()
            cmass = cmass.cumsum()
            cdens = cmass / scores.sum()
            scores = cdens.sort_index()
        elif self.score_method == 'count':
            _logger.info('scoring items with their rating counts')
            scores = scores.sort_index()
        else:
            raise ValueError('invalid scoring method ' + repr(self.score_method))

        self.item_scores_ = scores

        return self

    def predict_for_user(self, user, items, ratings=None):
        return self.item_scores_.reindex(items)

    def __str__(self):
        return 'PopScore({})'.format(self.score_method)


class Memorized(Predictor):
    """
    The memorized algorithm memorizes socres provided at construction time.
    """

    def __init__(self, scores):
        """
        Args:
            scores(pandas.DataFrame): the scores to memorize.
        """

        self.scores = scores

    def fit(self, *args, **kwargs):
        return self

    def predict_for_user(self, user, items, ratings=None):
        uscores = self.scores[self.scores.user == user]
        urates = uscores.set_index('item').rating
        return urates.reindex(items)


class Fallback(Predictor):
    """
    The Fallback algorithm predicts with its first component, uses the second to fill in
    missing values, and so forth.
    """

    def __init__(self, algorithms, *others):
        """
        Args:
            algorithms: a list of component algorithms.  Each one will be trained.
            others:
                additional algorithms, in which case ``algorithms`` is taken to be
                a single algorithm.
        """
        if others:
            self.algorithms = [algorithms] + list(others)
        elif isinstance(algorithms, Iterable) or isinstance(algorithms, Sequence):
            self.algorithms = algorithms
        else:
            self.algorithms = [algorithms]

    def fit(self, ratings, **kwargs):
        for algo in self.algorithms:
            algo.fit(ratings, **kwargs)

        return self

    def predict_for_user(self, user, items, ratings=None):
        remaining = pd.Index(items)
        preds = None

        for algo in self.algorithms:
            _logger.debug('predicting for %d items for user %s', len(remaining), user)
            aps = algo.predict_for_user(user, remaining, ratings=ratings)
            aps = aps[aps.notna()]
            if preds is None:
                preds = aps
            else:
                preds = pd.concat([preds, aps])
            remaining = remaining.difference(preds.index)
            if len(remaining) == 0:
                break

        return preds.reindex(items)

    def __str__(self):
        str_algos = [str(algo) for algo in self.algorithms]
        return 'Fallback([{}])'.format(', '.join(str_algos))


class EmptyCandidateSelector(CandidateSelector):
    """
    :class:`CandidateSelector` that never returns any candidates.
    """

    dtype_ = np.int64

    def fit(self, ratings, **kwarsg):
        self.dtype_ = ratings['item'].dtype

    def candidates(self, user, ratings=None):
        return np.array([], dtype=self.dtype_)


class UnratedItemCandidateSelector(CandidateSelector):
    """
    :class:`CandidateSelector` that selects items a user has not rated as
    candidates.  When this selector is fit, it memorizes the rated items.

    Attributes:
        items_(pandas.Index): All known items.
        users_(pandas.Index): All known users.
        user_items_(CSR):
            Items rated by each known user, as positions in the ``items`` index.
    """
    items_ = None
    users_ = None
    user_items_ = None

    def fit(self, ratings, **kwargs):
        r2 = ratings[['user', 'item']]
        sparse = sparse_ratings(r2)
        _logger.info('trained unrated candidate selector for %d ratings', sparse.matrix.nnz)
        self.items_ = sparse.items
        self.users_ = sparse.users
        self.user_items_ = sparse.matrix

        return self

    def candidates(self, user, ratings=None):
        if ratings is None:
            try:
                uidx = self.users_.get_loc(user)
                uis = self.user_items_.row_cs(uidx)
            except KeyError:
                uis = None
        else:
            uis = self.items_.get_indexer(self.rated_items(ratings))
            uis = uis[uis >= 0]

        if uis is not None:
            mask = np.full(len(self.items_), True)
            mask[uis] = False
            return self.items_.values[mask]
        else:
            return self.items_.values


class AllItemsCandidateSelector(CandidateSelector):
    """
    :class:`CandidateSelector` that selects all items, regardless of whether
    the user has rated them, as candidates.  When this selector is fit, it memorizes
    the set of items.

    Attributes:
        items_(numpy.ndarray): All known items.
    """
    items_ = None

    def fit(self, ratings, **kwargs):
        self.items_ = ratings['item'].unique()
        return self

    def candidates(self, user, ratings=None):
        return self.items_.copy()


class Random(Recommender):
    """
    A random-item recommender.

    Attributes:
        selector(CandidateSelector):
            Selects candidate items for recommendation.
            Default is :class:`UnratedItemCandidateSelector`.
        rng_spec:
            Seed or random state for generating recommendations.  Pass
            ``'user'`` to deterministically derive per-user RNGS from
            the user IDs for reproducibility.
    """

    def __init__(self, selector=None, rng_spec=None):
        if selector:
            self.selector = selector
        else:
            self.selector = UnratedItemCandidateSelector()
        # Get a Pandas-compatible RNG
        self.rng_source = derivable_rng(rng_spec, legacy=True)
        self.items = None

    def fit(self, ratings, **kwargs):
        self.selector.fit(ratings, **kwargs)
        items = pd.DataFrame(ratings['item'].unique(), columns=['item'])
        self.items = items
        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        if candidates is None:
            candidates = self.selector.candidates(user, ratings)
        if n is None:
            n = len(candidates)

        rng = self.rng_source(user)
        c_df = pd.DataFrame(candidates, columns=['item'])
        recs = c_df.sample(n, random_state=rng)
        return recs.reset_index(drop=True)

    def __str__(self):
        return 'Random'


class KnownRating(Predictor):
    """
    The known rating algorithm memorizes ratings provided in the fit method.
    """

    def fit(self, ratings, **kwargs):
        self.ratings = ratings.set_index(['user', 'item']).sort_index()
        return self

    def predict_for_user(self, user, items, ratings=None):
        uscores = self.ratings.xs(user, level='user', drop_level=True)
        return uscores.rating.reindex(items)
