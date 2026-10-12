"""Market Regime Discovery Module for RACM-Forecast.

Provides market regime identification using Hidden Markov Models (GaussianHMM)
with graceful fallback to Gaussian Mixture Models, semantic regime mapping,
and change-point detection.
"""

from enum import IntEnum
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.mixture import GaussianMixture

# Optional HMM and change-point imports with graceful fallbacks
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    GaussianHMM = None
    HAS_HMMLEARN = False

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    rpt = None
    HAS_RUPTURES = False

logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    """Semantic market regimes mapped to canonical integer values."""
    BULL = 0
    BEAR = 1
    HIGH_VOLATILITY = 2
    CRISIS = 3


class MarketRegimeDetector:
    """Discovers market regimes and maps them to canonical semantic categories.

    Uses a Gaussian Hidden Markov Model (or Gaussian Mixture Model fallback)
    fitted on time series features such as returns and volatility estimates. Discovered
    hidden states are ordered and mapped to:
        - 0: BULL (high positive return, lower volatility)
        - 1: BEAR (negative return, moderate volatility)
        - 2: HIGH_VOLATILITY (modest return, high volatility)
        - 3: CRISIS (most negative return, highest volatility)
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 42) -> None:
        """Initialize the MarketRegimeDetector.

        Args:
            n_regimes: Number of regime states to model (default 4).
            random_state: Random seed for reproducibility.
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model: Optional[Union[GaussianHMM, GaussianMixture]] = None
        self._state_to_regime: Dict[int, int] = {}
        self.is_fitted: bool = False

    def _init_model(self) -> Union[GaussianHMM, GaussianMixture]:
        """Instantiate either GaussianHMM or fallback GaussianMixture."""
        if HAS_HMMLEARN and GaussianHMM is not None:
            try:
                return GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=100,
                    random_state=self.random_state,
                )
            except Exception as e:
                logger.warning("Failed to initialize GaussianHMM (%s). Falling back to GaussianMixture.", e)

        return GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            max_iter=100,
            random_state=self.random_state,
        )

    def _extract_state_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mean return and volatility for each discovered state."""
        if hasattr(self.model, "means_"):
            means = self.model.means_
        else:
            raise RuntimeError("Model does not have fitted means.")

        n_components = self.n_regimes
        means = np.asarray(means)

        if means.shape[1] >= 2:
            # Feature 0 is return, Feature 1 is volatility estimate
            returns = means[:, 0]
            vols = means[:, 1]
        else:
            returns = means[:, 0]
            # Derive volatility from covariance matrix
            if hasattr(self.model, "covars_"):
                covars = self.model.covars_
                if covars.ndim == 3:  # (n_components, n_features, n_features)
                    vols = np.sqrt(np.maximum(covars[:, 0, 0], 1e-8))
                elif covars.ndim == 2:  # (n_components, n_features)
                    vols = np.sqrt(np.maximum(covars[:, 0], 1e-8))
                else:
                    vols = np.sqrt(np.maximum(covars, 1e-8))
            else:
                vols = np.ones(n_components)

        return returns, vols

    def _build_semantic_mapping(self, returns: np.ndarray, vols: np.ndarray) -> Dict[int, int]:
        """Sort and map discovered states into semantic regimes:
        - Bull (0): high positive return, lower volatility
        - Bear (1): negative return, moderate volatility
        - High Volatility (2): modest return, high volatility
        - Crisis (3): most negative return with highest volatility
        """
        mapping: Dict[int, int] = {}

        if self.n_regimes == 4:
            available_states = list(range(4))

            # 1. Bull: State with the highest mean return
            bull_state = int(max(available_states, key=lambda s: returns[s]))
            available_states.remove(bull_state)

            # 2. Crisis: State with lowest return and extreme volatility
            # In stress periods, returns are strongly negative and volatility is maximized
            crisis_state = int(min(available_states, key=lambda s: (returns[s] - vols[s])))
            available_states.remove(crisis_state)

            # 3. Remaining states are Bear and High Volatility
            # High Volatility has higher volatility than Bear
            state_a, state_b = available_states[0], available_states[1]
            if vols[state_a] > vols[state_b]:
                high_vol_state, bear_state = state_a, state_b
            else:
                high_vol_state, bear_state = state_b, state_a

            mapping[bull_state] = int(MarketRegime.BULL)
            mapping[bear_state] = int(MarketRegime.BEAR)
            mapping[high_vol_state] = int(MarketRegime.HIGH_VOLATILITY)
            mapping[crisis_state] = int(MarketRegime.CRISIS)
        else:
            # Fallback for arbitrary regime counts: rank by return descending
            sorted_by_return = sorted(range(self.n_regimes), key=lambda s: returns[s], reverse=True)
            for rank, state in enumerate(sorted_by_return):
                mapping[state] = rank

        return mapping

    def fit(self, X: np.ndarray) -> "MarketRegimeDetector":
        """Fit regime detector on feature array X.

        Args:
            X: Input time series features of shape (n_samples, n_features),
               typically where feature 0 is return and feature 1 is volatility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model = self._init_model()

        try:
            model.fit(X)
            self.model = model
        except Exception as e:
            logger.warning("Primary model fit failed (%s). Retrying with GaussianMixture.", e)
            fallback = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                max_iter=100,
                random_state=self.random_state,
            )
            fallback.fit(X)
            self.model = fallback

        returns, vols = self._extract_state_statistics()
        self._state_to_regime = self._build_semantic_mapping(returns, vols)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict market regime for each sample in X.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            1D integer array of shape (n_samples,) with values in {0, 1, 2, 3}.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("MarketRegimeDetector must be fitted before predict().")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        raw_preds = self.model.predict(X)
        mapped_preds = np.array(
            [self._state_to_regime.get(int(s), int(s)) for s in raw_preds],
            dtype=np.int64,
        )
        return mapped_preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior regime probabilities for each sample in X.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            2D float array of posterior probabilities of shape (n_samples, 4).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("MarketRegimeDetector must be fitted before predict_proba().")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        raw_proba = self.model.predict_proba(X)
        n_samples = X.shape[0]
        mapped_proba = np.zeros((n_samples, self.n_regimes), dtype=np.float64)

        for state_idx, regime_idx in self._state_to_regime.items():
            if regime_idx < self.n_regimes and state_idx < raw_proba.shape[1]:
                mapped_proba[:, regime_idx] = raw_proba[:, state_idx]

        # Re-normalize to ensure sum to 1.0
        row_sums = mapped_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        mapped_proba = mapped_proba / row_sums

        return mapped_proba

    def detect_change_points(self, X: np.ndarray, n_bkps: int = 5) -> List[int]:
        """Detect structural change points in time series X.

        Uses ruptures Binseg/Pelt if available, with a sliding-window
        variance/mean shift fallback.

        Args:
            X: Time series array of shape (n_samples, n_features) or (n_samples,).
            n_bkps: Number of breakpoints to detect.

        Returns:
            List of breakpoint indices.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)
        if n_samples <= n_bkps + 1:
            return list(range(1, n_samples + 1))

        if HAS_RUPTURES and rpt is not None:
            try:
                algo = rpt.Binseg(model="rbf").fit(X)
                bkps = algo.predict(n_bkps=n_bkps)
                return [int(b) for b in bkps if 0 <= b <= n_samples]
            except Exception as e:
                logger.warning("Ruptures change-point detection failed (%s). Using fallback.", e)

        # Sliding window variance/mean shift fallback
        window_size = max(5, n_samples // (2 * (n_bkps + 1)))
        shifts: List[Tuple[float, int]] = []

        for t in range(window_size, n_samples - window_size):
            pre = X[t - window_size : t]
            post = X[t : t + window_size]
            mean_diff = np.linalg.norm(np.mean(post, axis=0) - np.mean(pre, axis=0))
            var_diff = np.linalg.norm(np.var(post, axis=0) - np.var(pre, axis=0))
            shift_score = float(mean_diff + var_diff)
            shifts.append((shift_score, t))

        # Select top non-overlapping breakpoints
        shifts.sort(key=lambda item: item[0], reverse=True)
        selected_bkps: List[int] = []

        for _, t in shifts:
            if len(selected_bkps) >= n_bkps:
                break
            if all(abs(t - chosen) >= window_size for chosen in selected_bkps):
                selected_bkps.append(t)

        selected_bkps.sort()
        # Append n_samples to conform with standard breakpoint conventions
        if not selected_bkps or selected_bkps[-1] != n_samples:
            selected_bkps.append(n_samples)

        return selected_bkps
