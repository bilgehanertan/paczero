"""
PAC Privacy utilities for PACZero (sign-quantized mechanism).
"""

import numpy as np
import torch
from scipy.special import roots_hermitenorm


# ---- Exact binary-channel MI via Gauss-Hermite quadrature ------------------

# Nodes/weights for f(x) integrated against standard normal density.
# sum_i w_i f(x_i) ~= E[f(N(0,1))] (probabilist's Hermite).
_GH_NODES, _GH_WEIGHTS = roots_hermitenorm(60)
_GH_WEIGHTS = _GH_WEIGHTS / np.sqrt(2 * np.pi)  # normalize to probability density


def binary_channel_mi(q_plus, sigma):
    """
    I(S; S + N(0, sigma^2)) where S in {-1, +1} with P(S=+1) = q_plus.

    Computed exactly in nats via Gauss-Hermite quadrature. Robust to
    small/large sigma: integration grid is the {-1,+1}-centered mixture,
    and log-sum-exp is used for numerical stability.

    Returns I in nats (>= 0).
    """
    if q_plus <= 0.0 or q_plus >= 1.0:
        return 0.0  # deterministic S
    if sigma <= 0.0:
        # noiseless: I = H(S) = -q*log q - (1-q)*log(1-q)
        return float(-q_plus * np.log(q_plus) - (1 - q_plus) * np.log(1 - q_plus))

    q_minus = 1.0 - q_plus
    # y-grid for S=+1 branch: y = sigma*x + 1
    y_plus = sigma * _GH_NODES + 1.0
    # log f_{Y|+1}(y_plus) = -log(sigma) - 0.5 * x^2
    log_f_plus_at_yplus = -np.log(sigma) - 0.5 * _GH_NODES**2
    # log f_{Y|-1}(y_plus) = -log(sigma) - 0.5 * ((y_plus + 1)/sigma)^2
    log_f_minus_at_yplus = -np.log(sigma) - 0.5 * ((y_plus + 1.0) / sigma) ** 2
    # log f_Y(y_plus) via logsumexp over two components
    a = np.log(q_plus + 1e-300) + log_f_plus_at_yplus
    b = np.log(q_minus + 1e-300) + log_f_minus_at_yplus
    m = np.maximum(a, b)
    log_fY_at_yplus = m + np.log(np.exp(a - m) + np.exp(b - m))
    # KL(f_{Y|+1} || f_Y) = integral of f_{Y|+1}(y) * (log f_{Y|+1} - log f_Y) dy
    # Under x ~ N(0,1), dy = sigma*dx, f_{Y|+1}(y)*dy = phi(x)*dx,
    # so integral = sum_i w_i * (log_f_plus_at_yplus[i] - log_fY_at_yplus[i])
    kl_plus = np.sum(_GH_WEIGHTS * (log_f_plus_at_yplus - log_fY_at_yplus))

    # Symmetric computation for S=-1 branch: y = sigma*x - 1
    y_minus = sigma * _GH_NODES - 1.0
    log_f_minus_at_yminus = -np.log(sigma) - 0.5 * _GH_NODES**2
    log_f_plus_at_yminus = -np.log(sigma) - 0.5 * ((y_minus - 1.0) / sigma) ** 2
    a = np.log(q_plus + 1e-300) + log_f_plus_at_yminus
    b = np.log(q_minus + 1e-300) + log_f_minus_at_yminus
    m = np.maximum(a, b)
    log_fY_at_yminus = m + np.log(np.exp(a - m) + np.exp(b - m))
    kl_minus = np.sum(_GH_WEIGHTS * (log_f_minus_at_yminus - log_fY_at_yminus))

    mi = q_plus * kl_plus + q_minus * kl_minus
    return float(max(mi, 0.0))


def sigma_for_binary_mi(q_plus, beta, tol=1e-6, max_iter=64):
    """
    Find sigma such that I(S; S+N(0,sigma^2)) = beta for binary S with q_+.

    Returns sigma > 0. If q_+ is in {0,1} or beta <= 0, returns None (caller
    should treat as "no noise needed" or "infeasible").

    Uses bisection on sigma in log-space. MI is monotonically decreasing in
    sigma (DPI via Y_sigma_large = Y_sigma_small + independent noise).
    """
    if q_plus <= 0.0 or q_plus >= 1.0:
        return None  # deterministic; no noise needed
    if beta <= 0.0:
        return float("inf")

    # H(S) is an upper bound on I (no noise can help beyond full leakage).
    h_s = -q_plus * np.log(q_plus) - (1 - q_plus) * np.log(1 - q_plus)
    if beta >= h_s:
        # Even sigma=0 gives I = H(S) <= beta. No noise needed.
        return 0.0

    # Bracket sigma: small gives MI ~ H(S), large gives MI ~ 0.
    log_sigma_lo = np.log(1e-4)
    log_sigma_hi = np.log(1e4)
    # Sanity check brackets
    mi_lo = binary_channel_mi(q_plus, np.exp(log_sigma_lo))
    mi_hi = binary_channel_mi(q_plus, np.exp(log_sigma_hi))
    if mi_lo <= beta:
        return float(np.exp(log_sigma_lo))
    if mi_hi >= beta:
        # MI still too high at huge sigma; shouldn't happen but guard
        return float(np.exp(log_sigma_hi))

    for _ in range(max_iter):
        log_mid = 0.5 * (log_sigma_lo + log_sigma_hi)
        mi_mid = binary_channel_mi(q_plus, np.exp(log_mid))
        if abs(mi_mid - beta) < tol:
            break
        if mi_mid > beta:
            log_sigma_lo = log_mid
        else:
            log_sigma_hi = log_mid
    return float(np.exp(0.5 * (log_sigma_lo + log_sigma_hi)))


def update_p_binary(p, signs, noisy_real_release, sigma):
    """
    Posterior update for binary-support PAC mechanism.

    Likelihood: N(noisy_real_release; s_m, sigma^2) for m in 1..M.
    Use the *real-valued* (pre-sign) release for tighter Bayesian update.

    Args:
        p: (M,) current posterior, float64
        signs: (M,) each in {-1, +1}, the quantized m_ghats
        noisy_real_release: float, s_secret + N(0, sigma^2)
        sigma: float, noise std used (sigma=0 -> deterministic branch, no update)

    Returns:
        (M,) updated posterior, float64
    """
    if sigma <= 0.0:
        return p
    diff = signs - noisy_real_release  # (M,)
    log_lik = -0.5 * (diff**2) / (sigma**2)
    log_p = np.log(p + 1e-300) + log_lik
    c = log_p.max()
    return np.exp(log_p - c - np.log(np.sum(np.exp(log_p - c))))


class SubsetAwareDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset to add PAC Privacy membership vectors.

    Two constructions (same marginal MIA prior = 1/2):

    - `disjoint_pairs=False` (default): each sample is in exactly m/2 of m
      subsets, drawn independently by `np.random.choice(m, m/2)`. Subsets
      are random halves without complement structure.

    - `disjoint_pairs=True`: m = 2K subsets arranged as K complementary
      pairs. Draw K independent random halves A_1, ..., A_K. Subsets:
      S_{2k}   = A_k       (sample in if u_i in A_k)
      S_{2k+1} = A_k^c     (sample in if u_i not in A_k)
      Each sample is in exactly K = m/2 subsets, one from each pair.

    Returns original data plus a 'membership_vector' field.

    Works with both:
    - Dict items: {input_ids, labels, option_len, ...} (only_train_option mode)
    - List-of-dict items: [{...}, {...}] (train_as_classification mode)
    """

    def __init__(self, dataset, m=128, disjoint_pairs=False, sampling_rate=None):
        """
        sampling_rate: if given, each sample appears in exactly `sampling_rate`
        of the m subsets (uniformly random). Overrides default m/2. Only
        applies when disjoint_pairs=False. Changes the attack priors:
          MIA prior                 = 1 − sampling_rate/m
          Positive-ID prior         = sampling_rate/m
          Reconstruction prior      = sampling_rate/m
        Default (None) uses m/2 → MIA prior = 1/2 (classical setting).
        """
        self.dataset = dataset
        self.m = m
        self.disjoint_pairs = disjoint_pairs
        self.sampling_rate = sampling_rate
        total_samples = len(dataset)

        assert m % 2 == 0, "m should be even"
        half_m = m // 2
        k = sampling_rate if sampling_rate is not None else half_m
        assert 1 <= k <= m, f"sampling_rate={k} must be in [1, m={m}]"

        self.membership_matrix = torch.zeros((total_samples, m), dtype=torch.bool)

        if disjoint_pairs:
            assert (
                sampling_rate is None
            ), "disjoint_pairs and sampling_rate are mutually exclusive"
            # Build K = m/2 independent random halves, then pair each with its complement.
            K = half_m
            for kk in range(K):
                # Random half for pair kk: roughly N/2 samples.
                half = np.random.rand(total_samples) < 0.5
                self.membership_matrix[half, 2 * kk] = True  # A_kk
                self.membership_matrix[~half, 2 * kk + 1] = True  # A_kk^c
        else:
            for element in range(total_samples):
                sampled = np.random.choice(m, k, replace=False)
                self.membership_matrix[element, sampled] = True

    def __getitem__(self, index):
        item = self.dataset[index]
        membership_vector = self.membership_matrix[index]

        if isinstance(item, dict):
            # only_train_option mode: single dict per sample
            item = dict(item)  # shallow copy to avoid mutating original
            item["membership_vector"] = membership_vector
        elif isinstance(item, list):
            # train_as_classification mode: list of dicts (one per candidate)
            # Add the SAME membership vector to each candidate dict
            item = [dict(d) for d in item]
            for d in item:
                d["membership_vector"] = membership_vector
        else:
            raise TypeError(f"SubsetAwareDataset: unexpected item type {type(item)}")

        return item

    def __len__(self):
        return len(self.dataset)


def compute_subset_ghats(ghat_per_sample, membership_vectors, m):
    """
    Aggregate per-sample ghat scalars into M subset ghats.

    Args:
        ghat_per_sample: np.ndarray shape (batch_size,) float64
        membership_vectors: np.ndarray shape (batch_size, m) bool
        m: int, number of subsets

    Returns:
        np.ndarray shape (m,) float64, mean ghat per subset
    """
    membership_float = membership_vectors.astype(np.float64)  # (batch_size, m)
    counts = membership_float.sum(axis=0)  # (m,)
    counts = np.maximum(counts, 1.0)  # avoid division by zero for empty subsets
    weighted = membership_float.T @ ghat_per_sample  # (m,)
    return weighted / counts
