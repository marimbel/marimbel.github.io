"""
Temporal filters for pose keypoint trajectories.

Every filter here works on ONE scalar signal (e.g. the x coordinate of the
left knee).  To filter a whole skeleton, create one filter instance per
landmark per coordinate.  All causal filters expose  .filter(value, t)  so
they can run frame-by-frame in a real-time app.

Author: Marianna Belmares  (CS663 Project 1)
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# 1. Moving average (causal, window N)
# ----------------------------------------------------------------------------
class MovingAverage:
    def __init__(self, window=5):
        self.window = window
        self.buf = []

    def filter(self, x, t=None):
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        return sum(self.buf) / len(self.buf)


# ----------------------------------------------------------------------------
# 2. Exponential moving average  y_t = a*x_t + (1-a)*y_{t-1}
# ----------------------------------------------------------------------------
class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.y = None

    def filter(self, x, t=None):
        self.y = x if self.y is None else self.alpha * x + (1 - self.alpha) * self.y
        return self.y


# ----------------------------------------------------------------------------
# 3. One Euro filter (Casiez, Roussel & Vogel, CHI 2012)
# ----------------------------------------------------------------------------
def _smoothing_factor(te, cutoff):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / te)


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        te = t - self.t_prev
        if te <= 0:
            return self.x_prev
        # (a) estimate and smooth the speed
        dx = (x - self.x_prev) / te
        a_d = _smoothing_factor(te, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # (b) speed decides the cutoff: slow -> heavy smoothing, fast -> light
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(te, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ----------------------------------------------------------------------------
# 4. Kalman filter, constant-velocity model, state s = [position, velocity]
# ----------------------------------------------------------------------------
class Kalman1D:
    def __init__(self, q=50.0, r=1e-4):
        self.q = q          # process noise (how much acceleration we allow)
        self.r = r          # measurement noise variance (detector jitter)
        self.s = None       # state  [p, v]
        self.P = None       # covariance 2x2
        self.t_prev = None

    def predict(self, t):
        dt = t - self.t_prev
        F = np.array([[1, dt], [0, 1]])
        # white-noise-acceleration process covariance
        Q = self.q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q
        self.t_prev = t

    def update(self, z, r_scale=1.0, gate=None):
        H = np.array([[1.0, 0.0]])
        R = self.r * r_scale
        y = z - (H @ self.s)[0]                  # innovation
        S = (H @ self.P @ H.T)[0, 0] + R         # innovation covariance
        if gate is not None and y * y / S > gate:
            return False                         # outlier: reject measurement
        K = (self.P @ H.T) / S                   # Kalman gain (2x1)
        self.s = self.s + K[:, 0] * y
        self.P = (np.eye(2) - K @ H) @ self.P
        return True

    def filter(self, z, t, confidence=1.0, min_conf=0.5, gate=16.0):
        """z may be None (missing).  Low confidence inflates R."""
        if self.s is None:
            if z is None:
                return None
            self.s = np.array([z, 0.0])
            self.P = np.diag([self.r, 1.0])
            self.t_prev = t
            return z
        self.predict(t)
        if z is not None and confidence >= min_conf:
            # confidence-aware: trust a 0.6-visibility point less than a 0.99 one
            self.update(z, r_scale=1.0 / max(confidence, 1e-3) ** 2, gate=gate)
            # gate=16 -> reject innovations beyond 4 standard deviations
        return self.s[0]


# ----------------------------------------------------------------------------
# Offline (non-causal) filters - need future frames, so not for live feedback
# ----------------------------------------------------------------------------
def gaussian_offline(x, sigma=2.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma, mode="nearest")


def savgol_offline(x, window=11, order=3):
    from scipy.signal import savgol_filter
    return savgol_filter(x, window, order, mode="interp")


# ----------------------------------------------------------------------------
# Gap filling for missing / low-confidence landmarks
# ----------------------------------------------------------------------------
def interpolate_gaps(x, valid, max_gap=10):
    """Linear interpolation over gaps no longer than max_gap frames.
    Longer gaps stay NaN (the landmark is declared lost, not invented)."""
    x = np.asarray(x, float).copy()
    x[~valid] = np.nan
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    out = x.copy()
    i = 0
    while i < len(x):
        if np.isnan(x[i]):
            j = i
            while j < len(x) and np.isnan(x[j]):
                j += 1
            if i > 0 and j < len(x) and (j - i) <= max_gap:
                out[i:j] = np.interp(idx[i:j], [i - 1, j], [x[i - 1], x[j]])
            i = j
        else:
            i += 1
    return out
