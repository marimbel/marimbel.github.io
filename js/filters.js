/* Temporal filters for keypoint trajectories (JavaScript versions of
   experiment/filters.py). Each causal filter has .filter(value, tSeconds). */
(function (g) {
  class MovingAverage {
    constructor(window = 5) { this.w = window; this.buf = []; }
    filter(x) {
      this.buf.push(x); if (this.buf.length > this.w) this.buf.shift();
      return this.buf.reduce((a, b) => a + b, 0) / this.buf.length;
    }
  }

  class EMA {
    constructor(alpha = 0.3) { this.a = alpha; this.y = null; }
    filter(x) { this.y = this.y === null ? x : this.a * x + (1 - this.a) * this.y; return this.y; }
  }

  const alphaFor = (te, cutoff) => { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / te); };

  class OneEuro {
    constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
      this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
      this.xPrev = null; this.dxPrev = 0; this.tPrev = null;
    }
    filter(x, t) {
      if (this.xPrev === null) { this.xPrev = x; this.tPrev = t; return x; }
      const te = t - this.tPrev; if (te <= 0) return this.xPrev;
      const dx = (x - this.xPrev) / te;
      const ad = alphaFor(te, this.dCutoff);
      const dxHat = ad * dx + (1 - ad) * this.dxPrev;
      const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
      const a = alphaFor(te, cutoff);
      const xHat = a * x + (1 - a) * this.xPrev;
      this.xPrev = xHat; this.dxPrev = dxHat; this.tPrev = t;
      return xHat;
    }
  }

  /* Constant-velocity Kalman filter, state [p, v], 2x2 covariance written out
     by hand so it runs fast without a matrix library. */
  class Kalman1D {
    constructor(q = 10, r = 1e-4, gate = 16) { this.q = q; this.r = r; this.gate = gate; this.p = null; }
    filter(z, t, conf = 1, minConf = 0.5) {
      if (this.p === null) {
        if (z === null || z === undefined) return null;
        this.p = z; this.v = 0; this.P = [this.r, 0, 0, 1]; this.t = t; return z;
      }
      // ---- predict
      const dt = t - this.t; this.t = t;
      const [P00, P01, P10, P11] = this.P, q = this.q;
      this.p += this.v * dt;
      let n00 = P00 + dt * (P10 + P01) + dt * dt * P11 + q * dt ** 4 / 4;
      let n01 = P01 + dt * P11 + q * dt ** 3 / 2;
      let n10 = P10 + dt * P11 + q * dt ** 3 / 2;
      let n11 = P11 + q * dt * dt;
      // ---- update (skip if missing, low confidence, or an outlier)
      if (z !== null && z !== undefined && conf >= minConf) {
        const R = this.r / Math.max(conf, 1e-3) ** 2;
        const y = z - this.p, S = n00 + R;
        if (y * y / S <= this.gate) {
          const k0 = n00 / S, k1 = n10 / S;
          this.p += k0 * y; this.v += k1 * y;
          const m00 = (1 - k0) * n00, m01 = (1 - k0) * n01;
          const m10 = n10 - k1 * n00, m11 = n11 - k1 * n01;
          n00 = m00; n01 = m01; n10 = m10; n11 = m11;
        }
      }
      this.P = [n00, n01, n10, n11];
      return this.p;
    }
  }

  function jointAngle(a, b, c) {
    const v1x = a[0] - b[0], v1y = a[1] - b[1], v2x = c[0] - b[0], v2y = c[1] - b[1];
    const cos = (v1x * v2x + v1y * v2y) / (Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y));
    return Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
  }

  g.PoseFilters = { MovingAverage, EMA, OneEuro, Kalman1D, jointAngle };
})(window);
