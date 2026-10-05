"""
Experimental comparison of temporal filters on a squat sequence.

Two modes
---------
1) python run_experiment.py --simulate
   Builds a synthetic side-view squat (hip/knee/ankle) with known ground truth,
   then corrupts it with the error types real detectors produce: frame-to-frame
   jitter, slow wobble, outlier spikes, a low-confidence (motion-blur) burst and
   a short full dropout.  Because the truth is known we can measure error.

2) python run_experiment.py --csv landmarks.csv
   Uses landmarks exported by extract_landmarks.py from YOUR OWN video
   (MediaPipe).  No ground truth exists here, so only jitter / lag / rep count
   style metrics are produced.

Outputs: ../img/exp_*.png figures, ../data/squat_sim.json (for the web demo),
         results.json (metrics table).
"""
import argparse, json, math, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from filters import (MovingAverage, EMA, OneEuro, Kalman1D,
                     gaussian_offline, savgol_offline, interpolate_gaps)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "img")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(IMG, exist_ok=True); os.makedirs(DATA, exist_ok=True)

FPS = 30.0
JOINTS = ["hip", "knee", "ankle"]

# ---------------------------------------------------------------- palette
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e1"
COL = {"Raw": "#b5b3ad", "Truth": INK, "One Euro": "#2a78d6", "Kalman": "#eb6834",
       "EMA": "#1baf7a", "Moving avg": "#eda100", "Gaussian (offline)": "#e87ba4",
       "Savitzky-Golay (offline)": "#4a3aa7"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "legend.frameon": False})


# ================================================================ simulation
def knee_angle_profile(t):
    """Ground-truth knee flexion angle (deg) over time.
    0-10 s : 4 slow squats (2.5 s each)
    10-14 s: standing still (tests jitter at rest)
    14-20 s: 6 fast squats (1 s each) (tests lag)"""
    a = np.full_like(t, 175.0)
    slow = t < 10
    a[slow] = 175 - 90 * (1 - np.cos(2 * np.pi * t[slow] / 2.5)) / 2
    fast = t >= 14
    a[fast] = 175 - 80 * (1 - np.cos(2 * np.pi * (t[fast] - 14) / 1.0)) / 2
    return a


def forward_kinematics(theta_deg):
    """Side view, person faces +x, image y grows downward (like MediaPipe)."""
    ankle = np.array([0.55, 0.90])
    L_shank, L_thigh = 0.22, 0.23
    th = np.radians(theta_deg)
    phi = np.radians((180 - theta_deg) * 0.45)          # shank leans forward
    knee = ankle + L_shank * np.array([np.sin(phi), -np.cos(phi)])
    u = (ankle - knee) / L_shank                        # knee -> ankle
    # rotate u by +/- theta, keep the one that puts the hip behind the knee
    def rot(v, a):
        return np.array([v[0] * np.cos(a) - v[1] * np.sin(a),
                         v[0] * np.sin(a) + v[1] * np.cos(a)])
    c1, c2 = rot(u, th), rot(u, -th)
    w = c1 if c1[0] < c2[0] else c2
    hip = knee + L_thigh * w
    return hip, knee, ankle


def simulate(seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 20, 1 / FPS)
    n = len(t)
    theta = knee_angle_profile(t)
    truth = {j: np.zeros((n, 2)) for j in JOINTS}
    for i, th in enumerate(theta):
        h, k, a = forward_kinematics(th)
        truth["hip"][i], truth["knee"][i], truth["ankle"][i] = h, k, a

    raw = {j: truth[j].copy() for j in JOINTS}
    vis = {j: np.full(n, 0.97) for j in JOINTS}
    for j in JOINTS:
        sigma = 0.0045 if j != "ankle" else 0.0035
        white = rng.normal(0, sigma, (n, 2))                 # frame jitter
        wob = np.zeros((n, 2))                               # slow wobble AR(1)
        for i in range(1, n):
            wob[i] = 0.9 * wob[i - 1] + rng.normal(0, 0.0012, 2)
        raw[j] += white + wob
        vis[j] += rng.normal(0, 0.015, n)
        spikes = rng.random(n) < 0.012                        # outlier spikes
        raw[j][spikes] += rng.normal(0, 0.03, (spikes.sum(), 2))
        vis[j][spikes] -= 0.25
    # motion-blur burst on the knee: garbage + low visibility
    blur = slice(int(4.6 * FPS), int(5.0 * FPS))
    raw["knee"][blur] += rng.normal(0, 0.035, (blur.stop - blur.start, 2))
    vis["knee"][blur] = rng.uniform(0.15, 0.35, blur.stop - blur.start)
    # full dropout (detector returns nothing) during a fast rep
    drop = slice(int(16.2 * FPS), int(16.45 * FPS))
    raw["knee"][drop] = np.nan
    vis["knee"][drop] = 0.0
    for j in JOINTS:
        vis[j] = np.clip(vis[j], 0, 1)
    return t, theta, truth, raw, vis


# ================================================================ geometry
def joint_angle(a, b, c):
    """Angle ABC in degrees for arrays of points (n,2)."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# ================================================================ filtering
CONF_MIN = 0.5


def run_causal(make, t, raw, vis, hold_missing=True):
    """Apply a causal per-coordinate filter. Low-confidence / missing frames are
    replaced by the last good sample ("hold") before filtering."""
    out = {}
    cost = 0.0
    for j in JOINTS:
        res = np.zeros_like(raw[j])
        for d in range(2):
            f = make()
            last = None
            for i in range(len(t)):
                z = raw[j][i, d]
                ok = (not np.isnan(z)) and vis[j][i] >= CONF_MIN
                if not ok:
                    z = last if last is not None else raw[j][0, d]
                else:
                    last = z
                t0 = time.perf_counter()
                res[i, d] = f.filter(z, t[i])
                cost += time.perf_counter() - t0
        out[j] = res
    return out, cost / (len(t) * len(JOINTS) * 2) * 1e6   # us per scalar update


def run_kalman(t, raw, vis, q=10.0, r=1e-4):
    out = {}
    cost = 0.0
    for j in JOINTS:
        res = np.zeros_like(raw[j])
        for d in range(2):
            k = Kalman1D(q=q, r=r)
            for i in range(len(t)):
                z = raw[j][i, d]
                z = None if np.isnan(z) else z
                t0 = time.perf_counter()
                res[i, d] = k.filter(z, t[i], confidence=vis[j][i], min_conf=CONF_MIN)
                cost += time.perf_counter() - t0
        out[j] = res
    return out, cost / (len(t) * len(JOINTS) * 2) * 1e6


def run_offline(fn, raw, vis):
    out = {}
    for j in JOINTS:
        res = np.zeros_like(raw[j])
        for d in range(2):
            valid = (~np.isnan(raw[j][:, d])) & (vis[j] >= CONF_MIN)
            filled = interpolate_gaps(raw[j][:, d], valid, max_gap=15)
            filled = np.where(np.isnan(filled), np.nanmean(filled), filled)
            res[:, d] = fn(filled)
        out[j] = res
    return out


# ================================================================ metrics
def lag_ms(sig, ref, max_lag=10):
    """Delay (ms) that best aligns sig to ref (positive = sig is late)."""
    best, best_k = -1e9, 0
    a = ref - ref.mean()
    for k in range(0, max_lag + 1):
        b = sig[k:] - sig[k:].mean()
        c = np.dot(a[:len(b)], b) / len(b)
        if c > best:
            best, best_k = c, k
    return best_k / FPS * 1000


def count_reps(angle, down=115, up=160):
    """Hysteresis rep counter: a rep = go below `down` then back above `up`."""
    reps, state = 0, "up"
    for a in angle:
        if np.isnan(a):
            continue
        if state == "up" and a < down:
            state = "down"
        elif state == "down" and a > up:
            state = "up"; reps += 1
    return reps


def count_reps_naive(angle, thr=130):
    """Single threshold, no hysteresis - what many beginner apps do."""
    reps, below = 0, False
    for a in angle:
        if np.isnan(a):
            continue
        if a < thr and not below:
            below = True
        elif a >= thr and below:
            below = False; reps += 1
    return reps


def metrics(t, ang, truth_ang, tc=None):
    rest = (t > 10.5) & (t < 13.5)
    fast = (t >= 14.5) & (t < 20)
    err = ang - truth_ang
    jitter = np.sqrt(np.nanmean(np.diff(ang[rest], 2) ** 2))
    return {
        "rmse_deg": float(np.sqrt(np.nanmean(err ** 2))),
        "rest_jitter_deg": float(jitter),
        "fast_lag_ms": float(lag_ms(np.nan_to_num(ang[fast], nan=175), truth_ang[fast])),
        "fast_peak_err_deg": float(np.nanmax(np.abs(err[fast]))),
        "reps_hysteresis": count_reps(ang),
        "reps_naive": count_reps_naive(ang),
        "us_per_update": tc,
    }


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()
    if args.csv:
        return run_on_csv(args.csv)

    t, theta, truth, raw, vis = simulate()
    ang_truth = joint_angle(truth["hip"], truth["knee"], truth["ankle"])
    ang_raw = joint_angle(raw["hip"], raw["knee"], raw["ankle"])

    results, angles = {}, {"Truth": ang_truth, "Raw": ang_raw}
    results["Raw"] = metrics(t, ang_raw, ang_truth, 0.0)

    ma, c = run_causal(lambda: MovingAverage(7), t, raw, vis)
    angles["Moving avg"] = joint_angle(ma["hip"], ma["knee"], ma["ankle"])
    results["Moving avg"] = metrics(t, angles["Moving avg"], ang_truth, c)

    em, c = run_causal(lambda: EMA(0.3), t, raw, vis)
    angles["EMA"] = joint_angle(em["hip"], em["knee"], em["ankle"])
    results["EMA"] = metrics(t, angles["EMA"], ang_truth, c)

    oe, c = run_causal(lambda: OneEuro(min_cutoff=1.0, beta=15.0, d_cutoff=1.0), t, raw, vis)
    angles["One Euro"] = joint_angle(oe["hip"], oe["knee"], oe["ankle"])
    results["One Euro"] = metrics(t, angles["One Euro"], ang_truth, c)

    kf, c = run_kalman(t, raw, vis)
    angles["Kalman"] = joint_angle(kf["hip"], kf["knee"], kf["ankle"])
    results["Kalman"] = metrics(t, angles["Kalman"], ang_truth, c)

    ga = run_offline(lambda x: gaussian_offline(x, 2.0), raw, vis)
    angles["Gaussian (offline)"] = joint_angle(ga["hip"], ga["knee"], ga["ankle"])
    results["Gaussian (offline)"] = metrics(t, angles["Gaussian (offline)"], ang_truth, None)

    sg = run_offline(lambda x: savgol_offline(x, 11, 3), raw, vis)
    angles["Savitzky-Golay (offline)"] = joint_angle(sg["hip"], sg["knee"], sg["ankle"])
    results["Savitzky-Golay (offline)"] = metrics(t, angles["Savitzky-Golay (offline)"], ang_truth, None)

    results["_true_reps"] = count_reps(ang_truth)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for k, v in results.items():
        print(k, v)

    # ---------------- data for the browser demo
    demo = {"fps": FPS, "t": np.round(t, 4).tolist(),
            "truth_angle": np.round(ang_truth, 2).tolist()}
    for j in JOINTS:
        demo[j] = {"x": [None if np.isnan(v) else round(float(v), 5) for v in raw[j][:, 0]],
                   "y": [None if np.isnan(v) else round(float(v), 5) for v in raw[j][:, 1]],
                   "vis": np.round(vis[j], 3).tolist()}
        demo[j + "_truth"] = {"x": np.round(truth[j][:, 0], 5).tolist(),
                              "y": np.round(truth[j][:, 1], 5).tolist()}
    with open(os.path.join(DATA, "squat_sim.json"), "w") as f:
        json.dump(demo, f, separators=(",", ":"))

    make_figures(t, angles, results, raw, vis, truth, kf, oe)


# ================================================================ figures
def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points",
                color="#52514e", fontsize=9, va="center")


def make_figures(t, A, R, raw, vis, truth, kf, oe):
    # 1. overview: raw vs truth
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, A["Raw"], color=COL["Raw"], lw=1.2, label="Raw detector output")
    ax.plot(t, A["Truth"], color=INK, lw=2, ls="--", label="Ground truth")
    for x0, x1, txt in [(0, 10, "4 slow squats"), (10, 14, "standing"), (14, 20, "6 fast squats")]:
        ax.text((x0 + x1) / 2, 186, txt, ha="center", color="#52514e", fontsize=9)
    ax.axvspan(4.6, 5.0, color="#e34948", alpha=0.12)
    ax.text(4.8, 70, "motion blur", ha="center", fontsize=8, color="#52514e")
    ax.axvspan(16.2, 16.45, color="#e34948", alpha=0.12)
    ax.text(16.33, 70, "dropout", ha="center", fontsize=8, color="#52514e")
    ax.set_ylim(60, 192); ax.set_xlabel("time (s)"); ax.set_ylabel("knee angle (deg)")
    ax.set_title("Knee angle computed from raw keypoints vs. ground truth", loc="left", fontsize=11)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp_raw_vs_truth.png"), dpi=150); plt.close(fig)

    # 2. rest jitter zoom
    fig, ax = plt.subplots(figsize=(10, 3.4))
    m = (t > 10.3) & (t < 13.7)
    for k in ["Raw", "EMA", "Kalman", "One Euro"]:
        ax.plot(t[m], A[k][m], color=COL[k], lw=2 if k != "Raw" else 1.2, label=k)
    ax.plot(t[m], A["Truth"][m], color=INK, lw=1.5, ls="--", label="Truth")
    ax.set_xlabel("time (s)"); ax.set_ylabel("knee angle (deg)")
    ax.set_title("Standing still: how much does the angle wobble?", loc="left", fontsize=11)
    ax.legend(ncol=5, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp_rest_jitter.png"), dpi=150); plt.close(fig)

    # 3. fast phase lag zoom
    fig, ax = plt.subplots(figsize=(10, 3.6))
    m = (t > 16.8) & (t < 19.2)
    for k in ["Moving avg", "EMA", "Kalman", "One Euro"]:
        ax.plot(t[m], A[k][m], color=COL[k], lw=2, label=k)
    ax.plot(t[m], A["Truth"][m], color=INK, lw=1.8, ls="--", label="Truth")
    ax.set_xlabel("time (s)"); ax.set_ylabel("knee angle (deg)")
    ax.set_title("Fast squats (1 rep/s): heavy smoothing arrives late and misses the bottom",
                 loc="left", fontsize=11)
    ax.legend(ncol=5, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp_fast_lag.png"), dpi=150); plt.close(fig)

    # 4. occlusion / dropout handling on knee y
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (a, b, title) in zip(axs, [(4.2, 5.4, "Motion-blur burst (visibility < 0.35)"),
                                       (15.9, 16.8, "Full dropout (no detection)")]):
        m = (t > a) & (t < b)
        ax.plot(t[m], raw["knee"][m, 1], "o", ms=3.5, color=COL["Raw"], label="Raw knee y")
        lowc = m & (vis["knee"] < CONF_MIN)
        ax.plot(t[lowc], raw["knee"][lowc, 1], "x", ms=6, color="#e34948", label="rejected (low conf.)")
        ax.plot(t[m], truth["knee"][m, 1], color=INK, ls="--", lw=1.5, label="Truth")
        ax.plot(t[m], oe["knee"][m, 1], color=COL["One Euro"], lw=2, label="One Euro + hold")
        ax.plot(t[m], kf["knee"][m, 1], color=COL["Kalman"], lw=2, label="Kalman predict")
        ax.set_title(title, loc="left", fontsize=10); ax.set_xlabel("time (s)")
        ax.invert_yaxis()
    axs[0].set_ylabel("knee y (normalized, down = +)")
    axs[1].legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp_occlusion.png"), dpi=150); plt.close(fig)

    # 5. metric bars (two separate charts: error and lag - never dual axis)
    names = ["Raw", "Moving avg", "EMA", "One Euro", "Kalman", "Gaussian (offline)", "Savitzky-Golay (offline)"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, key, title in zip(axs, ["rest_jitter_deg", "rmse_deg", "fast_lag_ms"],
                              ["Jitter at rest (deg/frame², lower=better)",
                               "Angle RMSE (deg, lower=better)",
                               "Lag on fast reps (ms, lower=better)"]):
        vals = [R[n][key] for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, color=[COL[n] for n in names], height=0.62)
        ax.set_yticks(y); ax.set_yticklabels(names if ax is axs[0] else [""] * len(names))
        ax.invert_yaxis(); ax.set_title(title, loc="left", fontsize=9.5); ax.grid(axis="y", visible=False)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.1f}" if key != "fast_lag_ms" else f" {v:.0f}", va="center", fontsize=8, color="#52514e")
        ax.set_xlim(0, max(vals) * 1.25 + 1e-6)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp_metrics.png"), dpi=150); plt.close(fig)

    # 6. One Euro cutoff vs speed curve
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    speed = np.linspace(0, 0.6, 200)
    for beta, col in [(0.0, "#b5b3ad"), (1.0, "#1baf7a"), (15.0, "#2a78d6"), (30.0, "#eb6834")]:
        fc = 1.0 + beta * speed
        ax.plot(speed, fc, color=col, lw=2)
        ax.annotate(f"β = {beta:g}", (speed[-1], fc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color="#52514e", va="center")
    ax.set_xlabel("smoothed keypoint speed |dx̂/dt| (image widths / s)")
    ax.set_ylabel("cutoff frequency f_c (Hz)")
    ax.set_title("One Euro: cutoff rises with speed (min_cutoff = 1.0 Hz)", loc="left", fontsize=10.5)
    ax.set_xlim(0, 0.72)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "oneeuro_cutoff.png"), dpi=150); plt.close(fig)

    # 7. moving-average frequency response / lag illustration: step response
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    n = 45; tt = np.arange(n) / FPS * 1000
    step = np.r_[np.zeros(10), np.ones(n - 10)]
    for name, mk, col in [("Moving avg (N=7)", lambda: MovingAverage(7), COL["Moving avg"]),
                          ("EMA (α=0.3)", lambda: EMA(0.3), COL["EMA"]),
                          ("One Euro", lambda: OneEuro(1.0, 15.0), COL["One Euro"])]:
        f = mk(); y = [f.filter(v, i / FPS) for i, v in enumerate(step)]
        ax.plot(tt, y, color=col, lw=2, label=name)
    ax.plot(tt, step, color=INK, ls="--", lw=1.5, label="input step")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("output")
    ax.set_title("Step response: how long until the filter 'catches up'?", loc="left", fontsize=10.5)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "step_response.png"), dpi=150); plt.close(fig)


# ================================================================ real video
def run_on_csv(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows])
    raw, vis = {}, {}
    side = rows[0].get("side", "left")
    for j in JOINTS:
        raw[j] = np.array([[float(r[f"{j}_x"]) if r[f"{j}_x"] else np.nan,
                            float(r[f"{j}_y"]) if r[f"{j}_y"] else np.nan] for r in rows])
        vis[j] = np.array([float(r[f"{j}_vis"]) if r[f"{j}_vis"] else 0.0 for r in rows])
    ang = {"Raw": joint_angle(raw["hip"], raw["knee"], raw["ankle"])}
    for name, mk in [("EMA", lambda: EMA(0.3)), ("One Euro", lambda: OneEuro(1.0, 15.0))]:
        o, _ = run_causal(mk, t, raw, vis)
        ang[name] = joint_angle(o["hip"], o["knee"], o["ankle"])
    kf, _ = run_kalman(t, raw, vis)
    ang["Kalman"] = joint_angle(kf["hip"], kf["knee"], kf["ankle"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for k, v in ang.items():
        ax.plot(t, v, color=COL[k], lw=1.2 if k == "Raw" else 2, label=k)
    ax.set_xlabel("time (s)"); ax.set_ylabel("knee angle (deg)"); ax.legend(ncol=4)
    ax.set_title(f"My video ({side} side): raw vs filtered knee angle", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp_real_video.png"), dpi=150)
    for k, v in ang.items():
        d2 = np.diff(v, 2)
        print(f"{k:10s} jitter={np.sqrt(np.nanmean(d2**2)):.2f}  reps(hyst)={count_reps(v)}  reps(naive)={count_reps_naive(v)}")


if __name__ == "__main__":
    main()
