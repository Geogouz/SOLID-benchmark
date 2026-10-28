"""The block likelihood of the scene link of Section I, the figure of S-IV, from the committed records.

The link joins agent 85 and anchor 80 in scenario 603054 of the sixth training scan of TWR-CLOUD (scene-link.json):
ten readings that agree within 2 cm and lie about 36 cm above the true distance. This script evaluates its block
likelihood, equations (packet), (shift) and (block) of Section IV-A, at the fitted link parameters of TWR-CLOUD (the
anchor-only profile that Table III prints, the same in every mode), with the core scale of each reading the fitted
kappa times its reported scale, by quadrature over the shifted distance v = d + w on a grid of 0.002 cm. It records the
log block likelihood relative to its peak over a range of distances, with its values at the true distance and at the
distance as far above the median reading as the true distance lies below it. It evaluates the paper's equations
directly, independently of the tables that the method builds (S-I-B checks those against such quadratures), and it
fits nothing.

Usage: scene_curve.py [--scene research/artifacts/solid/figures/data/scene-link.json]
                      [--out research/artifacts/solid/figures/data/scene-curve.json] [--plot DIR]
       --plot writes DIR/scene-link.pdf, the figure of S-IV, in the style of the paper's figures
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "research/artifacts/solid/figures/data/scene-link.json"
PROFILE = ROOT / "research/artifacts/solid/profiles/bern-anchor-only.json"
STEP = 0.002          # cm, quadrature step of v and of w
D_STEP = 0.1          # cm, step of the distances, a whole number of quadrature steps
D_RANGE = (290.0, 460.0)
RECORD_EVERY = 5      # keep every fifth distance, 0.5 cm apart, in the record


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_student(eps, sigma, nu):
    """Log density of the Student t law of a reading error with scale sigma and nu degrees of freedom, eq. (packet)."""
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi) - math.log(sigma)
    return c - (nu + 1) / 2 * np.log1p(eps ** 2 / (nu * sigma ** 2))


def shift_density(w, pi_b, mu, tau):
    """The density of the shift w = u + b, eq. (shift): the Gaussian offset alone or plus an exponential excess."""
    erf = np.vectorize(math.erf)
    phi_norm = np.exp(-0.5 * (w / tau) ** 2) / (math.sqrt(2 * math.pi) * tau)
    cdf = 0.5 * (1 + erf((w / tau - tau / mu) / math.sqrt(2)))
    return (1 - pi_b) * phi_norm + pi_b / mu * np.exp(tau ** 2 / (2 * mu ** 2) - w / mu) * cdf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    scene = json.loads(args.scene.read_text())["scene"]
    p = json.loads(PROFILE.read_text())["anchor_only"]
    y = np.array(scene["readings_cm"])
    s = np.array(scene["scales_cm"])
    nu, kappa = p["nu"], p["kappa"]
    sigma = kappa * s
    sigma_med = statistics.median(sigma)
    tau = p["persistent_sd_rel"] * sigma_med
    mu = p["excess_mean_rel"] * sigma_med
    pi_b = p["excess_probability"]

    # G(v), the product of the reading densities at the shifted distance v, on a grid around the readings
    v = np.arange(round((y.min() - 20) / STEP), round((y.max() + 20) / STEP) + 1) * STEP
    log_g = sum(log_student(yk - v, sk, nu) for yk, sk in zip(y, sigma))
    g_peak = log_g.max()
    g = np.exp(log_g - g_peak)
    # f(w) on one grid that every shift v - d of the distances below uses, so each d reads a contiguous slice
    k = round(D_STEP / STEP)
    d_index = np.arange(round(D_RANGE[0] / STEP), round(D_RANGE[1] / STEP) + 1, k)
    w_lo = int(round(v[0] / STEP)) - d_index[-1]
    w_hi = int(round(v[-1] / STEP)) - d_index[0]
    w = np.arange(w_lo, w_hi + 1) * STEP
    f = shift_density(w, pi_b, mu, tau)
    start = int(round(v[0] / STEP))
    log_l = np.array([math.log(STEP * float(np.dot(g, f[start - di - w_lo:start - di - w_lo + len(g)]))) + g_peak
                      for di in d_index])
    d = d_index * STEP
    peak = int(np.argmax(log_l))
    rel = log_l - log_l[peak]

    median = float(np.median(y))
    true = scene["true_distance_cm"]
    above = 2 * median - true
    at = lambda x: float(np.interp(x, d, rel))
    result = dict(
        sources=dict(scene_link=sha256(args.scene), profile=sha256(PROFILE)),
        link=dict(scan=scene["scan"], scenario=scene["scenario_id"], agent=scene["agent"], anchor=scene["anchor"],
                  readings_cm=scene["readings_cm"], scales_cm=scene["scales_cm"], true_distance_cm=true,
                  median_reading_cm=round(median, 2)),
        parameters=dict(nu=nu, kappa=kappa, excess_probability=pi_b, excess_mean_rel=p["excess_mean_rel"],
                        persistent_sd_rel=p["persistent_sd_rel"], median_core_scale_cm=round(sigma_med, 4),
                        offset_sd_cm=round(tau, 3), excess_mean_cm=round(mu, 3)),
        quadrature=dict(step_cm=STEP, distance_step_cm=D_STEP),
        peak_distance_cm=round(float(d[peak]), 2),
        relative_log_likelihood_nats=dict(at_true_distance=round(at(true), 3),
                                         at_mirrored_distance=round(at(above), 3), mirrored_distance_cm=round(above, 2)),
        curve=dict(distance_cm=[round(float(x), 2) for x in d[::RECORD_EVERY]],
                   relative_log_likelihood_nats=[round(float(x), 4) for x in rel[::RECORD_EVERY]]),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print("wrote", args.out)
    if args.plot:
        import style
        style.setup()
        plt = style.plt
        fig, ax = plt.subplots(figsize=(style.COLUMN, 1.55))
        ax.plot(d, rel, color=style.COLORS["solid"], linewidth=1.2)
        ax.axvline(true, color="0.35", linestyle="--", linewidth=0.7)
        ax.text(true - 2, -7.4, f"true {true:.1f} cm", ha="right", va="bottom", fontsize=7, color="0.25")
        for yk in y:
            ax.plot([yk, yk], [-8.0, -7.55], color="0.2", linewidth=0.5)
        ax.text(y.max() + 2, -7.9, "readings", ha="left", va="bottom", fontsize=7, color="0.25")
        ax.set_xlim(300, 440)
        ax.set_ylim(-8.0, 0.4)
        ax.set_xlabel("Distance $d$ (cm)")
        ax.set_ylabel("Log block likelihood\nless its peak (nats)")
        args.plot.mkdir(parents=True, exist_ok=True)
        # no creation date, so that a rerun gives the same bytes
        fig.savefig(args.plot / "scene-link.pdf", metadata={"CreationDate": None})
        print("wrote", args.plot / "scene-link.pdf")


if __name__ == "__main__":
    main()
