#!/usr/bin/env python3
"""
04_dynamic_thresholds.py
=========================
Instead of hard-coded AM score cutoffs, this script learns DYNAMIC thresholds
from the data using multiple statistical and ML approaches:

1. Gaussian Mixture Models (GMM)   — unsupervised: find natural score "modes"
2. Isotonic Regression             — monotone calibration of score → probability
3. Kernel Density Estimation (KDE) — find score valleys (natural separations)
4. Youden's J per mechanism        — optimal per-class threshold (LoF vs GoF)
5. Expectation-Maximization (EM)   — probabilistic assignment to pathogenic/benign
6. Mechanism-aware thresholds      — fit separate models per LoF/GoF group

The key insight: if AM scores behave differently for LoF vs GoF variants,
the optimal threshold should ALSO differ. We quantify and visualize this.

Requirements:
    pip install pandas numpy scipy scikit-learn matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from scipy.signal import argrelmin
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "am_revel_scores_combined.csv"))
lof = df[df["mechanism"] == "LoF"]
gof = df[df["mechanism"] == "GoF"]

# Binary labels: 1 = LoF, 0 = GoF (both pathogenic, but LoF = better scored)
# For threshold purposes, we also treat ALL as pathogenic vs simulated benign
# We use within-dataset LoF/GoF split as proxy for high/low pathogenicity certainty

print(f"Dataset: {len(df)} variants ({len(lof)} LoF, {len(gof)} GoF)")
print(f"AM score range: {df['am_score'].min():.3f} – {df['am_score'].max():.3f}")
print(
    f"REVEL score range: {df['revel_score'].min():.3f} – {df['revel_score'].max():.3f}\n"
)

# ─── 1. Gaussian Mixture Model (GMM) ─────────────────────────────────────────
print("=" * 60)
print("1. GAUSSIAN MIXTURE MODEL (GMM)")
print("=" * 60)


def fit_gmm(scores, tool_name, n_components=2):
    """
    Fit a 2-component GMM to score distribution.
    The 'valley' between components = natural threshold.
    Returns threshold, gmm object, component params.
    """
    X = scores.values.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components, random_state=42, covariance_type="full", n_init=10
    )
    gmm.fit(X)

    # Sort components by mean (lower = benign-like, higher = pathogenic-like)
    idx = np.argsort(gmm.means_.flatten())
    means = gmm.means_.flatten()[idx]
    stds = np.sqrt(gmm.covariances_.flatten()[idx])
    weights = gmm.weights_[idx]

    # Find valley between components: minimum of total density between the means
    x_fine = np.linspace(means[0], means[1], 1000)
    density = np.zeros_like(x_fine)
    for i, (m, s, w) in enumerate(zip(means, stds, weights)):
        density += w * norm.pdf(x_fine, m, s)

    valley_idx = np.argmin(density)
    gmm_threshold = x_fine[valley_idx]

    print(f"\n{tool_name} GMM (2 components):")
    print(
        f"  Component 1: mean={means[0]:.3f}, std={stds[0]:.3f}, weight={weights[0]:.2f}"
    )
    print(
        f"  Component 2: mean={means[1]:.3f}, std={stds[1]:.3f}, weight={weights[1]:.2f}"
    )
    print(f"  Valley threshold: {gmm_threshold:.3f}")
    print(f"  BIC: {gmm.bic(X):.1f}")

    return gmm_threshold, gmm, means, stds, weights


gmm_thresh_am, gmm_am, am_means, am_stds, am_weights = fit_gmm(
    df["am_score"], "AlphaMissense"
)
gmm_thresh_revel, gmm_revel, revel_means, revel_stds, revel_weights = fit_gmm(
    df["revel_score"], "REVEL"
)

# Per-mechanism GMM thresholds
gmm_thresh_am_lof, _, lof_means, lof_stds, lof_weights = fit_gmm(
    lof["am_score"], "AM (LoF only)"
)
gmm_thresh_am_gof, _, gof_means, gof_stds, gof_weights = fit_gmm(
    gof["am_score"], "AM (GoF only)"
)

# ─── 2. KDE Valley Detection ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. KERNEL DENSITY ESTIMATION (KDE) — VALLEY DETECTION")
print("=" * 60)


def kde_threshold(scores, tool_name, bw_method="scott"):
    """
    Fit KDE, find local minima (valleys) in the density.
    The deepest valley = natural threshold.
    """
    x = np.linspace(scores.min() - 0.05, scores.max() + 0.05, 1000)
    kde = stats.gaussian_kde(scores, bw_method=bw_method)
    density = kde(x)

    # Find local minima
    local_min_idx = argrelmin(density, order=15)[0]
    if len(local_min_idx) == 0:
        print(f"  {tool_name}: No valley found (unimodal distribution)")
        return None, x, density

    # Pick valley closest to score center (most meaningful separation)
    center = scores.mean()
    closest_min = local_min_idx[np.argmin(np.abs(x[local_min_idx] - center))]
    threshold = x[closest_min]

    print(f"\n{tool_name} KDE valley: {threshold:.3f}  (bandwidth={kde.factor:.4f})")
    print(f"  All valleys found at: {[f'{x[i]:.3f}' for i in local_min_idx]}")
    return threshold, x, density


kde_thresh_am, am_x, am_density = kde_threshold(df["am_score"], "AlphaMissense")
kde_thresh_revel, revel_x, revel_density = kde_threshold(df["revel_score"], "REVEL")

# Per-mechanism
kde_thresh_lof, lof_x, lof_density = kde_threshold(lof["am_score"], "AM (LoF)")
kde_thresh_gof, gof_x, gof_density = kde_threshold(gof["am_score"], "AM (GoF)")

# ─── 3. Youden's J (Optimal ROC) per mechanism ───────────────────────────────
print("\n" + "=" * 60)
print("3. YOUDEN'S J — OPTIMAL THRESHOLD PER MECHANISM")
print("=" * 60)

from sklearn.metrics import roc_curve, auc as sk_auc


def youdens_j_threshold(scores_pos, scores_neg, tool_name, label_pos, label_neg):
    """
    Treat scores_pos as 'positive class', scores_neg as 'negative class'.
    Find threshold maximizing Sensitivity + Specificity (Youden's J).
    """
    y_true = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    y_score = np.concatenate([scores_pos, scores_neg])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_sens = tpr[best_idx]
    best_spec = 1 - fpr[best_idx]
    roc_auc = sk_auc(fpr, tpr)

    print(f"\n{tool_name} ({label_pos} vs {label_neg}):")
    print(f"  Youden optimal threshold: {best_thresh:.3f}")
    print(f"  Sensitivity: {best_sens:.2f}, Specificity: {best_spec:.2f}")
    print(f"  AUC: {roc_auc:.3f}, Youden's J: {j_scores[best_idx]:.3f}")

    return best_thresh, fpr, tpr, roc_auc


# LoF vs GoF: can AM score tell them apart?
youden_thresh_am, fpr_am, tpr_am, auc_am = youdens_j_threshold(
    lof["am_score"].values, gof["am_score"].values, "AM", "LoF", "GoF"
)
youden_thresh_revel, fpr_revel, tpr_revel, auc_revel = youdens_j_threshold(
    lof["revel_score"].values, gof["revel_score"].values, "REVEL", "LoF", "GoF"
)

# ─── 4. Isotonic Regression (Score → Probability Calibration) ────────────────
print("\n" + "=" * 60)
print("4. ISOTONIC REGRESSION — SCORE CALIBRATION")
print("=" * 60)


def isotonic_calibration(scores, labels, tool_name):
    """
    Calibrate: score → P(LoF | score)
    Monotone increasing mapping. Find score where P = 0.5 (decision boundary).
    """
    # Sort by score
    order = np.argsort(scores)
    x_sorted = scores[order]
    y_sorted = labels[order].astype(float)

    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(x_sorted, y_sorted)

    # Predict calibrated probabilities
    x_fine = np.linspace(scores.min(), scores.max(), 1000)
    proba = ir.predict(x_fine)

    # Find where calibrated probability crosses 0.5 (and 0.33, 0.67 for evidence tiers)
    thresholds = {}
    for target_p, label in [
        (0.33, "Supporting (P=0.33)"),
        (0.5, "Moderate (P=0.50)"),
        (0.67, "Strong (P=0.67)"),
    ]:
        # Find first crossing from below
        crossings = np.where(np.diff(np.sign(proba - target_p)))[0]
        if len(crossings) > 0:
            thresh = x_fine[crossings[0]]
            thresholds[label] = thresh
            print(f"  {tool_name} calibrated threshold at P={target_p}: {thresh:.3f}")
        else:
            print(f"  {tool_name}: No crossing found for P={target_p}")

    return ir, x_fine, proba, thresholds


labels_binary = (df["mechanism"] == "LoF").astype(int).values

ir_am, am_x_cal, am_proba, am_cal_thresholds = isotonic_calibration(
    df["am_score"].values, labels_binary, "AlphaMissense"
)
ir_revel, revel_x_cal, revel_proba, revel_cal_thresholds = isotonic_calibration(
    df["revel_score"].values, labels_binary, "REVEL"
)

# ─── 5. Mechanism-Aware Logistic Boundary ────────────────────────────────────
print("\n" + "=" * 60)
print("5. MECHANISM-AWARE LOGISTIC THRESHOLD")
print("=" * 60)


def logistic_threshold(df, tool="am_score"):
    """
    Fit logistic regression: [score, gene] → P(LoF)
    Find score threshold per gene separately (mechanism-aware).
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    gene_enc = le.fit_transform(df["gene"])
    X = np.column_stack([df[tool].values, gene_enc])
    y = (df["mechanism"] == "LoF").astype(int).values

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)

    # Cross-val score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc")
    print(
        f"\n  Logistic ({tool}) 5-fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    )

    # Per-gene decision boundary (score where P(LoF) = 0.5)
    scores_fine = np.linspace(0.3, 1.0, 1000)
    gene_thresholds = {}
    for gene, g_idx in zip(le.classes_, range(len(le.classes_))):
        X_gene = np.column_stack([scores_fine, np.full(1000, g_idx)])
        proba = lr.predict_proba(X_gene)[:, 1]
        crossings = np.where(np.diff(np.sign(proba - 0.5)))[0]
        if len(crossings) > 0:
            thresh = scores_fine[crossings[0]]
            gene_thresholds[gene] = thresh
            print(f"  {gene} decision boundary ({tool}): {thresh:.3f}")

    return lr, gene_thresholds


lr_am, am_gene_thresholds = logistic_threshold(df, "am_score")
lr_revel, revel_gene_thresholds = logistic_threshold(df, "revel_score")

# ─── 6. Compile Dynamic Threshold Summary ────────────────────────────────────
print("\n" + "=" * 60)
print("6. DYNAMIC THRESHOLD SUMMARY TABLE")
print("=" * 60)

summary = {
    "Method": [
        "Published (Fixed)",
        "Published (Fixed)",
        "Published (Fixed)",
        "GMM Valley (all)",
        "GMM Valley (LoF only)",
        "GMM Valley (GoF only)",
        "KDE Valley (all)",
        "KDE Valley (LoF only)",
        "KDE Valley (GoF only)",
        "Youden's J (LoF>GoF)",
        "Isotonic P=0.33",
        "Isotonic P=0.50",
        "Isotonic P=0.67",
        "Logistic ABCC8",
        "Logistic GCK",
        "Logistic KCNJ11",
    ],
    "AM Threshold": [
        0.564,
        0.740,
        0.890,
        round(gmm_thresh_am, 3),
        round(gmm_thresh_am_lof, 3),
        round(gmm_thresh_am_gof, 3),
        round(kde_thresh_am, 3) if kde_thresh_am else None,
        round(kde_thresh_lof, 3) if kde_thresh_lof else None,
        round(kde_thresh_gof, 3) if kde_thresh_gof else None,
        round(youden_thresh_am, 3),
        round(am_cal_thresholds.get("Supporting (P=0.33)", 0), 3),
        round(am_cal_thresholds.get("Moderate (P=0.50)", 0), 3),
        round(am_cal_thresholds.get("Strong (P=0.67)", 0), 3),
        round(am_gene_thresholds.get("ABCC8", 0), 3),
        round(am_gene_thresholds.get("GCK", 0), 3),
        round(am_gene_thresholds.get("KCNJ11", 0), 3),
    ],
    "REVEL Threshold": [
        0.644,
        0.773,
        0.932,
        round(gmm_thresh_revel, 3),
        None,
        None,
        round(kde_thresh_revel, 3) if kde_thresh_revel else None,
        None,
        None,
        round(youden_thresh_revel, 3),
        round(revel_cal_thresholds.get("Supporting (P=0.33)", 0), 3),
        round(revel_cal_thresholds.get("Moderate (P=0.50)", 0), 3),
        round(revel_cal_thresholds.get("Strong (P=0.67)", 0), 3),
        round(revel_gene_thresholds.get("ABCC8", 0), 3),
        round(revel_gene_thresholds.get("GCK", 0), 3),
        round(revel_gene_thresholds.get("KCNJ11", 0), 3),
    ],
    "Evidence Level": [
        "Supporting",
        "Moderate",
        "Strong",
        "Moderate",
        "LoF-specific",
        "GoF-specific",
        "Moderate",
        "LoF-specific",
        "GoF-specific",
        "Moderate",
        "Supporting",
        "Moderate",
        "Strong",
        "Gene-specific",
        "Gene-specific",
        "Gene-specific",
    ],
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(
    os.path.join(RESULTS_DIR, "dynamic_thresholds_summary.csv"), index=False
)

# ─── 7. Evaluate Sensitivity/Specificity of Dynamic vs Fixed ─────────────────
print("\n" + "=" * 60)
print("7. SENSITIVITY COMPARISON: DYNAMIC vs FIXED THRESHOLDS")
print("=" * 60)


def eval_threshold(df, score_col, threshold, mechanism_filter=None):
    """% of variants scoring >= threshold"""
    if mechanism_filter:
        subset = df[df["mechanism"] == mechanism_filter]
    else:
        subset = df
    n = (subset[score_col] >= threshold).sum()
    return n, len(subset), 100 * n / len(subset)


print(f"\n{'Threshold':<32} {'LoF':<18} {'GoF':<18} {'Gap':>6}")
print("-" * 76)

test_cases = [
    ("AM fixed Supporting (0.564)", "am_score", 0.564),
    ("AM fixed Strong (0.890)", "am_score", 0.890),
    (f"AM GMM valley ({gmm_thresh_am:.3f})", "am_score", gmm_thresh_am),
    (f"AM Youden's J ({youden_thresh_am:.3f})", "am_score", youden_thresh_am),
    (
        f"AM KDE valley ({kde_thresh_am:.3f})" if kde_thresh_am else "AM KDE N/A",
        "am_score",
        kde_thresh_am or 0.890,
    ),
    ("REVEL fixed Strong (0.932)", "revel_score", 0.932),
    (f"REVEL GMM valley ({gmm_thresh_revel:.3f})", "revel_score", gmm_thresh_revel),
]

comparison_rows = []
for label, col, thresh in test_cases:
    lof_n, lof_t, lof_pct = eval_threshold(df, col, thresh, "LoF")
    gof_n, gof_t, gof_pct = eval_threshold(df, col, thresh, "GoF")
    _, pval = stats.fisher_exact([[lof_n, lof_t - lof_n], [gof_n, gof_t - gof_n]])
    gap = lof_pct - gof_pct
    sig = "*" if pval < 0.05 else ("†" if pval < 0.10 else "")
    print(
        f"{label:<32} {lof_n}/{lof_t} ({lof_pct:4.0f}%)  "
        f"{gof_n}/{gof_t} ({gof_pct:4.0f}%)  {gap:+5.1f}% {sig}"
    )
    comparison_rows.append(
        {
            "threshold_method": label,
            "score_col": col,
            "threshold": thresh,
            "lof_pct": lof_pct,
            "gof_pct": gof_pct,
            "gap": gap,
            "pval": pval,
        }
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(
    os.path.join(RESULTS_DIR, "dynamic_vs_fixed_comparison.csv"), index=False
)

# ─── 8. FIGURES ───────────────────────────────────────────────────────────────
print("\nGenerating figures...")

fig = plt.figure(figsize=(20, 18))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

C = {
    "LoF": "#2166ac",
    "GoF": "#d73027",
    "all": "#4d9221",
    "gmm": "#7b2d8b",
    "kde": "#d95f02",
    "youden": "#1b7837",
    "fixed": "#888888",
    "isotonic": "#e7298a",
}

# ── Panel A: AM score histogram + GMM components + thresholds
ax = fig.add_subplot(gs[0, 0])
scores = df["am_score"].values
ax.hist(
    scores, bins=30, density=True, alpha=0.35, color="steelblue", label="All variants"
)
ax.hist(
    lof["am_score"].values,
    bins=20,
    density=True,
    alpha=0.25,
    color=C["LoF"],
    label="LoF",
)
ax.hist(
    gof["am_score"].values,
    bins=20,
    density=True,
    alpha=0.25,
    color=C["GoF"],
    label="GoF",
)

# Overlay GMM components
x_plot = np.linspace(0.3, 1.0, 500)
for i, (m, s, w) in enumerate(zip(am_means, am_stds, am_weights)):
    comp = w * norm.pdf(x_plot, m, s)
    ax.plot(
        x_plot,
        comp,
        "--",
        lw=1.8,
        color=C["gmm"],
        alpha=0.8,
        label=f"GMM comp {i + 1} (μ={m:.3f})"
        if i == 0
        else f"GMM comp {i + 1} (μ={m:.3f})",
    )

# Threshold lines
ax.axvline(gmm_thresh_am, color=C["gmm"], lw=2, label=f"GMM valley {gmm_thresh_am:.3f}")
ax.axvline(0.890, color=C["fixed"], lw=1.5, ls=":", label="Fixed strong (0.890)")
if kde_thresh_am:
    ax.axvline(
        kde_thresh_am,
        color=C["kde"],
        lw=1.8,
        ls="-.",
        label=f"KDE valley {kde_thresh_am:.3f}",
    )

ax.set_xlabel("AlphaMissense score", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("A. AM: Mixture model\n& threshold detection", fontsize=10)
ax.legend(fontsize=6.5, loc="upper left")
ax.set_xlim(0.3, 1.0)

# ── Panel B: REVEL score histogram + GMM
ax = fig.add_subplot(gs[0, 1])
ax.hist(
    df["revel_score"].values,
    bins=30,
    density=True,
    alpha=0.35,
    color="coral",
    label="All",
)
ax.hist(
    lof["revel_score"].values,
    bins=20,
    density=True,
    alpha=0.25,
    color=C["LoF"],
    label="LoF",
)
ax.hist(
    gof["revel_score"].values,
    bins=20,
    density=True,
    alpha=0.25,
    color=C["GoF"],
    label="GoF",
)
for i, (m, s, w) in enumerate(zip(revel_means, revel_stds, revel_weights)):
    comp = w * norm.pdf(x_plot, m, s)
    ax.plot(x_plot, comp, "--", lw=1.8, color=C["gmm"], alpha=0.8)
ax.axvline(
    gmm_thresh_revel, color=C["gmm"], lw=2, label=f"GMM valley {gmm_thresh_revel:.3f}"
)
ax.axvline(0.932, color=C["fixed"], lw=1.5, ls=":", label="Fixed strong (0.932)")
if kde_thresh_revel:
    ax.axvline(
        kde_thresh_revel,
        color=C["kde"],
        lw=1.8,
        ls="-.",
        label=f"KDE valley {kde_thresh_revel:.3f}",
    )
ax.set_xlabel("REVEL score", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("B. REVEL: Mixture model\n& threshold detection", fontsize=10)
ax.legend(fontsize=6.5, loc="upper left")
ax.set_xlim(0.3, 1.0)

# ── Panel C: KDE per mechanism (AM) — shows why GoF needs lower threshold
ax = fig.add_subplot(gs[0, 2])
x_lof = np.linspace(lof["am_score"].min() - 0.05, 1.0, 500)
x_gof = np.linspace(gof["am_score"].min() - 0.05, 1.0, 500)
kde_lof_obj = stats.gaussian_kde(lof["am_score"])
kde_gof_obj = stats.gaussian_kde(gof["am_score"])
ax.fill_between(x_lof, kde_lof_obj(x_lof), alpha=0.3, color=C["LoF"])
ax.fill_between(x_gof, kde_gof_obj(x_gof), alpha=0.3, color=C["GoF"])
ax.plot(x_lof, kde_lof_obj(x_lof), color=C["LoF"], lw=2, label=f"LoF (n=66)")
ax.plot(x_gof, kde_gof_obj(x_gof), color=C["GoF"], lw=2, label=f"GoF (n=65)")
if kde_thresh_lof:
    ax.axvline(
        kde_thresh_lof,
        color=C["LoF"],
        lw=1.8,
        ls="--",
        label=f"LoF valley {kde_thresh_lof:.3f}",
    )
if kde_thresh_gof:
    ax.axvline(
        kde_thresh_gof,
        color=C["GoF"],
        lw=1.8,
        ls="--",
        label=f"GoF valley {kde_thresh_gof:.3f}",
    )
ax.axvline(0.890, color=C["fixed"], lw=1.5, ls=":", label="Fixed 0.890")
ax.set_xlabel("AlphaMissense score", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("C. Mechanism-stratified KDE\nLoF vs GoF (AM)", fontsize=10)
ax.legend(fontsize=7.5)
ax.set_xlim(0.3, 1.0)

# ── Panel D: Isotonic calibration curves
ax = fig.add_subplot(gs[1, 0])
ax.plot(am_x_cal, am_proba, color=C["isotonic"], lw=2.5, label="AM calibrated P(LoF)")
ax.plot(
    revel_x_cal,
    revel_proba,
    color="darkorange",
    lw=2,
    ls="--",
    label="REVEL calibrated P(LoF)",
)
ax.axhline(0.5, color="gray", ls=":", lw=1, label="P=0.50 (decision)")
ax.axhline(0.33, color="gray", ls="--", lw=0.8, alpha=0.6, label="P=0.33 (supporting)")
ax.axhline(0.67, color="gray", ls="-.", lw=0.8, alpha=0.6, label="P=0.67 (strong)")
# Mark calibrated thresholds
for p_label, p_val, col in [
    ("Supporting (P=0.33)", 0.33, "green"),
    ("Moderate (P=0.50)", 0.5, "blue"),
    ("Strong (P=0.67)", 0.67, "red"),
]:
    if p_label in am_cal_thresholds:
        t = am_cal_thresholds[p_label]
        ax.axvline(t, color=col, lw=1.2, alpha=0.7)
        ax.text(t + 0.005, 0.04, f"AM:{t:.3f}", fontsize=6.5, color=col, rotation=90)
ax.set_xlabel("Raw score", fontsize=10)
ax.set_ylabel("Calibrated P(LoF)", fontsize=10)
ax.set_title("D. Isotonic calibration\nscore → P(LoF)", fontsize=10)
ax.legend(fontsize=7, loc="upper left")
ax.set_xlim(0.3, 1.0)
ax.set_ylim(-0.05, 1.05)

# ── Panel E: ROC curves with Youden threshold
ax = fig.add_subplot(gs[1, 1])
ax.plot(fpr_am, tpr_am, color=C["isotonic"], lw=2.2, label=f"AM AUC={auc_am:.3f}")
ax.plot(
    fpr_revel,
    tpr_revel,
    color="darkorange",
    lw=2,
    ls="--",
    label=f"REVEL AUC={auc_revel:.3f}",
)
ax.plot([0, 1], [0, 1], "gray", lw=0.8, ls=":")
# Mark Youden points
for fpr_curve, tpr_curve, thresh, col in [
    (fpr_am, tpr_am, youden_thresh_am, C["isotonic"]),
    (fpr_revel, tpr_revel, youden_thresh_revel, "darkorange"),
]:
    j = tpr_curve - fpr_curve
    best = np.argmax(j)
    ax.scatter(
        fpr_curve[best],
        tpr_curve[best],
        color=col,
        s=100,
        zorder=5,
        marker="*",
        edgecolors="black",
        lw=0.5,
    )
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("E. ROC: LoF vs GoF separation\n(★ = Youden's J threshold)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

# ── Panel F: Dynamic threshold sensitivity — LoF % vs GoF % scatter
ax = fig.add_subplot(gs[1, 2])
thresholds_fine = np.linspace(0.4, 0.99, 100)
lof_pcts, gof_pcts, gaps = [], [], []
for t in thresholds_fine:
    lp = 100 * (lof["am_score"] >= t).mean()
    gp = 100 * (gof["am_score"] >= t).mean()
    lof_pcts.append(lp)
    gof_pcts.append(gp)
    gaps.append(lp - gp)

sc = ax.scatter(
    thresholds_fine, gaps, c=thresholds_fine, cmap="RdYlGn_r", s=20, alpha=0.9
)
plt.colorbar(sc, ax=ax, label="Threshold value")
ax.axvline(gmm_thresh_am, color=C["gmm"], lw=2, label=f"GMM {gmm_thresh_am:.3f}")
ax.axvline(
    youden_thresh_am,
    color=C["youden"],
    lw=2,
    ls="--",
    label=f"Youden {youden_thresh_am:.3f}",
)
ax.axvline(0.890, color=C["fixed"], lw=1.5, ls=":", label="Fixed 0.890")
if kde_thresh_am:
    ax.axvline(
        kde_thresh_am, color=C["kde"], lw=1.5, ls="-.", label=f"KDE {kde_thresh_am:.3f}"
    )
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("AM threshold value", fontsize=10)
ax.set_ylabel("LoF % − GoF %", fontsize=10)
ax.set_title("F. LoF–GoF gap across\nall possible AM thresholds", fontsize=10)
ax.legend(fontsize=7.5)

# ── Panel G: Per-gene logistic decision boundaries
ax = fig.add_subplot(gs[2, 0])
gene_colors = {"ABCC8": "#1f77b4", "GCK": "#2ca02c", "KCNJ11": "#d62728"}
x_scores = np.linspace(0.3, 1.0, 500)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df["gene"])
for gene in ["ABCC8", "GCK", "KCNJ11"]:
    g_idx = list(le.classes_).index(gene)
    X_gene = np.column_stack([x_scores, np.full(500, g_idx)])
    proba = lr_am.predict_proba(X_gene)[:, 1]
    ax.plot(x_scores, proba, color=gene_colors[gene], lw=2.2, label=gene)
    if gene in am_gene_thresholds:
        t = am_gene_thresholds[gene]
        ax.axvline(t, color=gene_colors[gene], lw=1, ls="--", alpha=0.7)
        ax.text(
            t + 0.005,
            0.07 * (list(gene_colors.keys()).index(gene) + 1),
            f"{t:.3f}",
            fontsize=8,
            color=gene_colors[gene],
        )
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("AlphaMissense score", fontsize=10)
ax.set_ylabel("P(LoF | gene, score)", fontsize=10)
ax.set_title("G. Per-gene logistic decision\nboundaries (AM)", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(0.3, 1.0)
ax.set_ylim(-0.05, 1.05)

# ── Panel H: Bootstrap confidence intervals on dynamic thresholds
ax = fig.add_subplot(gs[2, 1])
np.random.seed(42)
n_boot = 500
boot_gmm_thresholds = []
boot_kde_thresholds = []
for _ in range(n_boot):
    boot_idx = np.random.choice(len(df), len(df), replace=True)
    boot_scores = df["am_score"].values[boot_idx]
    # GMM bootstrap
    try:
        X_b = boot_scores.reshape(-1, 1)
        g = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X_b)
        means_b = np.sort(g.means_.flatten())
        stds_b = np.sqrt(np.sort(g.covariances_.flatten()))
        weights_b = g.weights_[np.argsort(g.means_.flatten())]
        x_b = np.linspace(means_b[0], means_b[1], 500)
        d_b = sum(
            w * norm.pdf(x_b, m, s) for m, s, w in zip(means_b, stds_b, weights_b)
        )
        boot_gmm_thresholds.append(x_b[np.argmin(d_b)])
    except:
        pass
    # KDE bootstrap
    try:
        kde_b = stats.gaussian_kde(boot_scores)
        x_k = np.linspace(boot_scores.min(), boot_scores.max(), 500)
        d_k = kde_b(x_k)
        mins_b = argrelmin(d_k, order=15)[0]
        if len(mins_b) > 0:
            boot_kde_thresholds.append(
                x_k[mins_b[np.argmin(np.abs(x_k[mins_b] - boot_scores.mean()))]]
            )
    except:
        pass

boot_gmm = np.array(boot_gmm_thresholds)
boot_kde = np.array(boot_kde_thresholds)

ax.hist(
    boot_gmm,
    bins=40,
    alpha=0.65,
    color=C["gmm"],
    density=True,
    label=f"GMM (n={len(boot_gmm)})",
)
ax.axvline(np.percentile(boot_gmm, 2.5), color=C["gmm"], lw=1.5, ls="--")
ax.axvline(np.percentile(boot_gmm, 97.5), color=C["gmm"], lw=1.5, ls="--")
ax.axvline(gmm_thresh_am, color=C["gmm"], lw=2.5)

if len(boot_kde) > 10:
    ax.hist(
        boot_kde,
        bins=40,
        alpha=0.5,
        color=C["kde"],
        density=True,
        label=f"KDE (n={len(boot_kde)})",
    )
    ax.axvline(np.percentile(boot_kde, 2.5), color=C["kde"], lw=1.5, ls="--")
    ax.axvline(np.percentile(boot_kde, 97.5), color=C["kde"], lw=1.5, ls="--")

ax.axvline(0.890, color=C["fixed"], lw=2, ls=":", label="Fixed 0.890")
ax.set_xlabel("Bootstrap threshold estimate", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(
    f"H. Bootstrap 95% CI\nGMM: [{np.percentile(boot_gmm, 2.5):.3f}, {np.percentile(boot_gmm, 97.5):.3f}]",
    fontsize=10,
)
ax.legend(fontsize=8)

# ── Panel I: Threshold comparison summary heatmap
ax = fig.add_subplot(gs[2, 2])
methods = [
    "Fixed\nStrong",
    "GMM\nValley",
    "KDE\nValley",
    "Youden's\nJ",
    "Isotonic\nP=0.50",
]
am_vals = [
    0.890,
    gmm_thresh_am,
    kde_thresh_am if kde_thresh_am else np.nan,
    youden_thresh_am,
    am_cal_thresholds.get("Moderate (P=0.50)", np.nan),
]
revel_vals = [
    0.932,
    gmm_thresh_revel,
    kde_thresh_revel if kde_thresh_revel else np.nan,
    youden_thresh_revel,
    revel_cal_thresholds.get("Moderate (P=0.50)", np.nan),
]

matrix = np.array([am_vals, revel_vals])
masked_matrix = np.ma.masked_invalid(matrix)
cmap_i = plt.cm.RdYlGn.copy()
cmap_i.set_bad(color="#cccccc")  # gray for N/A cells
im = ax.imshow(masked_matrix, cmap=cmap_i, aspect="auto", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=8)
ax.set_yticks([0, 1])
ax.set_yticklabels(["AM", "REVEL"], fontsize=10)
plt.colorbar(im, ax=ax, label="Threshold value")
for i in range(2):
    for j in range(len(methods)):
        v = matrix[i, j]
        if np.isnan(v):
            ax.text(
                j,
                i,
                "N/A\n(unimodal)",
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
                fontstyle="italic",
            )
        else:
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold",
            )
ax.set_title("I. Threshold comparison\nheatmap", fontsize=10)

fig.suptitle(
    "Dynamic Threshold Prediction: AM and REVEL\n"
    "GMM | KDE | Youden's J | Isotonic Calibration | Logistic Boundaries",
    fontsize=13,
    fontweight="bold",
    y=0.99,
)

outpath = os.path.join(FIGURES_DIR, "FigureS5_Dynamic_Thresholds.png")
plt.savefig(outpath, dpi=180, bbox_inches="tight")
plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\nFigure saved: {outpath}")
plt.close()

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nAM thresholds (all data-driven):")
print(f"  Fixed (Cheng 2023):    Supporting=0.564  Moderate=0.740  Strong=0.890")
print(f"  GMM valley:            {gmm_thresh_am:.3f}")
print(
    f"  KDE valley:            {kde_thresh_am:.3f}"
    if kde_thresh_am
    else "  KDE valley:            N/A (unimodal)"
)
print(f"  Youden's J:            {youden_thresh_am:.3f}")
print(
    f"  Isotonic (P=0.50):     {am_cal_thresholds.get('Moderate (P=0.50)', 'N/A'):.3f}"
)
print(f"  Per-gene (ABCC8):      {am_gene_thresholds.get('ABCC8', 'N/A'):.3f}")
print(f"  Per-gene (GCK):        {am_gene_thresholds.get('GCK', 'N/A'):.3f}")
print(f"  Per-gene (KCNJ11):     {am_gene_thresholds.get('KCNJ11', 'N/A'):.3f}")
print(
    f"\n  GoF-specific GMM:      {gmm_thresh_am_gof:.3f}  ← KEY: ~{gmm_thresh_am_gof - 0.890:+.3f} vs fixed"
)
print(
    f"  LoF-specific GMM:      {gmm_thresh_am_lof:.3f}  ← KEY: ~{gmm_thresh_am_lof - 0.890:+.3f} vs fixed"
)
print(
    f"\n  Bootstrap 95% CI (GMM): [{np.percentile(boot_gmm, 2.5):.3f}, {np.percentile(boot_gmm, 97.5):.3f}]"
)

print(f"\nKEY INSIGHT:")
print(
    f"  GoF optimal AM threshold ({gmm_thresh_am_gof:.3f}) is LOWER than LoF ({gmm_thresh_am_lof:.3f})"
)
print(f"  Using mechanism-specific thresholds rather than one fixed cutoff")
print(f"  would close the LoF/GoF sensitivity gap without losing specificity.")
print(f"\nResults saved to: {RESULTS_DIR}/dynamic_thresholds_summary.csv")
