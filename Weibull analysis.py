#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. READ EXCEL DATA
# =====================================================
file_path = "Impact data All.xlsx"
sheet_name = "Sheet1"

df = pd.read_excel(file_path, sheet_name=sheet_name)
df.columns = df.columns.str.strip()
df = df[['Condition', 'Breaking_Load']].dropna()
df = df[df['Breaking_Load'] > 0]

conditions = df['Condition'].unique()

# =====================================================
# 2. MEDIAN RANK FORMULAS (10 METHODS - FROM CODE 2)
# =====================================================
def median_rank(i, n, method):
    if method == "Bernard":
        return (i - 0.3) / (n + 0.4)
    elif method == "Hazen":
        return (i - 0.5) / n
    elif method == "Benard-BosLevenbach":
        return (i - 0.31) / (n + 0.38)
    elif method == "Blom":
        return (i - 0.375) / (n + 0.25)
    elif method == "Kaplan-Meier":
        return i / n
    elif method == "Mean":
        return i / (n + 1)
    elif method == "Gringorten":
        return (i - 0.44) / (n + 0.12)
    elif method == "Filliben":
        return (i - 0.3175) / (n + 1.635)
    elif method == "Chegodaye":
        return (i - 0.5) / (n + 0.5)
    elif method == "Cunnane":
        return (i - 0.4) / (n + 0.2)
    else:
        raise ValueError(f"Unknown method: {method}")

methods = [
    "Bernard", "Hazen", "Benard-BosLevenbach", "Blom",
    "Kaplan-Meier", "Mean", "Gringorten", "Filliben",
    "Chegodaye", "Cunnane"
]

# =====================================================
# 3. SINGLE-RANK WEIBULL FIT
# =====================================================
def fit_weibull_single_rank(data, method):
    data = np.sort(data)
    n = len(data)
    i = np.arange(1, n + 1)

    P = median_rank(i, n, method)
    mask = (P > 0) & (P < 1)

    data = data[mask]
    P = P[mask]

    x = np.log(data)
    y = np.log(-np.log(1 - P))

    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    beta = slope
    eta = np.exp(-intercept / beta)
    r2 = r_value**2

    return beta, eta, P, r2, data, x, y, slope, intercept

# =====================================================
# 4. RANK MATRIX (ALL METHODS)
# =====================================================
def rank_matrix(data):
    data = np.sort(data)
    n = len(data)
    i = np.arange(1, n + 1)

    P_all = []
    for m in methods:
        P_all.append(median_rank(i, n, m))

    return np.column_stack(P_all), data

# =====================================================
# 5. ML OBJECTIVE FUNCTION
# =====================================================
def weibull_r2_loss(w, Pmat, data):
    if np.any(w < 0):
        return 1e6

    P = Pmat @ w

    if np.any(P <= 0) or np.any(P >= 1):
        return 1e6

    x = np.log(data)
    y = np.log(-np.log(1 - P))

    _, _, r, _, _ = stats.linregress(x, y)
    return 1 - r**2

# =====================================================
# 6. ML-ENSEMBLE WEIBULL FIT
# =====================================================
def fit_weibull_ml_rank(data):
    Pmat, data = rank_matrix(data)

    w0 = np.ones(len(methods)) / len(methods)

    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    res = minimize(
        weibull_r2_loss,
        w0,
        args=(Pmat, data),
        bounds=[(0, 1)] * len(methods),
        constraints=constraint,
        method="SLSQP"
    )

    w = res.x
    P = Pmat @ w
    eps = 1e-10
    P = np.clip(P, eps, 1 - eps)

    x = np.log(data)
    y = np.log(-np.log(1 - P))

    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    beta = slope
    eta = np.exp(-intercept / beta)
    r2 = r_value**2

    return beta, eta, P, w, r2, data, x, y, slope, intercept

# =====================================================
# 7. WEIBULL CDF & PDF
# =====================================================
def weibull_cdf(x, beta, eta):
    return 1 - np.exp(-(x / eta)**beta)

def weibull_reliability(x, beta, eta):
    return np.exp(-(x / eta)**beta)

# =====================================================
# 8. GOODNESS-OF-FIT TESTS (FROM CODE 1)
# =====================================================
def anderson_darling_weibull(data, beta, eta):
    """Anderson-Darling test for Weibull distribution"""
    data = np.sort(data)
    n = len(data)
    
    # Theoretical CDF
    F = weibull_cdf(data, beta, eta)
    
    # Anderson-Darling statistic
    i = np.arange(1, n + 1)
    AD = -n - np.sum((2*i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n
    
    return AD

def kolmogorov_smirnov_weibull(data, beta, eta):
    """Kolmogorov-Smirnov test for Weibull distribution"""
    data = np.sort(data)
    n = len(data)
    
    # Empirical CDF
    F_empirical = np.arange(1, n + 1) / n
    
    # Theoretical CDF
    F_theoretical = weibull_cdf(data, beta, eta)
    
    # KS statistic
    D = np.max(np.abs(F_empirical - F_theoretical))
    
    # Critical value at 0.05 significance (approximate)
    D_critical = 1.36 / np.sqrt(n)
    
    return D, D_critical

# =====================================================
# 9. PERCENTILE PREDICTIONS (FROM CODE 1)
# =====================================================
def weibull_percentile(p, beta, eta):
    """Calculate x value at which F(x) = p"""
    return eta * (-np.log(1 - p))**(1/beta)

def percentile_confidence_interval(beta, eta, beta_ci, eta_ci, p):
    """Bootstrap-based CI for percentiles"""
    # Lower bound: use lower beta and lower eta
    x_lower = eta_ci[0] * (-np.log(1 - p))**(1/beta_ci[0])
    
    # Upper bound: use upper beta and upper eta
    x_upper = eta_ci[1] * (-np.log(1 - p))**(1/beta_ci[1])
    
    # Point estimate
    x_point = eta * (-np.log(1 - p))**(1/beta)
    
    return x_point, x_lower, x_upper

# =====================================================
# 10. TWO-SAMPLE WEIBULL COMPARISON (FROM CODE 1)
# =====================================================
def compare_weibull_distributions(data1, data2, label1="Condition 1", label2="Condition 2"):
    """Compare two Weibull distributions using likelihood ratio test"""
    
    # Fit individual models
    beta1, eta1, _, _, _, d1, _, _, _, _ = fit_weibull_ml_rank(data1)
    beta2, eta2, _, _, _, d2, _, _, _, _ = fit_weibull_ml_rank(data2)
    
    # Log-likelihood for individual fits
    LL1 = np.sum(np.log(beta1/eta1) + (beta1-1)*np.log(d1/eta1) - (d1/eta1)**beta1)
    LL2 = np.sum(np.log(beta2/eta2) + (beta2-1)*np.log(d2/eta2) - (d2/eta2)**beta2)
    
    # Combined data fit (null hypothesis: same distribution)
    combined_data = np.concatenate([data1, data2])
    beta_c, eta_c, _, _, _, d_c, _, _, _, _ = fit_weibull_ml_rank(combined_data)
    LL_combined = np.sum(np.log(beta_c/eta_c) + (beta_c-1)*np.log(d_c/eta_c) - (d_c/eta_c)**beta_c)
    
    # Likelihood ratio test
    LR = 2 * ((LL1 + LL2) - LL_combined)
    df = 2  # 2 additional parameters (beta and eta for second distribution)
    p_value = 1 - stats.chi2.cdf(LR, df)
    
    # Effect size: ratio of characteristic strengths
    strength_ratio = eta2 / eta1
    strength_diff_pct = (eta2 - eta1) / eta1 * 100
    
    return {
        'beta1': beta1, 'eta1': eta1,
        'beta2': beta2, 'eta2': eta2,
        'LR_statistic': LR,
        'p_value': p_value,
        'strength_ratio': strength_ratio,
        'strength_diff_pct': strength_diff_pct,
        'significant': p_value < 0.05
    }

# =====================================================
# 11. BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================
def bootstrap_ci(data, n_boot=2000):
    betas, etas = [], []

    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        beta, eta, _, _, _, _, _, _, _, _ = fit_weibull_ml_rank(sample)
        betas.append(beta)
        etas.append(eta)

    beta_ci = np.percentile(betas, [2.5, 97.5])
    eta_ci = np.percentile(etas, [2.5, 97.5])

    return beta_ci, eta_ci

# =====================================================
# 12. CDF CONFIDENCE BANDS (FROM CODE 2)
# =====================================================
def bootstrap_cdf_bands(data, x_vals, n_boot=1000):
    """Bootstrap confidence bands for the CDF"""
    cdf_boot = []

    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        beta, eta, _, _, _, _, _, _, _, _ = fit_weibull_ml_rank(sample)
        cdf_boot.append(weibull_cdf(x_vals, beta, eta))

    cdf_boot = np.array(cdf_boot)
    lower = np.percentile(cdf_boot, 2.5, axis=0)
    upper = np.percentile(cdf_boot, 97.5, axis=0)

    return lower, upper

# =====================================================
# MAIN ANALYSIS LOOP
# =====================================================
print("\n" + "="*80)
print("COMPREHENSIVE WEIBULL ANALYSIS RESULTS (COMBINED VERSION)")
print("="*80)

# Storage for results
results = {}
excel_results = []
comparison_results = []

for cond in conditions:
    data = df[df['Condition'] == cond]['Breaking_Load'].values
    
    # Find best single rank method
    best_r2 = -np.inf
    for m in methods:
        beta, eta, P, r2, d, x, y, slope, intercept = fit_weibull_single_rank(data, m)
        if r2 > best_r2:
            best_r2 = r2
            best_method = m
            best_vals = (beta, eta, P, r2, d, x, y, slope, intercept)
    
    # ML ensemble
    beta_m, eta_m, P_m, w, r2_m, d_m, x_m, y_m, slope_m, intercept_m = fit_weibull_ml_rank(data)
    
    # Bootstrap CI
    beta_ci, eta_ci = bootstrap_ci(data)
    
    # Goodness-of-fit tests
    AD_stat = anderson_darling_weibull(data, beta_m, eta_m)
    KS_stat, KS_critical = kolmogorov_smirnov_weibull(data, beta_m, eta_m)
    
    # Percentile predictions (B10, B50, B90)
    B10, B10_lower, B10_upper = percentile_confidence_interval(beta_m, eta_m, beta_ci, eta_ci, 0.10)
    B50, B50_lower, B50_upper = percentile_confidence_interval(beta_m, eta_m, beta_ci, eta_ci, 0.50)
    B90, B90_lower, B90_upper = percentile_confidence_interval(beta_m, eta_m, beta_ci, eta_ci, 0.90)
    
    # CDF confidence bands
    x_vals = np.linspace(min(data), max(data), 100)
    cdf_lower, cdf_upper = bootstrap_cdf_bands(data, x_vals)
    
    results[cond] = {
        'best': best_vals,
        'best_method': best_method,
        'ml': (beta_m, eta_m, P_m, w, r2_m, d_m, x_m, y_m, slope_m, intercept_m),
        'ci': (beta_ci, eta_ci),
        'data': data,
        'gof': {'AD': AD_stat, 'KS': KS_stat, 'KS_critical': KS_critical},
        'percentiles': {
            'B10': (B10, B10_lower, B10_upper),
            'B50': (B50, B50_lower, B50_upper),
            'B90': (B90, B90_lower, B90_upper)
        },
        'cdf_bands': (x_vals, cdf_lower, cdf_upper)
    }
    
    # Store for Excel output
    excel_row = {
        'Condition': cond,
        'n': len(data),
        'Best_Method': best_method,
        'Beta_Best': best_vals[0],
        'Eta_Best': best_vals[1],
        'R2_Best': best_vals[3],
        'Beta_ML': beta_m,
        'Eta_ML': eta_m,
        'R2_ML': r2_m,
        'R2_Improvement': r2_m - best_vals[3],
        'Beta_CI_Lower': beta_ci[0],
        'Beta_CI_Upper': beta_ci[1],
        'Eta_CI_Lower': eta_ci[0],
        'Eta_CI_Upper': eta_ci[1],
        'AD_Statistic': AD_stat,
        'KS_Statistic': KS_stat,
        'KS_Critical': KS_critical,
        'KS_Pass': 'Yes' if KS_stat < KS_critical else 'No',
        'B10': B10,
        'B10_CI_Lower': B10_lower,
        'B10_CI_Upper': B10_upper,
        'B50': B50,
        'B50_CI_Lower': B50_lower,
        'B50_CI_Upper': B50_upper,
        'B90': B90,
        'B90_CI_Lower': B90_lower,
        'B90_CI_Upper': B90_upper,
    }
    
    # Add weights
    for method_name, weight in zip(methods, w):
        excel_row[f'Weight_{method_name}'] = weight
    
    excel_results.append(excel_row)
    
    # Print console output
    print(f"\n{'─'*80}")
    print(f"Condition: {cond}")
    print(f"{'─'*80}")
    print(f"  Sample size: n = {len(data)}")
    print(f"  Best single method: {best_method} (R² = {best_r2:.4f})")
    print(f"  ML ensemble R²: {r2_m:.4f}")
    print(f"  R² Improvement: {r2_m - best_r2:.4f}")
    print(f"\n  Weibull Parameters (ML Ensemble):")
    print(f"    Shape (β):          {beta_m:.3f}  [95% CI: {beta_ci[0]:.3f}, {beta_ci[1]:.3f}]")
    print(f"    Scale (η):          {eta_m:.2f} N  [95% CI: {eta_ci[0]:.2f}, {eta_ci[1]:.2f}]")
    print(f"\n  Method Weights (Top 5):")
    weight_dict = {m: w_val for m, w_val in zip(methods, w)}
    sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
    for method_name, weight in sorted_weights[:5]:
        print(f"    {method_name:20s}: {weight:.4f}")
    print(f"\n  Goodness-of-Fit:")
    print(f"    Anderson-Darling:   {AD_stat:.4f}  (lower is better)")
    print(f"    Kolmogorov-Smirnov: {KS_stat:.4f}  (critical = {KS_critical:.4f})")
    print(f"    KS Test Result:     {'PASS' if KS_stat < KS_critical else 'FAIL'} (α=0.05)")
    print(f"\n  Characteristic Strengths:")
    print(f"    B10 (10% failure):  {B10:.2f} N  [95% CI: {B10_lower:.2f}, {B10_upper:.2f}]")
    print(f"    B50 (50% failure):  {B50:.2f} N  [95% CI: {B50_lower:.2f}, {B50_upper:.2f}]")
    print(f"    B90 (90% failure):  {B90:.2f} N  [95% CI: {B90_lower:.2f}, {B90_upper:.2f}]")

# =====================================================
# STATISTICAL COMPARISON BETWEEN CONDITIONS
# =====================================================
print(f"\n\n{'='*80}")
print("STATISTICAL COMPARISON BETWEEN CONDITIONS")
print(f"{'='*80}")

condition_list = list(results.keys())

if len(condition_list) > 1:
    for i in range(len(condition_list)):
        for j in range(i+1, len(condition_list)):
            cond1 = condition_list[i]
            cond2 = condition_list[j]
            
            comp = compare_weibull_distributions(
                results[cond1]['data'],
                results[cond2]['data'],
                cond1, cond2
            )
            
            comparison_results.append({
                'Condition_1': cond1,
                'Condition_2': cond2,
                'Beta_1': comp['beta1'],
                'Eta_1': comp['eta1'],
                'Beta_2': comp['beta2'],
                'Eta_2': comp['eta2'],
                'Eta_Difference_Pct': comp['strength_diff_pct'],
                'LR_Statistic': comp['LR_statistic'],
                'p_value': comp['p_value'],
                'Significant_at_0.05': 'Yes' if comp['significant'] else 'No'
            })
            
            print(f"\n{cond1} vs {cond2}:")
            print(f"  β₁ = {comp['beta1']:.3f},  β₂ = {comp['beta2']:.3f}")
            print(f"  η₁ = {comp['eta1']:.2f} N,  η₂ = {comp['eta2']:.2f} N")
            print(f"  Difference: {comp['strength_diff_pct']:+.1f}%")
            print(f"  Likelihood Ratio: {comp['LR_statistic']:.3f}")
            print(f"  p-value: {comp['p_value']:.4f}")
            print(f"  Significant (α=0.05): {'YES***' if comp['significant'] else 'No'}")

# =====================================================
# SAVE RESULTS TO EXCEL (FROM CODE 2 STYLE)
# =====================================================
excel_df = pd.DataFrame(excel_results)
comparison_df = pd.DataFrame(comparison_results) if comparison_results else pd.DataFrame()

with pd.ExcelWriter("weibull_analysis_combined_results.xlsx") as writer:
    excel_df.to_excel(writer, sheet_name="Full Results", index=False)
    if not comparison_df.empty:
        comparison_df.to_excel(writer, sheet_name="Pairwise Comparisons", index=False)

print(f"\n\n{'='*80}")
print("EXCEL OUTPUT SAVED")
print(f"{'='*80}")
print("✓ File: weibull_analysis_combined_results.xlsx")
print("  - Sheet 1: Full Results (all parameters, weights, CIs)")
if not comparison_df.empty:
    print("  - Sheet 2: Pairwise Comparisons (statistical tests)")

# Also save CSV files (from Code 1 style)
summary_data = []
for cond in results.keys():
    beta_m = results[cond]['ml'][0]
    eta_m = results[cond]['ml'][1]
    beta_ci = results[cond]['ci'][0]
    eta_ci = results[cond]['ci'][1]
    B10 = results[cond]['percentiles']['B10'][0]
    B50 = results[cond]['percentiles']['B50'][0]
    B90 = results[cond]['percentiles']['B90'][0]
    
    summary_data.append({
        'Condition': cond,
        'n': len(results[cond]['data']),
        'β': f"{beta_m:.3f}",
        'β_CI': f"[{beta_ci[0]:.3f}, {beta_ci[1]:.3f}]",
        'η_N': f"{eta_m:.2f}",
        'η_CI_N': f"[{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]",
        'B10_N': f"{B10:.2f}",
        'B50_N': f"{B50:.2f}",
        'B90_N': f"{B90:.2f}",
        'R²': f"{results[cond]['ml'][4]:.4f}",
        'AD': f"{results[cond]['gof']['AD']:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("weibull_summary_table.csv", index=False)
if not comparison_df.empty:
    comparison_df.to_csv("weibull_comparison_table.csv", index=False)

print("\n✓ CSV files also saved:")
print("  - weibull_summary_table.csv")
if not comparison_df.empty:
    print("  - weibull_comparison_table.csv")

# =====================================================
# FIGURE 1: SIDE-BY-SIDE WITH CDF BANDS (ENHANCED)
# =====================================================
fig1, axes = plt.subplots(
    len(conditions), 2,
    figsize=(12, 4 * len(conditions)),
    sharey=True
)

# Ensure axes is 2D even if one condition
if len(conditions) == 1:
    axes = axes.reshape(1, -1)

# Font size settings
TITLE_FONTSIZE = 14
XLABEL_FONTSIZE = 20
YLABEL_FONTSIZE = 20
MAJOR_TICK_FONTSIZE = 16
MINOR_TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
SCATTER_MARKERSIZE = 45
LINE_WIDTH = 2

for i, cond in enumerate(conditions):
    # Extract best and ML results
    beta_s, eta_s, P_s, r2_s, d_s, x_s, y_s, slope_s, intercept_s = results[cond]['best']
    best_method = results[cond]['best_method']
    beta_m, eta_m, P_m, w, r2_m, d_m, x_m, y_m, slope_m, intercept_m = results[cond]['ml']
    x_vals, cdf_lower, cdf_upper = results[cond]['cdf_bands']

    # ------------------- Best Rank Plot -------------------
    axes[i, 0].scatter(d_s, P_s, s=SCATTER_MARKERSIZE, zorder=3)
    axes[i, 0].plot(d_s, weibull_cdf(d_s, beta_s, eta_s), 'r-', lw=LINE_WIDTH, zorder=2)
    axes[i, 0].set_title(
        f"{cond}\nBest Rank: {best_method}\nβ={beta_s:.3f}, η={eta_s:.1f} N, R²={r2_s:.4f}",
        fontsize=TITLE_FONTSIZE
    )
    axes[i, 0].set_xlabel("Notch Impact Strength (kJ/$m^2$)", fontsize=XLABEL_FONTSIZE)
    axes[i, 0].grid(alpha=0.3)

    # ------------------- ML Ensemble Plot -------------------
    axes[i, 1].scatter(d_m, P_m, s=SCATTER_MARKERSIZE, zorder=3)
    axes[i, 1].plot(d_m, weibull_cdf(d_m, beta_m, eta_m), 'r-', lw=LINE_WIDTH, zorder=2, label='Fitted CDF')
    axes[i, 1].fill_between(
        x_vals, cdf_lower, cdf_upper, alpha=0.3, color='blue', label='95% CI', zorder=1
    )
    axes[i, 1].set_title(
        f"{cond}\nML Ensemble\nβ={beta_m:.3f}, η={eta_m:.1f} N, R²={r2_m:.4f}",
        fontsize=TITLE_FONTSIZE
    )
    axes[i, 1].set_xlabel("Notch Impact Strength (kJ/$m^2$)", fontsize=XLABEL_FONTSIZE)
    axes[i, 1].legend(loc='best', fontsize=LEGEND_FONTSIZE)
    axes[i, 1].grid(alpha=0.3)

    # ------------------- Major and Minor Tick Font Sizes -------------------
    for j in range(2):
        axes[i, j].minorticks_on()  # enable minor ticks
        axes[i, j].tick_params(axis='x', which='major', labelsize=MAJOR_TICK_FONTSIZE)
        axes[i, j].tick_params(axis='x', which='minor', labelsize=MINOR_TICK_FONTSIZE)
        axes[i, j].tick_params(axis='y', which='major', labelsize=MAJOR_TICK_FONTSIZE)
        axes[i, j].tick_params(axis='y', which='minor', labelsize=MINOR_TICK_FONTSIZE)

# ------------------- Shared Y-axis Label -------------------
axes[0, 0].set_ylabel("Failure Probability", fontsize=YLABEL_FONTSIZE)

# ------------------- Layout and Save -------------------
plt.tight_layout()
plt.savefig("Fig1_best_vs_ml_with_bands_impact.png", dpi=300, bbox_inches="tight")
plt.close()


# =====================================================
# FIGURE 2: WEIBULL PROBABILITY PLOTS
# =====================================================
fig2, axes2 = plt.subplots(
    len(conditions), 1,
    figsize=(6, 4 * len(conditions))
)

if len(conditions) == 1:
    axes2 = [axes2]

# Font size settings
XLABEL_FONTSIZE = 20
YLABEL_FONTSIZE = 20
TITLE_FONTSIZE = 14
MAJOR_TICK_FONTSIZE = 16
MINOR_TICK_FONTSIZE = 12
TEXTBOX_FONTSIZE = 12
MARKER_SIZE = 6
LINE_WIDTH = 2

for i, cond in enumerate(conditions):
    beta_s, eta_s, P_s, r2_s, d_s, x_s, y_s, slope_s, intercept_s = results[cond]['best']
    best_method = results[cond]['best_method']

    # Plot data points and regression line
    axes2[i].plot(x_s, y_s, 'o', markersize=MARKER_SIZE)
    axes2[i].plot(x_s, slope_s * x_s + intercept_s, 'r-', lw=LINE_WIDTH)

    # Equation textbox
    equation = f"y = {slope_s:.2f}x - {abs(intercept_s):.2f}" if intercept_s < 0 else f"y = {slope_s:.2f}x + {intercept_s:.2f}"
    axes2[i].text(
        0.05, 0.95, equation, transform=axes2[i].transAxes, 
        fontsize=TEXTBOX_FONTSIZE, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Labels and title
    axes2[i].set_xlabel(r"$x = \ln(\sigma_i)$", fontsize=XLABEL_FONTSIZE)
    axes2[i].set_ylabel(r"$y = \ln[\ln(1/(1-P_i))]$", fontsize=YLABEL_FONTSIZE)
    axes2[i].set_title(f"{cond} - Best Method: {best_method} (R²={r2_s:.4f})", fontsize=TITLE_FONTSIZE)
    axes2[i].grid(alpha=0.3)

    # ------------------- Major and Minor Tick Font Sizes -------------------
    axes2[i].minorticks_on()  # enable minor ticks
    axes2[i].tick_params(axis='x', which='major', labelsize=MAJOR_TICK_FONTSIZE)
    axes2[i].tick_params(axis='x', which='minor', labelsize=MINOR_TICK_FONTSIZE)
    axes2[i].tick_params(axis='y', which='major', labelsize=MAJOR_TICK_FONTSIZE)
    axes2[i].tick_params(axis='y', which='minor', labelsize=MINOR_TICK_FONTSIZE)

plt.tight_layout()
plt.savefig("Fig2_weibull_probability_plots_impact.png", dpi=300, bbox_inches="tight")
plt.close()


# =====================================================
# FIGURE 3: TRADITIONAL VS ENSEMBLE COMPARISON
# =====================================================
fig3, axes3 = plt.subplots(
    len(conditions), 1,
    figsize=(8, 4 * len(conditions))
)

if len(conditions) == 1:
    axes3 = [axes3]

# Font size settings
XLABEL_FONTSIZE = 20
YLABEL_FONTSIZE = 20
TITLE_FONTSIZE = 14
MAJOR_TICK_FONTSIZE = 16
MINOR_TICK_FONTSIZE = 12
SCATTER_MARKERSIZE = 6
LINE_WIDTH = 2

for i, cond in enumerate(conditions):
    beta_s, eta_s, P_s, r2_s, d_s, x_s, y_s, slope_s, intercept_s = results[cond]['best']
    best_method = results[cond]['best_method']
    beta_m, eta_m, P_m, w, r2_m, d_m, x_m, y_m, slope_m, intercept_m = results[cond]['ml']

    # Plot traditional data
    axes3[i].plot(x_s, y_s, 'o', label=f'Data ({best_method})', markersize=SCATTER_MARKERSIZE, color='blue')
    axes3[i].plot(x_s, slope_s * x_s + intercept_s, '-', lw=LINE_WIDTH,
                 label=f'Traditional (R²={r2_s:.4f})', color='red')

    # Plot ML Ensemble data
    axes3[i].plot(x_m, y_m, 's', label='Data (ML Ensemble)', markersize=SCATTER_MARKERSIZE-1,
                 color='green', alpha=0.6)
    axes3[i].plot(x_m, slope_m * x_m + intercept_m, '--', lw=LINE_WIDTH,
                 label=f'ML Ensemble (R²={r2_m:.4f})', color='orange')

    # Labels and title
    axes3[i].set_xlabel(r"$x = \ln(\sigma_i)$", fontsize=XLABEL_FONTSIZE)
    axes3[i].set_ylabel(r"$y = \ln[\ln(1/(1-P_i))]$", fontsize=YLABEL_FONTSIZE)
    axes3[i].set_title(f"{cond} - Weibull Plot Comparison", fontsize=TITLE_FONTSIZE)
    axes3[i].legend(loc='best')
    axes3[i].grid(alpha=0.3)

    # ------------------- Major and Minor Tick Font Sizes -------------------
    axes3[i].minorticks_on()  # enable minor ticks
    axes3[i].tick_params(axis='x', which='major', labelsize=MAJOR_TICK_FONTSIZE)
    axes3[i].tick_params(axis='x', which='minor', labelsize=MINOR_TICK_FONTSIZE)
    axes3[i].tick_params(axis='y', which='major', labelsize=MAJOR_TICK_FONTSIZE)
    axes3[i].tick_params(axis='y', which='minor', labelsize=MINOR_TICK_FONTSIZE)

plt.tight_layout()
plt.savefig("Fig3_traditional_vs_ensemble_impact.png", dpi=300, bbox_inches="tight")
plt.close()


# =====================================================
# FIGURE 4: RELIABILITY CURVES
# =====================================================
fig4, axes4 = plt.subplots(
    len(conditions), 1,
    figsize=(8, 4 * len(conditions))
)

if len(conditions) == 1:
    axes4 = [axes4]

for i, cond in enumerate(conditions):
    beta_s, eta_s, P_s, r2_s, d_s, x_s, y_s, slope_s, intercept_s = results[cond]['best']
    best_method = results[cond]['best_method']
    beta_m, eta_m, P_m, w, r2_m, d_m, x_m, y_m, slope_m, intercept_m = results[cond]['ml']
    
    x_range = np.linspace(d_s.min() * 0.9, d_s.max() * 1.1, 200)
    
    R_traditional = weibull_reliability(x_range, beta_s, eta_s)
    R_ensemble = weibull_reliability(x_range, beta_m, eta_m)
    
    axes4[i].plot(x_range, R_traditional, '-', lw=2.5, 
                 label=f'Traditional (β={beta_s:.2f}, η={eta_s:.1f})', color='blue')
    axes4[i].plot(x_range, R_ensemble, '--', lw=2.5, 
                 label=f'ML Ensemble (β={beta_m:.2f}, η={eta_m:.1f})', color='red')
    
    R_data = 1 - P_s
    axes4[i].plot(d_s, R_data, 'o', markersize=5, color='gray', 
                 alpha=0.5, label='Empirical')
    
    axes4[i].set_xlabel("Notch Impact Strength (kJ/$m^2$)")
    axes4[i].set_ylabel("Reliability R(σ)")
    axes4[i].set_title(f"{cond} - Weibull Reliability Function")
    axes4[i].legend(loc='best')
    axes4[i].grid(alpha=0.3)
    axes4[i].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig("Fig4_reliability_curves_impact.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# FIGURE 5: PERCENTILE PREDICTIONS WITH CI
# =====================================================
fig5, ax5 = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(conditions))
width = 0.25

percentiles = ['B10', 'B50', 'B90']
colors = ['#e74c3c', '#3498db', '#2ecc71']
offsets = [-width, 0, width]

for i, perc in enumerate(percentiles):
    values = [results[cond]['percentiles'][perc][0] for cond in conditions]
    lower = [results[cond]['percentiles'][perc][1] for cond in conditions]
    upper = [results[cond]['percentiles'][perc][2] for cond in conditions]
    
    errors = [[abs(values[j] - lower[j]) for j in range(len(values))],
              [abs(upper[j] - values[j]) for j in range(len(values))]]
    
    ax5.bar(x_pos + offsets[i], values, width, label=perc,
            color=colors[i], alpha=0.7, edgecolor='black')
    ax5.errorbar(x_pos + offsets[i], values, yerr=errors, 
                 fmt='none', color='black', capsize=4, linewidth=1.5)

# ------------------- Labels and Title -------------------
ax5.set_xlabel('Condition', fontsize=12, fontweight='bold')
ax5.set_ylabel('Notch Impact Strength (kJ/$m^2$)', fontsize=12, fontweight='bold')
ax5.set_title('Characteristic Strengths (B10, B50, B90) with 95% Confidence Intervals', 
              fontsize=14, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(conditions, rotation=45, ha='right')
ax5.legend(title='Percentile', fontsize=10)
ax5.grid(axis='y', alpha=0.3)

# ------------------- Major and Minor Tick Font Sizes -------------------
ax5.minorticks_on()  # enable minor ticks
ax5.tick_params(axis='x', which='major', labelsize=12)
ax5.tick_params(axis='x', which='minor', labelsize=10)  # increase minor X tick font
ax5.tick_params(axis='y', which='major', labelsize=12)
ax5.tick_params(axis='y', which='minor', labelsize=10)  # increase minor Y tick font

plt.tight_layout()
plt.savefig("Fig5_percentile_predictions_impact.png", dpi=300, bbox_inches="tight")
plt.close()


# =====================================================
# FIGURE 6: METHOD WEIGHTS VISUALIZATION (NEW)
# =====================================================
fig6, axes6 = plt.subplots(
    len(conditions), 1,
    figsize=(10, 4 * len(conditions))
)

if len(conditions) == 1:
    axes6 = [axes6]

# Font size settings
XLABEL_FONTSIZE = 20
YLABEL_FONTSIZE = 20
TITLE_FONTSIZE = 16
MAJOR_TICK_FONTSIZE = 20
MINOR_TICK_FONTSIZE = 14
TEXT_FONTSIZE = 20

for i, cond in enumerate(conditions):
    w = results[cond]['ml'][3]
    
    # Sort weights
    weight_dict = {m: w_val for m, w_val in zip(methods, w)}
    sorted_items = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_methods = [item[0] for item in sorted_items]
    sorted_weights = [item[1] for item in sorted_items]
    
    # Bar plot
    bars = axes6[i].barh(sorted_methods, sorted_weights, color='steelblue', edgecolor='black')
    
    # Add weight values on bars
    for j, (method, weight) in enumerate(zip(sorted_methods, sorted_weights)):
        axes6[i].text(weight + 0.01, j, f'{weight:.2f}', 
                      va='center', fontsize=TEXT_FONTSIZE)
    
    # Labels and title
    axes6[i].set_xlabel('Weight', fontsize=XLABEL_FONTSIZE)
    axes6[i].set_title(f'{cond} - Optimized Method Weights', fontsize=TITLE_FONTSIZE)

    # ------------------- Major and Minor Tick Font Sizes -------------------
    axes6[i].minorticks_on()  # enable minor ticks
    axes6[i].tick_params(axis='x', which='major', labelsize=MAJOR_TICK_FONTSIZE)
    axes6[i].tick_params(axis='x', which='minor', labelsize=MINOR_TICK_FONTSIZE)
    axes6[i].tick_params(axis='y', which='major', labelsize=MAJOR_TICK_FONTSIZE)
    axes6[i].tick_params(axis='y', which='minor', labelsize=MINOR_TICK_FONTSIZE)
    
    # Set X-axis limits and grid
    axes6[i].set_xlim([0, max(sorted_weights) * 1.15])
    axes6[i].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("Fig6_method_weights_impact.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "="*80)
print("✓ ALL FIGURES AND TABLES GENERATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  Figures:")
print("    1. Fig1_best_vs_ml_with_bands.png (CDF plots with confidence bands)")
print("    2. Fig2_weibull_probability_plots.png (Linearized plots)")
print("    3. Fig3_traditional_vs_ensemble.png (Method comparison)")
print("    4. Fig4_reliability_curves.png (Reliability functions)")
print("    5. Fig5_percentile_predictions.png (B10/B50/B90 with CI)")
print("    6. Fig6_method_weights.png (Optimized weights visualization)")
print("\n  Data Files:")
print("    - weibull_analysis_combined_results.xlsx (Full results + comparisons)")
print("    - weibull_summary_table.csv (Summary statistics)")
if comparison_results:
    print("    - weibull_comparison_table.csv (Pairwise comparisons)")
print("\n" + "="*80)


# In[ ]:




