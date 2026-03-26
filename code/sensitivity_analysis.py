import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
from model_bioreactor import growth_model

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE CONDITIONS  (representative mid-cultivation steady state)
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_CONDITIONS = dict(
    x          = 1000.0,   # g/m3  – mid-range biomass
    irradiance = 100.0,    # W/m2  – moderate indoor/outdoor light
    temperature = 29.0,    # °C    – optimal temperature (Topt)
    pH         = 9.25,     # –     – midpoint of viable window [8, 10.5]
    DO         = 10.0,     # g/m3  – typical dissolved O2
    substrate  = 150.0,    # g/m3  – mid Haldane range
    carbon     = 20.0,     # g/m3  – moderate dissolved CO2
)

# ─────────────────────────────────────────────────────────────────────────────
# KINETIC PARAMETERS TO PERTURB
# key  : attribute name on growth_model instance
# label: display name for the tornado plot y-axis
# ─────────────────────────────────────────────────────────────────────────────
PARAMETERS = [
    ("miu_max", r"$\mu_{max}$  (day$^{-1}$)"),
    ("KIi",     r"$K_{Ii}$  (W m$^{-2}$)"),
    ("KC",      r"$K_C$  (g m$^{-3}$)"),
    ("Kc",      r"$K_c$  (g m$^{-3}$)"),
    ("KS",      r"$K_S$  (g m$^{-3}$)"),
    ("Ks",      r"$K_s$  (g m$^{-3}$)"),
    ("DO_max",  r"$DO_{max}$  (g m$^{-3}$)"),
    ("x_star",  r"$X^*$  (g m$^{-3}$)"),
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER – evaluate µ at given parameter value
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_miu(**override_params):
    """Create a growth_model with baseline conditions and optionally
    override one kinetic parameter, then return µ."""
    model = growth_model(**BASELINE_CONDITIONS)
    for attr, val in override_params.items():
        setattr(model, attr, val)
    result = model.solve()
    return result[0]  # µ (day-1)


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_sensitivity(perturbations):
    """
    perturbations : list of fractional changes, e.g. [0.25, -0.25]
    Returns a list of dicts with keys:
        label, baseline, miu_high, miu_low, delta_high, delta_low
    """
    frac_high, frac_low = max(perturbations), min(perturbations)

    baseline_miu = evaluate_miu()
    results = []

    for attr, label in PARAMETERS:
        baseline_val = getattr(growth_model(**BASELINE_CONDITIONS), attr)

        val_high = baseline_val * (1 + frac_high)
        val_low  = baseline_val * (1 + frac_low)

        miu_high = evaluate_miu(**{attr: val_high})
        miu_low  = evaluate_miu(**{attr: val_low})

        if label == r'$DO_{max}$  (g m$^{-3}$)':
            label = r'[$O_{2}$]$_{U}$ (g m$^{-3}$)'
        if label == r'$K_{Ii}$  (W m$^{-2}$)':
            label = r'$K_{I}$  (W m$^{-2}$)'
        results.append({
            "label":      label,
            "baseline":   baseline_miu,
            "miu_high":   miu_high,
            "miu_low":    miu_low,
            "delta_high": miu_high - baseline_miu,
            "delta_low":  miu_low  - baseline_miu,
        })

    # Sort by total sensitivity span (largest swing on top)
    results.sort(key=lambda r: abs(r["miu_high"] - r["miu_low"]), reverse=False)
    return results, baseline_miu


# ─────────────────────────────────────────────────────────────────────────────
# TORNADO PLOT
# ─────────────────────────────────────────────────────────────────────────────
def tornado_plot(results, baseline_miu, title, pct_label, ax, index):
    COLOR_HIGH = "#1f77b4"   # red  – positive perturbation
    COLOR_LOW  = "#d62728"   # blue – negative perturbation

    labels      = [r["label"]      for r in results]
    delta_high  = [r["delta_high"] for r in results]
    delta_low   = [r["delta_low"]  for r in results]
    y_pos       = np.arange(len(labels))

    for i, (dh, dl) in enumerate(zip(delta_high, delta_low)):
        # Bar from baseline to high perturbation
        ax.barh(i, dh, left=baseline_miu, color=COLOR_HIGH,
                height=0.55, zorder=3)
        # Bar from baseline to low perturbation
        ax.barh(i, dl, left=baseline_miu, color=COLOR_LOW,
                height=0.55, zorder=3)

        # Numeric annotations
        offset = max(abs(dh), abs(dl)) * 0.04 + 0.002
        ax.text(baseline_miu + dh + np.sign(dh) * offset,
                i, f"{dh/baseline_miu*100:+.1f}%", va='center', ha='left' if dh >= 0 else 'right',
                fontsize=10, color=COLOR_HIGH)
        ax.text(baseline_miu + dl + np.sign(dl) * offset,
                i, f"{dl/baseline_miu*100:+.1f}%", va='center', ha='left' if dl >= 0 else 'right',
                fontsize=10, color=COLOR_LOW)

    # Baseline vertical line
    ax.axvline(baseline_miu, color='black', linewidth=1.2,
               linestyle='--', zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(r"Specific Growth Rate  $\mu$  (day$^{-1}$)", fontsize=12)
    ax.grid(axis='x', linestyle=':', alpha=0.5, zorder=0)
    ax.text(-0.12, 1.05, index, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')
    ax.set_axisbelow(True)

    # Legend
    patch_high = mpatches.Patch(color=COLOR_HIGH, label=f'+{int(pct_label*100)}% perturbation')
    patch_low  = mpatches.Patch(color=COLOR_LOW,  label=f'-{int(pct_label*100)}% perturbation')
    ax.legend(handles=[patch_high, patch_low],
              fontsize=12, loc=4, frameon=False)

    # Annotate baseline value on x-axis
    ax.annotate(f'Baseline\n{baseline_miu:.4f}',
                xy=(baseline_miu, -0.8),
                xytext=(baseline_miu, -1.4),
                fontsize=7.5, ha='center', color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.set_xlim(0.2, 1.4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Running sensitivity analysis ...")
    print(f"Baseline conditions: {BASELINE_CONDITIONS}")
    print(f"Baseline µ = {evaluate_miu():.6f} day-1\n")

    results_25, baseline_25 = compute_sensitivity([ 0.25, -0.25])
    results_50, baseline_50 = compute_sensitivity([ 0.50, -0.50])

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=600)

    tornado_plot(results_25, baseline_25,
                 title=r"(a)  $\pm$25% Parameter Perturbation",
                 pct_label=0.25, ax=axes[0], index = 'a')

    tornado_plot(results_50, baseline_50,
                 title=r"(b)  $\pm$50% Parameter Perturbation",
                 pct_label=0.50, ax=axes[1], index = 'b')

    plt.tight_layout()
