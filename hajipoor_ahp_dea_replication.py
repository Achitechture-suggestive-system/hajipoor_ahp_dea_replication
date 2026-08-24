
"""
Reproduction / audit pipeline for:

Hajipoor, Motameni, Ebrahimnejad (2025)
"A hybrid AHP-DEA approach for software architecture evaluation and selection"

What this script DOES
---------------------
1) Stores Table 3 exactly as published (RIM(normal), rounded to 3 decimals).
2) Recomputes AHP quality-attribute weights as row means and compares them
   with the published W(q_i).
3) Stores Table 4 exactly as published: 23 architecture styles x 7 attributes.
4) Because the raw 10 expert RSM pairwise matrices are NOT published,
   reconstructs the RSM(total) that is implied by Table 4 using Eq. (6):

       CM[a,q] = W[q] * RSM_total[a,q]

   Therefore:

       RSM_total[a,q] = CM[a,q] / W[q]

   IMPORTANT: this is reverse reconstruction from published aggregate data,
   not recovery of the original individual questionnaires.
5) Reconstructs CM from W and inferred RSM(total) and verifies it against Table 4.
6) Splits the 7 attributes exactly as described in Phase Two:
       INPUTS  = Cost, Development Effort
       OUTPUTS = Security, Performance, Availability, Modifiability, Usability
7) Solves the paper's Eq. (9) in envelopment/slack form for each of the 23 DMUs.
8) Compares the independently recomputed theta values with Table 5.
9) Exports all intermediate matrices and diagnostic tables.

Important reproducibility finding
---------------------------------
Using the published Table 4 data and the printed Eq. (9), standard CCR
input-oriented DEA does NOT reproduce all Table 5 scores. The script reports
those differences rather than silently forcing the paper values.

Requirements
------------
pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# ============================================================
# STAGE 0 — PAPER CONSTANTS
# ============================================================

# Table 3: RIM(normal)
# NOTE: the paper's displayed row order is not identical to its header order.
RIM_COLUMNS = [
    "Security", "Cost", "Performance",
    "Availability", "Modifiability", "Usability", "Dev. Effort"
]

RIM_ROWS = [
    "Security",
    "Cost",
    "Performance",
    "Dev. Effort",
    "Availability",
    "Modifiability",
    "Usability",
]

RIM_NORMAL_VALUES = [
    [0.342, 0.260, 0.380, 0.422, 0.251, 0.458, 0.194],
    [0.053, 0.040, 0.022, 0.035, 0.025, 0.037, 0.121],
    [0.077, 0.157, 0.069, 0.069, 0.141, 0.058, 0.136],
    [0.143, 0.027, 0.052, 0.070, 0.059, 0.085, 0.081],
    [0.106, 0.150, 0.162, 0.130, 0.192, 0.110, 0.152],
    [0.154, 0.184, 0.069, 0.077, 0.113, 0.086, 0.156],
    [0.124, 0.183, 0.248, 0.197, 0.219, 0.167, 0.159],
]

# Last column W(q_i) in Table 3, exactly as displayed.
PUBLISHED_WEIGHTS = pd.Series({
    "Security":       0.330,
    "Cost":           0.047,
    "Performance":    0.101,
    "Dev. Effort":    0.074,
    "Availability":   0.143,
    "Modifiability":  0.120,
    "Usability":      0.185,
}, name="Published W(q_i)")

# Table 4: CM matrix, 23 architecture alternatives x 7 attributes.
ARCHITECTURES = [
    "Broker",
    "SOA",
    "Pipe filter",
    "Peer-to-peer",
    "Implicit-invocation",
    "Space-based",
    "Component-based",
    "Microservice",
    "RPC",
    "Microkernel",
    "Client-server",
    "Shared repository",
    "Reflection",
    "PAC",
    "Publish-subscribe",
    "Virtual-machine",
    "Batch-sequence",
    "Blackboard",
    "MVC",
    "Object oriented",
    "Grid computing",
    "Event-based",
    "Layer",
]

CM_COLUMNS = [
    "Security",
    "Performance",
    "Cost",
    "Availability",
    "Modifiability",
    "Usability",
    "Dev. Effort",
]

CM_VALUES = [
    [0.0738, 0.0556, 0.0792, 0.0387, 0.0349, 0.0263, 0.0551],
    [0.1286, 0.0839, 0.0329, 0.0736, 0.0711, 0.0298, 0.0275],
    [0.0281, 0.0503, 0.0416, 0.0282, 0.0601, 0.0354, 0.0446],
    [0.0877, 0.0155, 0.0558, 0.0579, 0.0515, 0.0356, 0.0393],
    [0.0461, 0.0178, 0.0644, 0.0347, 0.0324, 0.0372, 0.0516],
    [0.0703, 0.0393, 0.0467, 0.0214, 0.0169, 0.0421, 0.0471],
    [0.1415, 0.0692, 0.0269, 0.0528, 0.0917, 0.0733, 0.0176],
    [0.1885, 0.0935, 0.0215, 0.0693, 0.1082, 0.0599, 0.0088],
    [0.0374, 0.0231, 0.0671, 0.0409, 0.0483, 0.0313, 0.0642],
    [0.0374, 0.0286, 0.0613, 0.0449, 0.0252, 0.0292, 0.0435],
    [0.0493, 0.0976, 0.0596, 0.0706, 0.0624, 0.0347, 0.0842],
    [0.0296, 0.0104, 0.0343, 0.0561, 0.0471, 0.0438, 0.0331],
    [0.0237, 0.0187, 0.0505, 0.0223, 0.0141, 0.0255, 0.0362],
    [0.0259, 0.0122, 0.0591, 0.0316, 0.0218, 0.0436, 0.0898],
    [0.0471, 0.0418, 0.0404, 0.0465, 0.0233, 0.0199, 0.0515],
    [0.0406, 0.0058, 0.0436, 0.0063, 0.0184, 0.0231, 0.0438],
    [0.0174, 0.0137, 0.0368, 0.0087, 0.0196, 0.0157, 0.0466],
    [0.0252, 0.0272, 0.0754, 0.0074, 0.0513, 0.0318, 0.1015],
    [0.1161, 0.0758, 0.0287, 0.0751, 0.0829, 0.0676, 0.0212],
    [0.1627, 0.0872, 0.0183, 0.0794, 0.0937, 0.0663, 0.0110],
    [0.0221, 0.0591, 0.0313, 0.0428, 0.0302, 0.0296, 0.0547],
    [0.0298, 0.0426, 0.0246, 0.0419, 0.0274, 0.0286, 0.0597],
    [0.1402, 0.0622, 0.0297, 0.0173, 0.0685, 0.0497, 0.0331],
]

# Table 5 exactly as published.
PUBLISHED_TABLE5 = pd.Series({
    "Broker":              0.444,
    "SOA":                 0.530,
    "Pipe filter":         0.341,
    "Peer-to-peer":        0.545,
    "Implicit-invocation": 0.286,
    "Space-based":         0.512,
    "Component-based":     0.823,
    "Microservice":        1.000,
    "RPC":                 0.342,
    "Microkernel":         0.453,
    "Client-server":       0.340,
    "Shared repository":   0.691,
    "Reflection":          0.159,
    "PAC":                 0.258,
    "Publish-subscribe":   0.519,
    "Virtual-machine":     0.174,
    "Batch-sequence":      0.098,
    "Blackboard":          0.162,
    "MVC":                 0.768,
    "Object oriented":     0.894,
    "Grid computing":      0.400,
    "Event-based":         0.455,
    "Layer":               0.713,
}, name="Paper Table 5 theta")


# ============================================================
# STAGE 1 — TABLE 3: AHP RIM(normal) -> ATTRIBUTE WEIGHTS
# ============================================================

rim = pd.DataFrame(
    RIM_NORMAL_VALUES,
    index=RIM_ROWS,
    columns=RIM_COLUMNS
)

calculated_weights = rim.mean(axis=1).rename("Row mean from displayed Table 3")

weight_check = pd.concat(
    [calculated_weights, PUBLISHED_WEIGHTS],
    axis=1
)
weight_check["difference_due_to_rounding"] = (
    weight_check["Row mean from displayed Table 3"]
    - weight_check["Published W(q_i)"]
)

column_sum_check = rim.sum(axis=0).rename("column_sum")


# ============================================================
# STAGE 2 — TABLE 4 AND IMPLIED RSM(total)
# ============================================================

cm = pd.DataFrame(
    CM_VALUES,
    index=ARCHITECTURES,
    columns=CM_COLUMNS
)

# Eq. (6):
#   CM[a,q] = W[q] * s[a,q]
#
# Raw RSM expert matrices are not published.
# We therefore infer the aggregate RSM(total) that is compatible
# with the published CM and W:
#
#   s[a,q] = CM[a,q] / W[q]

weights_in_cm_order = PUBLISHED_WEIGHTS.reindex(CM_COLUMNS)

rsm_total_inferred = cm.div(weights_in_cm_order, axis="columns")

# Rebuild CM via Eq. (6) as a numerical check.
cm_reconstructed = rsm_total_inferred.mul(weights_in_cm_order, axis="columns")
cm_reconstruction_error = (cm_reconstructed - cm).abs()


# ============================================================
# STAGE 3 — OPTIONAL AHP-ONLY AGGREGATE SCORE
# ============================================================

# Since each CM cell already equals W_q * support(a,q),
# row sum is the weighted-support aggregate across the 7 attributes.
ahp_aggregate = cm.sum(axis=1).rename("AHP weighted-support sum")
ahp_rank = ahp_aggregate.rank(method="min", ascending=False).astype(int).rename("AHP rank")


# ============================================================
# STAGE 4 — SPLIT TABLE 4 INTO DEA INPUTS / OUTPUTS
# ============================================================

INPUT_COLUMNS = ["Cost", "Dev. Effort"]
OUTPUT_COLUMNS = [
    "Security",
    "Performance",
    "Availability",
    "Modifiability",
    "Usability",
]

X = cm[INPUT_COLUMNS].to_numpy(dtype=float)
Y = cm[OUTPUT_COLUMNS].to_numpy(dtype=float)


# ============================================================
# STAGE 5 — SOLVE EQ. (9) EXACTLY IN SLACK FORM
# ============================================================

def solve_eq9_ccr_input(X, Y, names, epsilon=1e-6):
    """
    Paper Eq. (9), written as an LP.

    For target DMU o:

      min theta - epsilon * (sum s^- + sum s^+)

      Y^T lambda - s^+ = y_o
     -X^T lambda - s^- + theta*x_o = 0

      lambda, s^+, s^- >= 0

    Variables:
      lambda_1 ... lambda_n
      theta
      splus_1 ... splus_s
      sminus_1 ... sminus_m

    theta is the radial input-efficiency score.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n_dmu, n_input = X.shape
    _, n_output = Y.shape

    result_rows = []
    lambda_rows = []
    splus_rows = []
    sminus_rows = []

    for o in range(n_dmu):

        theta_idx = n_dmu
        splus_start = n_dmu + 1
        sminus_start = n_dmu + 1 + n_output
        n_vars = n_dmu + 1 + n_output + n_input

        c = np.zeros(n_vars)
        c[theta_idx] = 1.0
        c[splus_start:splus_start+n_output] = -epsilon
        c[sminus_start:sminus_start+n_input] = -epsilon

        A_eq = []
        b_eq = []

        # Output equations:
        # sum_j y_rj*lambda_j - splus_r = y_ro
        for r in range(n_output):
            row = np.zeros(n_vars)
            row[:n_dmu] = Y[:, r]
            row[splus_start + r] = -1.0
            A_eq.append(row)
            b_eq.append(Y[o, r])

        # Input equations:
        # -sum_j x_ij*lambda_j - sminus_i + theta*x_io = 0
        for i in range(n_input):
            row = np.zeros(n_vars)
            row[:n_dmu] = -X[:, i]
            row[theta_idx] = X[o, i]
            row[sminus_start + i] = -1.0
            A_eq.append(row)
            b_eq.append(0.0)

        bounds = [(0, None)] * n_vars

        lp = linprog(
            c,
            A_eq=np.asarray(A_eq),
            b_eq=np.asarray(b_eq),
            bounds=bounds,
            method="highs",
        )

        if not lp.success:
            raise RuntimeError(
                f"DEA LP failed for {names[o]}: {lp.message}"
            )

        theta = float(lp.x[theta_idx])
        lambdas = lp.x[:n_dmu]
        splus = lp.x[splus_start:splus_start+n_output]
        sminus = lp.x[sminus_start:sminus_start+n_input]

        active_peers = [
            f"{names[j]}:{lambdas[j]:.6f}"
            for j in range(n_dmu)
            if lambdas[j] > 1e-8
        ]

        result_rows.append({
            "Architecture": names[o],
            "theta_recomputed": theta,
            "phi_reciprocal": 1.0/theta if theta > 0 else np.inf,
            "radial_input_reduction_%": (1.0-theta)*100.0,
            "active_reference_peers": ", ".join(active_peers),
        })

        lambda_rows.append(lambdas)
        splus_rows.append(splus)
        sminus_rows.append(sminus)

    result_df = pd.DataFrame(result_rows).set_index("Architecture")
    lambda_df = pd.DataFrame(lambda_rows, index=names, columns=names)
    splus_df = pd.DataFrame(splus_rows, index=names, columns=OUTPUT_COLUMNS)
    sminus_df = pd.DataFrame(sminus_rows, index=names, columns=INPUT_COLUMNS)

    return result_df, lambda_df, splus_df, sminus_df


dea, lambda_matrix, output_slacks, input_slacks = solve_eq9_ccr_input(
    X, Y, ARCHITECTURES
)


# ============================================================
# STAGE 6 — COMPARE RECOMPUTED EQ. (9) WITH PAPER TABLE 5
# ============================================================

comparison = dea.copy()
comparison["paper_table5_theta"] = PUBLISHED_TABLE5
comparison["difference_recomputed_minus_paper"] = (
    comparison["theta_recomputed"]
    - comparison["paper_table5_theta"]
)
comparison["absolute_difference"] = (
    comparison["difference_recomputed_minus_paper"].abs()
)
comparison["matches_paper_to_3dp"] = (
    comparison["theta_recomputed"].round(3)
    == comparison["paper_table5_theta"].round(3)
)

comparison["paper_rank"] = (
    comparison["paper_table5_theta"]
    .rank(method="min", ascending=False)
    .astype(int)
)
comparison["recomputed_rank"] = (
    comparison["theta_recomputed"]
    .rank(method="min", ascending=False)
    .astype(int)
)


# ============================================================
# STAGE 7 — MICROservice DIAGNOSTIC
# ============================================================

def architecture_diagnostic(name):
    x = cm.loc[name, INPUT_COLUMNS]
    y = cm.loc[name, OUTPUT_COLUMNS]
    theta = comparison.loc[name, "theta_recomputed"]
    lam = lambda_matrix.loc[name]
    active = lam[lam > 1e-8]
    radial_target = theta * x

    return {
        "architecture": name,
        "inputs": x.to_dict(),
        "outputs": y.to_dict(),
        "theta_recomputed": float(theta),
        "phi_reciprocal": float(1/theta),
        "radial_input_reduction_percent": float((1-theta)*100),
        "radial_input_target_theta_times_x": radial_target.to_dict(),
        "input_slacks": input_slacks.loc[name].to_dict(),
        "output_slacks": output_slacks.loc[name].to_dict(),
        "active_lambdas": active.to_dict(),
        "paper_table5_theta": float(PUBLISHED_TABLE5.loc[name]),
    }


# ============================================================
# STAGE 8 — EXPORT EVERYTHING
# ============================================================

rim.to_csv(OUT / "01_table3_rim_normal.csv")
weight_check.to_csv(OUT / "02_weight_check.csv")
column_sum_check.to_csv(OUT / "03_rim_column_sum_check.csv")

rsm_total_inferred.to_csv(OUT / "04_rsm_total_inferred_from_table4.csv")
cm.to_csv(OUT / "05_table4_cm_published.csv")
cm_reconstructed.to_csv(OUT / "06_cm_reconstructed_via_eq6.csv")
cm_reconstruction_error.to_csv(OUT / "07_cm_reconstruction_error.csv")

pd.concat([ahp_aggregate, ahp_rank], axis=1).to_csv(
    OUT / "08_ahp_weighted_support_and_rank.csv"
)

comparison.to_csv(OUT / "09_dea_eq9_vs_table5.csv")
lambda_matrix.to_csv(OUT / "10_lambda_matrix.csv")
output_slacks.to_csv(OUT / "11_output_slacks.csv")
input_slacks.to_csv(OUT / "12_input_slacks.csv")

for name in ["Microservice", "Component-based", "Object oriented", "MVC", "Layer"]:
    diagnostic = architecture_diagnostic(name)
    pd.Series(diagnostic, dtype=object).to_json(
        OUT / f"diagnostic_{name.replace(' ', '_').lower()}.json",
        indent=2
    )


# ============================================================
# STAGE 9 — PRINT A HUMAN-READABLE PIPELINE
# ============================================================

print("="*80)
print("HAJIPOOR ET AL. (2025) — AHP -> DEA REPLICATION / AUDIT")
print("="*80)

print("\n[STAGE 1] Table 3: RIM(normal)")
print(rim.to_string())
print("\nColumn sums should be ~1 because the displayed values are rounded:")
print(column_sum_check.round(6).to_string())

print("\nAHP row-mean weight check:")
print(weight_check.round(6).to_string())

print("\n[STAGE 2] Table 4 CM shape:")
print(cm.shape)
print("Expected: (23, 7)")

print("\n[STAGE 3] Eq. (6) reconstruction check")
print("Max absolute CM reconstruction error:",
      cm_reconstruction_error.to_numpy().max())

print("\n[STAGE 4] DEA matrices")
print("X inputs shape :", X.shape, INPUT_COLUMNS)
print("Y outputs shape:", Y.shape, OUTPUT_COLUMNS)

print("\n[STAGE 5/6] Eq. (9) recomputation vs paper Table 5")
display_cols = [
    "theta_recomputed",
    "paper_table5_theta",
    "difference_recomputed_minus_paper",
    "matches_paper_to_3dp",
    "active_reference_peers",
]
print(comparison[display_cols].round(6).to_string())

print("\nExact/3dp matches:",
      int(comparison["matches_paper_to_3dp"].sum()),
      "of", len(comparison))

print("\n[STAGE 7] Key diagnostics")
for name in ["Microservice", "Component-based", "Object oriented", "MVC", "Layer"]:
    d = architecture_diagnostic(name)
    print("\n---", name, "---")
    print("paper theta      =", d["paper_table5_theta"])
    print("recomputed theta =", round(d["theta_recomputed"], 6))
    print("phi=1/theta      =", round(d["phi_reciprocal"], 6))
    print("active lambdas   =", d["active_lambdas"])
    print("input slacks     =", d["input_slacks"])

print("\nOutputs written to:", OUT)
print("\nNOTE:")
print(
    "The raw expert pairwise matrices are not published, so this script does not "
    "invent them. It starts from the published aggregate Table 3 and Table 4. "
    "The independently solved Eq. (9) does not reproduce all Table 5 values; "
    "see 09_dea_eq9_vs_table5.csv."
)
