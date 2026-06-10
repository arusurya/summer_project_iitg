=== Data Cleaning Log ===

[LOAD] Rows: 392,936 | Members: 16,737

[DECISION 1] Duplicate rows (same member + year + month): 3,871 removed.
  Rationale: Activity grain must be unique per member-month. Keep first occurrence as conservative choice.

[DECISION 2] Members active before enrollment date: 247 removed.
  Members remaining: 16,490 (was 16,737)
  Rationale: Pre-enrollment activity is impossible — likely data entry errors or system migration artefacts. Excluding prevents target leakage in the model.

[DECISION 3] total_flights: no negative values found — no action.

[DECISION 3] distance: no negative values found — no action.

[DECISION 3] points_accumulated: no negative values found — no action.

[DECISION 3] points_redeemed: no negative values found — no action.

[DECISION 3b] CLV issues:
  Null CLV  : 0
  Zero CLV  : 0
  Neg  CLV  : 0
  Action: nulls filled with median (5,780), negatives clipped to 0.
  Rationale: Median imputation preserves distribution shape for a skewed metric.

[DECISION 4] total_flights: 229 rows capped at 99.9th pct (20.0).

[DECISION 4] distance: 384 rows capped at 99.9th pct (32,871.8).

[DECISION 4] points_accumulated: 384 rows capped at 99.9th pct (49,307.8).

[DECISION 4] points_redeemed: 374 rows capped at 99.9th pct (804.0).
  Rationale: Hard outliers distort distance-based features and model gradients. Capping at 99.9th pct retains the signal (high-value flyers) while removing likely data errors (e.g. single month with 10,000 flights).

[CHURN DEFINITION]
  Active-in-lookback  : ≥1 flight in 12m before 2018-06
  Churned-in-forward  : 0 flights + 0 redemptions in 6m after cutoff
  Churn rate          : 4.7% (628 of 13,475 members)

[CHURN COMPARISON]
  Formally cancelled              : 862
  Silent churners (missed by formal): 72
  Behavioural churn captures 11% more at-risk members.

[DECISION 5] salary: nulls filled with median (73544.00).