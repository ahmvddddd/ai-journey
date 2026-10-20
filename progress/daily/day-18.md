# Day 18 — Handling Missing Values in Pandas

## Status

* **Completion:** Completed
* **Topic:** Detection (`isna()`), Type Upcasting (`np.nan` vs `Int64`), Drop Strategies (`dropna()`), Statistical/Constant Imputation (`fillna()`), and Sequential Time-Series Fill (`ffill()`)
* **Time spent:** 2.5 hours

## Objectives

* [x] Distinguish between missing data representations (`np.nan`, `None`, and nullable `pd.NA`)[cite: 15]
* [x] Detect and quantify missing value patterns using `df.isna().sum()`[cite: 15]
* [x] Master conditional dropping strategies (`dropna(subset=...)`, `how='any'` vs `how='all'`)[cite: 15]
* [x] Execute statistical, constant, and sequential forward-fill (`ffill()`) imputations[cite: 15]
* [x] Analyze type upcasting penalties (`int64` → `float64`) and leverage modern nullable extension types (`Int64`)[cite: 15]

## Work Completed

Successfully explored detection, type behavior, dropping, and multi-strategy imputation for missing data:
* **Missing Value Detection:** Mapped missing patterns across sensor data using `df.isna().sum()`[cite: 15].
* **Selective Row Deletion:** Applied `df.dropna(subset=['temperature'])` to drop records based on critical feature requirements rather than discarding rows indiscriminately[cite: 15].
* **Imputation Pipelines:** Applied column-wise mean imputation for numerical features, constant string replacement (`"UNKNOWN"`) for categories, and sequential forward-filling (`ffill()`) for ordered pressure sensor telemetry[cite: 15].
* **Type Safety Analysis:** Evaluated how `np.nan` forces float upcasting on standard NumPy integer columns and demonstrated how Pandas nullable `Int64` preserves whole-number representations alongside `pd.NA`[cite: 15].

## Evidence

* Interactive notebook [`day-18-handling-missing-values.ipynb`](../../00-foundations/python/day-18-handling-missing-values.ipynb) executed successfully in workspace environment[cite: 15].

## Lessons Learned

* **Float Upcasting Hazard:** Standard `np.nan` is a float, which forces integer columns from `int64` to `float64`[cite: 15]. Using Pandas' nullable `Int64` (capital I) uses a separate boolean mask to hold `pd.NA` without converting whole numbers to floats[cite: 15].
* **Time-Series Imputation Safety:** Global statistical means ignore temporal trends and introduce look-ahead bias[cite: 15]. Sequential fills (`ffill()`) or linear interpolation (`interpolate()`) preserve chronological flow[cite: 15].
* **`dropna(how='all')` Behavior:** `how='all'` drops rows only if *every* column is NaN[cite: 15]. If each row contains at least one valid cell, `how='all'` removes zero rows regardless of total missing entries[cite: 15].

## Next Session

**Day 19:** Grouping, Aggregation & Split-Apply-Combine Workflows in Pandas.

## Suggested Commit

`feat: complete day 18 handling missing values and imputation strategies`