# AGENTE.md — AI Computing Project 2026.2 (UNIFOR)

**Course:** Computational Artificial Intelligence
**Professor:** Dra. Cynthia Moreira Maia
**Due date:** 09/09/2026
**Team:** 3 members

## General rules (apply to EVERYONE)
- ❌ `sklearn` and `pandas` are forbidden. Regression, R², adjusted R², RMSE, MAE and residuals must be implemented manually.
- ✅ Allowed: `numpy` (linear algebra), `matplotlib` (plots), `csv` or manual file parsing (to read the `.csv` without pandas).
- ❌ The work will be checked by plagiarism/AI detectors. Write explanations and the report in your own words.
- ✅ Submitting the implementation is mandatory. Punctuality counts toward the grade.
- Datasets: `arseniodataset.csv` (Problem 1) and `dose_radiacao_expandida.csv` (Problem 2), both on AVA.

---

# PART 1 — Member A: Problem 1 (Arsenic in toenails)

## Technical step-by-step

**Step 1 — Load the data**
- Read `arseniodataset.csv` without pandas (use `csv.reader` or `numpy.genfromtxt`).
- Extract columns: age, sex, water_drinking_use, water_cooking_use, arsenic_water, arsenic_toenails (response).
- Note: sex is not used as a regressor in this model (not requested in the assignment).

**Step 2 — Build the regressor matrix X**
- Regressors: age, water_drinking_use, water_cooking_use, arsenic_water → matrix X of shape (n, 4).
- Add a column of 1s to the left of X (for the intercept) → X shape (n, 5).
- y = arsenic_toenails vector, shape (n, 1).

**Step 3 — Compute coefficients via Ordinary Least Squares (OLS)**
- Formula: β = (XᵀX)⁻¹ Xᵀ y
- Implement with `numpy.linalg.inv` or, preferably, `numpy.linalg.solve` (more numerically stable).
- β[0] = intercept; β[1:] = coefficients for age, water_drinking_use, water_cooking_use, arsenic_water.

**Step 4 — Make the requested prediction**
- Input vector: [1, age=30, water_drinking_use=5, water_cooking_use=5, arsenic_water=0.135]
- ŷ = x_new · β
- Report the predicted arsenic value in toenails.

**Step 5 — Compute R²**
- ŷᵢ = X · β for all observations.
- SSR (sum of squared residuals) = Σ(yᵢ − ŷᵢ)²
- SST (total sum of squares) = Σ(yᵢ − ȳ)²
- R² = 1 − SSR/SST

**Step 6 — Compute adjusted R²**
- R²_adj = 1 − (1 − R²) × (n − 1)/(n − p − 1), where p = 4 (number of regressors, not counting the intercept).
- Write an explanation: R² always increases (or stays the same) when adding variables, even irrelevant ones; adjusted R² penalizes the inclusion of variables that don't improve the model proportionally, which is why it's preferred when comparing models with different numbers of predictors.

**Step 7 — Alternative model (arsenic in water only)**
- Repeat Steps 2–6 using only the arsenic_water column as regressor (X shape (n,2) with intercept).
- Compare R², adjusted R², RMSE and MAE between the two models.
- Conclude which model is better and why (does the R² gain justify the extra complexity? overfitting risk?).

**Step 8 — Model with intercept forced to zero**
- Without the column of 1s: X_no_intercept shape (n, 4).
- β = (XᵀX)⁻¹ Xᵀ y (same formula, no intercept).
- Compute R² (careful: with intercept = 0, the usual R² formula can be "inflated" — explain this in the report) and RMSE.
- Compare with the original model and justify which one to choose.

**Step 9 — Additional metrics**
- MSE = (1/n) Σ(yᵢ − ŷᵢ)²
- RMSE = √MSE
- MAE = (1/n) Σ|yᵢ − ŷᵢ|
- Compute for the full model and the alternative model (Step 7). Interpret them (same unit as the response variable, lower is better).

## Deliverable for this part
- Commented Python script (`problem1_model.py`) covering Steps 1–9.
- A summary file (`results_problem1.md` or `.txt`) with: coefficients, prediction, R², adjusted R², model comparison, metrics — ready for Member C to use in the report.

---

# PART 2 — Member B: Problem 1 Residuals + Problem 2 complete

## Block 2.1 — Residual analysis (Problem 1)
*(uses the full model already fitted by Member A — ask them for the β coefficients, or refit using the same code)*

**Step 1** — Compute ŷᵢ = X · β for all n observations.
**Step 2** — Compute eᵢ = yᵢ − ŷᵢ for all observations.
**Step 3** — Build a table with columns: Observation i | yᵢ | ŷᵢ | eᵢ (all rows of the dataset).
**Step 4** — Check model assumptions using plots (`matplotlib`):
  - Residuals (eᵢ) vs. fitted values (ŷᵢ) plot: should look scattered with no pattern (checks linearity/homoscedasticity).
  - Histogram of residuals: should look roughly symmetric/normal.
  - (Optional) QQ-plot of residuals vs. normal distribution.
  - Write 3–5 lines concluding whether the model's assumptions (linearity, homoscedasticity, normality) appear to hold, based on the plots.

## Block 2.2 — Problem 2 (Radiation dose)

**Step 1 — Load the data**
- Read `dose_radiacao_expandida.csv` without pandas.
- Columns: current (mAmp), exposure_time (min), radiation_dose (response).

**Step 2 — Build X and y**
- X = [current, exposure_time], add a column of 1s → shape (n, 3).
- y = radiation_dose.

**Step 3 — Fit the model (OLS)**
- β = (XᵀX)⁻¹ Xᵀ y (same formula as in Part 1).

**Step 4 — Requested prediction**
- x_new = [1, current=15, exposure_time=5]
- ŷ = x_new · β → report the predicted radiation dose.

**Step 5 — Compute R²** (same formula as Part 1, Step 5).

**Step 6 — Compute and explain adjusted R²** (p = 2 regressors here). Same theoretical explanation as Part 1, adapted.

**Step 7 — Alternative model (current only)**
- Repeat Steps 2–6 with X = [1, current] only.
- Compare R², RMSE, MAE with the full model. Conclude which is better.

**Step 8 — Intercept forced to zero**
- X without the column of 1s (shape (n,2): current, exposure_time).
- Recompute β, R², RMSE. Compare with the model with intercept and justify the choice.

**Step 9 — Additional metrics**
- MSE, RMSE, MAE for the full model and the alternative model (same formulas as Part 1, Step 9).

## Deliverable for this part
- Python script (`problem1_residuals.py`): residuals table + diagnostic plots.
- Python script (`problem2_model.py`): covering Block 2.2 in full.
- Summary (`results_problem2.md`) with coefficients, prediction, R², comparisons and metrics — ready for the report.

---

# PART 3 — Member C: Final Report

## Step-by-step for writing

**Step 1 — Gather inputs**
- Ask Member A for `results_problem1.md` (coefficients, prediction, R², comparisons, metrics).
- Ask Member B for the residuals table + plots + `results_problem2.md`.
- Do not proceed without this real data — never invent numbers.

**Step 2 — Write the Introduction** (short, 1 paragraph)
- Topic: using linear regression to predict arsenic exposure and radiation dose.
- Objective: fit, evaluate and compare multiple regression models.

**Step 3 — Write the Theoretical Framework**
- Define simple linear regression (1 variable) vs. multiple (several variables).
- Explain coefficient and intercept interpretation ("holding the other variables constant...").
- Explain R², adjusted R², RMSE, MAE in plain language, with 1 numeric example using real data from Problem 1.

**Step 4 — Write the Experimental Protocol**
- Data source (cite both datasets and the original articles mentioned in the assignment).
- Variables used in each problem.
- Tools (Python + NumPy, manual implementation without sklearn/pandas).
- Fitting steps (OLS via normal equations, residual checking, comparison with the alternative model and with the zero-intercept model).

**Step 5 — Write Results and Discussion** (organize by problem)
- For each problem: coefficient table, requested prediction, R²/adjusted R², comparison with the alternative model, discussion of the zero-intercept scenario, extra metrics (MSE/RMSE/MAE), residuals table and plots (Problem 1).
- Comment on each result (don't just report numbers — explain what they mean).

**Step 6 — Write the Conclusion**
- Summarize the main findings from both problems.
- Comment on practical usefulness (e.g., predicting arsenic exposure from well-water data).
- Suggest next steps (more data, other variables, non-linear models, etc.).

**Step 7 — Final review**
- Check that all numbers match the files sent by A and B.
- Run the text through a plagiarism/AI checker (or rewrite overly "generic"-sounding passages in your own words).
- Format as the final document (Word/PDF), double-check the deadline (09/09/2026).

## Deliverable for this part
- Complete, formatted final report, correctly citing the results from both problems.

---

## Team integration flow
1. A and B work in parallel, following their technical steps.
2. A and B deliver their summary files + scripts to C by an intermediate date agreed on by the team.
3. C assembles the report following Steps 1–7 of Part 3.
4. A and B review whether their results were correctly cited before final submission.