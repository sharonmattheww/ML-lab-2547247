# Mission Earth — Forecasting Next-Day Wildfire Risk from a Cheap Weather Station

**CIA-3 · Machine Learning (MCA 521-4) · ML for Social Good Ensemble Challenge**
---
**Sharon Mathew 2547247 4MCA-B**
---
Deliverable: `Mission_Earth_Wildfire_Risk_CIA3.ipynb`

---

## 1. What this project is

Rural forest districts get late fire warnings because the official early-warning system — the Canadian
**Fire Weather Index (FWI)** — needs a calibrated station and a trained analyst to compute six chained
fuel-moisture codes (FFMC, DMC, DC, ISI, BUI, FWI) every day.

Almost every published model on this dataset feeds those six codes into a classifier and reports
97–99% accuracy. That is close to predicting the alarm from the alarm.

**This project deliberately deletes all six FWI codes** and asks a harder, more useful question:

> Can an ensemble learn fire-day risk from only what a cheap weather station reports —
> temperature, humidity, wind, rainfall — plus the memory of the last few days?

| | |
|---|---|
| **Mission** | Earth (environment) |
| **Beneficiaries** | Rural fire brigades and forest guards in districts with no FWI capability |
| **Unit of analysis** | One region-day |
| **Target** | `fire` = 1 if that region-day was a recorded fire day |
| **Features used** | Temperature, RH, wind speed, rain + engineered lag/dryness features |
| **Features banned** | FFMC, DMC, DC, ISI, BUI, FWI |
| **Impact metric** | Recall on fire days at a realistic patrol budget |

---

## 2. Dataset (real, public, cited — nothing synthetic in training)

**Algerian Forest Fires Dataset** — Abid, F. & Izeboudjen, N. (2019), UCI Machine Learning Repository
(dataset id 547).

* 244 daily records, 1 June – 30 September 2012
* 2 regions: Bejaia (north-east Algeria), Sidi-Bel Abbes (north-west Algeria)
* Labels: `fire` (137) / `not fire` (106)
* Original page: <https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset>
* The notebook downloads a **byte-identical public GitHub mirror** so it runs with one click:
  `https://raw.githubusercontent.com/krishnaik06/Complete-Machine-Learning-2023/main/Algerian_forest_fires_dataset_UPDATE.csv`

No personal, identifiable or confidential data — every row is a public meteorological record.

---

## 3. How to run it


### Local machine (Jupyter)

```bash
# Python 3.9+ recommended
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib jupyter

jupyter notebook Mission_Earth_Wildfire_Risk_CIA3.ipynb
# then: Kernel ▸ Restart & Run All
```

### Headless re-execution (for verification)

```bash
pip install nbconvert ipykernel
jupyter nbconvert --to notebook --execute --inplace Mission_Earth_Wildfire_Risk_CIA3.ipynb
```

**Offline use:** download the CSV from the mirror URL above, save it next to the notebook as
`Algerian_forest_fires_dataset_UPDATE.csv`, and the notebook will use the cached copy instead of
downloading.

### Files produced when you run it

| File | What it is |
|---|---|
| `Algerian_forest_fires_dataset_UPDATE.csv` | cached raw dataset (auto-downloaded once) |
| `wildfire_risk_pipeline.joblib` | the trained best pipeline + feature list + alert threshold |

### Reproducibility

`RANDOM_STATE = 42` is fixed for every split, search and model. Re-running gives the same numbers on
the same library versions (verified with scikit-learn 1.8, xgboost 3.2, shap 0.51).

---

## 4. Notebook structure

| Section | Question | Marks | Contents |
|---|---|---|---|
| 1 | Q1 — Real-world impact framing | 5 | problem, beneficiaries, target, unit of analysis, why ML, dataset source, responsible-use limits |
| 2 | Q2 — Data wrangling & feature engineering | 6 | structural repair of the messy UCI file, audit, validity checks, outlier decision, 4 EDA blocks, 9 engineered features, temporal split |
| 3 | Q3 — Ensemble architecture, tuning, comparison | 8 | LR + Decision Tree baselines, tuned Random Forest, tuned XGBoost, Soft Voting, Stacking; leakage-safe CV; ROC/PR curves; confusion matrices; threshold tuning |
| 4 | Q4 — Explainability & ethics | 4 | SHAP global + local (2 real days), permutation importance, fairness audit across districts, privacy, uncertainty, FP/FN cost asymmetry, oversight limits |
| 5 | Q5 — Pitch & live demo | 2 | saved pipeline reloaded, live prediction on one realistic synthetic record, its SHAP explanation, compact results visual |

---

## 5. Key methodological decisions 

1. **The six FWI codes are dropped.** They are the expert warning system's own output — keeping them
   inflates the score for a model an under-resourced district could not run. The notebook shows their
   correlation with the target to justify the exclusion.
2. **Temporal split, not random.** Train = June–August, test = **September, untouched**. Weather is
   autocorrelated, so a random split would leak neighbouring days and flatter the model.
3. **The broken row is repaired, not imputed.** Line 170 of the raw file has `14.6 9` — a missing comma,
   not a missing value. Restoring the comma uses digits already in the file; imputing would have
   invented data.
4. **Outliers are kept.** The IQR flags are heavy-rain days — the most informative *low-risk* evidence
   in a fire problem. Removing them would delete signal.
5. **No SMOTE.** Class balance is ≈ 57/43 (mild). On 244 rows, synthetic oversampling would fabricate
   region-days; `class_weight='balanced'` / `scale_pos_weight` reweights the loss instead.
6. **All preprocessing lives inside the `Pipeline`.** Imputer, scaler and one-hot encoder are re-fitted
   inside every CV fold — never on the test set.
7. **Stacking uses out-of-fold predictions.** `StackingClassifier(cv=StratifiedKFold(5))` so the
   meta-learner never sees in-sample base predictions.
8. **The threshold is tuned on training out-of-fold scores**, then applied once to the test set —
   because a missed fire costs vastly more than a wasted patrol.

---

## 6. Results actually produced by this notebook

All numbers below come from executing the notebook end-to-end; the untouched test set is
September 2012 (60 region-days, 23 fire days).

| Model | CV ROC-AUC (train) | Test ROC-AUC | Test F1 | Recall |
|---|---|---|---|---|
| **Stacking (heterogeneous)** | 0.9629 | **0.9753** | 0.9020 | 1.0000 |
| Random Forest (bagging) | 0.9556 | 0.9730 | 0.8696 | 0.8696 |
| Soft Voting (heterogeneous) | 0.9623 | 0.9730 | 0.8889 | 0.8696 |
| XGBoost (boosting) | 0.9658 | 0.9636 | 0.9020 | 1.0000 |
| Logistic Regression (baseline) | 0.9228 | 0.9260 | 0.7500 | 0.6522 |
| Decision Tree (baseline) | 0.8877 | 0.8890 | 0.8627 | 0.9565 |

* **Best model:** Stacking (LR + tuned RF + tuned XGBoost → logistic meta-learner)
* **Gain over the baseline:** ROC-AUC **+0.0493**, F1 **+0.1520** → the ensemble does beat the baseline
* **Alert threshold:** 0.52, tuned on training folds only. At that threshold on September:
  **recall 1.000, precision 0.821**, confusion matrix `[[32, 5], [0, 23]]` — every fire day caught,
  5 wasted patrols out of 60 days
* **Fairness:** recall gap between the two districts = **0.000**
* **Live demo:** the synthetic hot/dry/9-rainless-days Bejaia record scores **P(fire) = 0.963 → HIGH RISK**

Headline: **without any FWI code**, four cheap sensor readings plus drought memory reach ROC-AUC 0.975
on a future month — evidence that low-cost fire screening is feasible where the expert index is not.

---

## 7. Acknowledgements

* **Dataset:** Abid, F. & Izeboudjen, N. (2019). *Algerian Forest Fires Dataset*. UCI Machine Learning
  Repository. <https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset>
* **Libraries:** scikit-learn, XGBoost, SHAP, pandas, NumPy, Matplotlib, seaborn, joblib
* **Methods:** Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*;
  Wolpert (1992), *Stacked Generalization*
* All code, feature engineering, framing and analysis are original work for this assessment. No borrowed
  notebooks or figures are reproduced.
