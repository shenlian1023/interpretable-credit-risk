# Implementation notes

[Project overview](../README.md) · [Archived Experiment 2 results](../results/experiment_2)

## Fitting transforms and samplers inside each fold

Source: [`build_pipeline` and `run_grid_search`](../src/experiment_2_imbalance.py), [preprocessing and split definitions](../src/preprocessing.py).

```python
return ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("sampler", RecordingSampler(sampler)),
        ("model", model),
    ]
)
```

For methods with a sampler, an imbalanced-learn pipeline fits preprocessing, resamples the training rows, and fits the estimator in that order. `GridSearchCV` receives the entire pipeline and `StratifiedKFold`, so preprocessing and resampling are fitted inside each training fold. The held-out fold is transformed and predicted without resampling it. Methods without a sampler use a scikit-learn pipeline containing preprocessing and the estimator.

This avoids placing synthetic or resampled observations into the fold-splitting step. Stratification retains the original label proportions approximately in each fold; it does not balance the labels by itself. Full search uses three folds, while quick mode uses two. A fixed random seed controls the split where applicable.

The scorer named `pr_auc` maps to `average_precision`, and `refit="pr_auc"` chooses and refits the candidate with the best mean cross-validation score. Average precision and trapezoidal area under a precision-recall curve are not interchangeable definitions. The code's scorer should accompany reported comparisons.

The preprocessing module defines a 70% training, 15% validation, 15% test split. A reserved test split only becomes evaluation evidence when the reporting code actually uses it. Experiment 1 and 2 currently report on validation data.

## Choosing a threshold after obtaining scores

Source: [`find_thresholds` and `evaluate_model`](../src/experiment_2_imbalance.py).

```python
if len(pr_thresholds) > 0:
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    thresholds["best_f1_threshold"] = pr_thresholds[f1_values.argmax()]
```

`precision_recall_curve` returns one more precision/recall point than thresholds. Slicing off the final point keeps each F1 value aligned with an actual threshold. The small denominator term avoids division by zero. The selected threshold is then applied as `y_score >= threshold`.

Changing this threshold changes binary decisions without retraining the classifier. That separation lets the experiment compare missed detections and false alarms for the same fitted model. The saved XGBoost comparison increases recall from 0.0183 to 0.4165, alongside false positives increasing from 39 to 4,831. Both outcomes belong in the interpretation.

The selected threshold and reported metrics use the same validation data. A final assessment should freeze the fitted model and threshold before evaluating an untouched test set. The present result does not establish generalization of the chosen operating point.

## Turning an assumed cost ratio into a measurable objective

Source: [`find_best_cost_threshold` and `evaluate_threshold`](../src/experiment_2_imbalance.py).

```python
y_pred = (y_score >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
cost = fn_cost * fn + fp_cost * fp
```

The labels fix the confusion-matrix order, including cases in which a prediction contains only one class. The objective charges separately for false negatives and false positives. Scenarios with FN:FP weights of 5:1 and 10:1 test how an assumed preference changes the threshold; they are not measured lending costs or monetary savings.

The function searches unique score values. When there are more than 500 values, it samples 501 quantiles of those unique values. The chosen threshold minimizes the objective over the evaluated candidates, not necessarily over every possible operating point. That reduced search is a computational tradeoff worth stating explicitly.

## What the explanations establish

Source: [interpretability experiment](../src/experiment_3_interpretability.py).

SHAP and EBM outputs describe fitted model behavior. Their global summaries support questions about which features the models use, while aggregated results cannot establish causal effects, fairness, or reliability for an individual borrower. Model selection, threshold choice, and explanation answer different questions and should retain separate evidence.

## Technical skills practiced

| Skill | Evidence in this project |
| --- | --- |
| Experiment construction | Stratified splits, fold-local transforms, and pipeline-based model search |
| Imbalanced classification | Sampling/weighting comparisons and threshold-specific error counts |
| Evaluation reasoning | Average-precision selection, validation-selection limits, and assumed cost scenarios |
| Model interpretation | SHAP/EBM summaries with predictive rather than causal claims |

This is an individual undergraduate research project. Archived metrics support the reported comparisons; dependencies remain unpinned and the full experiments were not rerun for these notes. No new predictive result or deployed lending decision system is claimed.
