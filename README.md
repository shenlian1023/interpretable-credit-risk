# Credit risk under class imbalance

An individual undergraduate research project on minority-class detection and interpretable credit-risk decisions. In the exhibition dataset, only about 8.07% of applicants have the positive label. A classifier can therefore report high accuracy while missing most of those cases.

The study compares five models, examines sampling and decision thresholds, and uses SHAP and Explainable Boosting Machine (EBM) to inspect prediction behavior. The exhibition's final choice is XGBoost as the main predictor, with threshold adjustment as the main decision strategy. Detecting more positive cases also creates more false alarms; both outcomes are reported.

`Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `EBM`

[Results](#threshold-results) · [Research design](#research-design) · [Implementation](#implementation) · [Model explanations](#looking-inside-the-models) · [Run](#run-the-experiments) · [Exhibition sources](docs/exhibition-notes.md)

![Credit-risk experiment design and fold-local fitting boundaries](assets/method-overview.png)

The upper panel separates the study's three questions. The lower panel distinguishes fitting from validation prediction and threshold selection. It is a method illustration, not a new experimental result. [Figure sources and scope](docs/method-overview-illustrated.md).

## Threshold results

Changing the threshold of the original XGBoost method in Experiment 2 produced the following saved **validation-set** results:

| Operating point | Threshold | Positive-class recall | Positive-class precision | False negatives | False positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default threshold | 0.5000 | 0.0183 | 0.6355 | 3,656 | 39 |
| Validation-selected best-F1 threshold | 0.1514 | 0.4165 | 0.2430 | 2,173 | 4,831 |

The adjusted threshold detects more positive cases, but creates substantially more false alarms. The comparison measures that tradeoff; it does not establish better lending decisions or financial savings. The [full comparison](results/experiment_2/experiment_2_full_all_v2_results.csv) includes other imbalance methods and operating points.

Relative to that model's default threshold, the changed decisions reduce FN by 1,483 and add 4,792 FP: about **3.23 added false positives per avoided false negative**. This is a count-based tradeoff, not a monetary cost. The fitted model and scores stay the same, so its average precision remains 0.2471 at both operating points. Threshold adjustment changes binary decisions, not score ranking.

Threshold selection and reporting use the same validation split, which can make the selected result optimistic. The code reserves a test split, but the current Experiment 1 and 2 reporting paths do not evaluate it. A separate evaluation is needed after fixing the model and threshold.

## Research design

The Home Credit Default Risk dataset labels `TARGET = 1` as payment difficulty and `TARGET = 0` as its absence. It does not measure monetary loss.

The exhibition asks whether accuracy reflects minority detection, how imbalance methods alter FN/FP, and which features support predictions. In Experiment 1, all five models have accuracy near 92% but positive-class recall below 3%. This motivates examining recall, average precision, and error counts alongside accuracy. The [model comparison](results/experiment_1/experiment_1_model_comparison.csv) is a separate fit from the Experiment 2 operating points above.

| Experiment | Question | Implementation | Saved results |
| --- | --- | --- | --- |
| 1: model comparison | How do logistic regression, random forest, XGBoost, LightGBM, and EBM compare? | [Model selection](src/experiment_1_model_selection.py) | [Tables](results/experiment_1) |
| 2: imbalance and thresholds | How do resampling, weighting, and thresholds change missed detections and false alarms? | [Imbalance experiments](src/experiment_2_imbalance.py) | [Metrics](results/experiment_2) |
| 3: interpretation | Which features contribute to model predictions? | [Interpretability](src/experiment_3_interpretability.py) | [Global importance](results/experiment_3) |

I carried out the project individually, from [numeric/categorical preprocessing](src/preprocessing.py) through model comparison, imbalance experiments, and interpretation. The scripts use stratified splits and a fixed random seed where applicable.

## Implementation

### Keep preprocessing and resampling inside each training fold

The [Experiment 2 pipeline](src/experiment_2_imbalance.py) places the sampler between preprocessing and the estimator:

```python
return ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("sampler", RecordingSampler(sampler)),
        ("model", model),
    ]
)
```

`GridSearchCV` fits this pipeline within each stratified training fold. Resampling therefore occurs within that fold's training data, rather than before creating all the folds. This separates fitting from the held-out fold used to score each candidate. The search's `pr_auc` scorer is scikit-learn's `average_precision`; it selects candidates without fixing a decision threshold.

The [implementation notes](docs/implementation.md) explain fold-local preprocessing/resampling, threshold selection, and error-cost scenarios. These choices exercise experimental design and metric interpretation. Selecting and reporting thresholds on the same validation split remains a separate limitation; the pipeline does not remove that optimism.

### Select a threshold from aligned precision-recall points

[`find_thresholds`](src/experiment_2_imbalance.py) computes F1 for each candidate operating point:

```python
if len(pr_thresholds) > 0:
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    thresholds["best_f1_threshold"] = pr_thresholds[f1_values.argmax()]
```

The precision-recall arrays have one more entry than the threshold array. `[:-1]` aligns their F1 values with actual candidates; the small denominator term avoids division by zero. Predictions then use `y_score >= threshold`. Separating score estimation from binary decisions makes the FN/FP tradeoff explicit, while also requiring an untouched final test after the threshold is fixed.

### Express a cost assumption as an objective

The [cost-threshold search](src/experiment_2_imbalance.py) evaluates:

```python
y_pred = (y_score >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
cost = fn_cost * fn + fp_cost * fp
```

Fixed labels keep the confusion-matrix ordering consistent. FN:FP weights of 5:1 and 10:1 are sensitivity scenarios, not measured lender costs. When unique scores exceed 500, the search uses 501 quantiles of those unique scores; the returned threshold minimizes the objective over the evaluated candidates.

These mechanisms practice pipeline-based experiment construction, imbalanced-classification evaluation, and translating a decision assumption into code. The [extended notes](docs/implementation.md) preserve the validation-selection and interpretation limits.

## Looking inside the models

![SHAP global feature importance](assets/shap-global-importance.png)

![EBM global feature importance](assets/ebm-global-importance.png)

These aggregated plots describe model behavior, not causal effects or fairness guarantees. Borrower-level records and local explanations are excluded.

The exhibition groups the observed signals into external credit scores (`EXT_SOURCE`), repayment burden, and applicant profile/stability. Its `EXT_SOURCE_MEAN` comparison shows broadly decreasing model contribution as the score rises in both XGBoost's SHAP dependence view and EBM's feature curve. That is a comparison of fitted-model behavior, not proof that changing the feature would cause a borrower's risk to change. The displayed plots above summarize global importance; the direction comparison belongs to the supplied exhibition slides.

## Run the experiments

Create a Python virtual environment and install the dependencies:

```bash
python -m venv .venv
# Linux or macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Obtain `application_train.csv` from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data), follow its access terms, and place it in `data/raw/`. From the repository root:

```bash
python src/data_check.py
python src/experiment_1_model_selection.py
python src/experiment_2_imbalance.py --quick-test
python src/experiment_3_interpretability.py --quick-test
```

Quick mode uses a smaller sample and does not reproduce the archived full-run metrics. For the full Experiment 2 configuration and interpretation run:

```bash
python src/experiment_2_imbalance.py --include-slow-methods --include-ebm --run-name full_all_v2
python src/experiment_3_interpretability.py
```

New artifacts go to `outputs/`; archived tables remain in `results/`. Dependencies are not version-pinned, so exact numerical reproduction may require the original library versions and runtime environment. The full dataset experiments were not rerun when preparing this repository.

## Interpretation limits and research support

- The 5:1 and 10:1 FN-to-FP cost ratios are assumed sensitivity scenarios, not measured lender costs.
- Experiments 1 and 2 use separate fits; their results are not the same baseline run.
- An NSTC undergraduate research grant was approved for 2026 to 2027. The research is ongoing; approval is not a completed research award or peer-reviewed publication.

The source comes from my [existing research repository](https://github.com/shenlian1023/Imbalanced_data_predict/tree/f131ecc372aaa393060a6bd7778d293dfccb305b). Aggregated tables and figures come from the project exhibition materials. The [exhibition notes](docs/exhibition-notes.md) record the poster and presentation used to check the narrative. Raw applicant data, trained models, personal documents, and credentials are not included.
