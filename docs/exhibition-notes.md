# Exhibition sources and interpretation

[Project overview](../README.md)

The research narrative was checked against the supplied `海報_五版.pdf` and `C29_簡報檔.pdf`. The poster's SHA-256 is `b4d27bb79a588d70e09462ac0d95130495af259367ed05140222b1f83a5f7fb6`. `C29_海報.pdf` has the same hash and is the same poster. The supplied ZIP packages the poster, presentation, video, and three exhibition photos; it is not an additional independent experiment.

## What the exhibition establishes

| Material | Research content reflected in the README |
| --- | --- |
| Poster: motivation and questions | About 8.07% positive labels; accuracy, minority detection, error tradeoffs, and explanation are separate concerns |
| Poster / presentation: Experiment 1 | Five baseline models; accuracy near 92% and positive-class recall below 3% |
| Poster / presentation: Experiment 2 | XGBoost sampling/weighting comparisons and threshold-based FN/FP decisions |
| Presentation: Experiment 3 | SHAP signals and an EBM direction comparison for `EXT_SOURCE_MEAN` |
| Exhibition conclusion | XGBoost as the main predictor and threshold adjustment as the main decision strategy |

The presentation reports 282,686 negative and 24,825 positive labels in the full labeled data. These counts describe the dataset, not the size of a particular validation fold.

## Connecting rounded exhibition values to archived results

The poster rounds Experiment 2's default recall to 0.018 and adjusted recall to 0.416. The [archived CSV](../results/experiment_2/experiment_2_full_all_v2_results.csv) records 0.0182599 and 0.4164876 for the Original XGBoost method. Its selected best-F1 threshold is 0.1514379978, shown in the README as 0.1514.

The count-based tradeoff follows directly from the same two CSV rows:

```text
reduced_FN = 3656 - 2173 = 1483
added_FP   = 4831 - 39   = 4792
added_FP / reduced_FN   = 3.2312879
```

This ratio describes those operating points. It is not evidence of globally optimal lending decisions or measured financial benefit. The poster's broader statement about a favorable tradeoff should retain its experimental scope; different methods can have different operating constraints.

Experiment 1's XGBoost baseline and Experiment 2's Original row are separate fits with different saved metrics. They should not be spliced into a single before/after comparison. Threshold selection and reporting also share validation data; the source reserves a test split without using it in the current Experiment 1 and 2 reports.

## Explanations and future work

The exhibition's `EXT_SOURCE_MEAN` plots show broadly similar decreasing contribution trends in XGBoost SHAP and EBM. They use different model contribution scales and should not be treated as interchangeable numerical effects. Explanations describe predictions; they do not establish causality, fairness, or calibrated default probabilities.

Measured FN/FP costs, probability calibration, fairness checks, and stability across splits are listed as future work. This repository does not present them as completed capabilities. The new method figure is a source-checked explanatory illustration, not an empirical plot.

Original PDFs, student-identifying exhibition photos, the ZIP, and the presentation video were not copied into the repository. The video was not analyzed as independent numerical evidence. Aggregated results and the existing global figures remain the repository's experimental record.
