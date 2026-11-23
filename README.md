# NYC Yellow Taxi 2026: Machine Learning with Spark MLlib

Three machine-learning projects built with Spark MLlib on about 12.25M cleaned NYC Yellow Taxi trips (January–May 2026) in Databricks: predicting the **fare**, predicting whether a rider will leave a **tip of 20% or more**, and **grouping pickup zones** by behaviour. Every model is compared with a simple baseline, tuned with cross-validation, scored once on data it never saw, then saved and tracked with MLflow.

**Dataset:** NYC Taxi and Limousine Commission (TLC) — [Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
**Input table:** `taxi.taxi_silver.yellow_taxi_trips_cleaned` (the Silver layer of the data-engineering project linked at the end)  
**Scale:** Projects A and B use a frozen 2% sample (about 245,000 trips, `taxi.taxi_ml.trips_sample`); Project C uses the full table

---

## Projects at a glance

| Project | Task | Models | Test result |
|---|---|---|---|
| A | Predict `fare_amount` (regression), at pickup and at drop-off | Linear regression, random forest, gradient-boosted trees (GBT) | At pickup, tuned GBT: RMSE $7.86, MAE $4.11, R² 0.816 (predicting the average: RMSE $18.32). At drop-off: RMSE $5.37, R² 0.914 |
| B | Predict whether a card tip is at least 20% (classification) | Logistic regression, random forest, GBT | Tuned GBT on May: AUC-ROC 0.670, AUC-PR 0.809 (no skill: 0.5 and 0.716) |
| C | Group pickup zones by behaviour (clustering) | KMeans | k = 6, silhouette 0.55; airports, downtown nightlife and the Manhattan core are stable across random seeds |

---

## Method

- **One rule against leakage:** a feature is allowed only if it is known when the prediction would be made. The fare model at pickup excludes distance and duration; the tip model excludes the tip itself.
- **Baselines first:** predicting the average (fare), always predicting a high tip (tip) and a zone-pair lookup (fare) set the bar every model has to beat.
- **Preprocessing inside the `Pipeline`:** encoders and scalers are fitted on training data only and applied unchanged to test data.
- **Validation:** grid search with cross-validation for Projects A and B (3 folds for fare, one fold per month for tips). The test set is used once, after every tuning decision.
- **Splits:** a random 80/20 split for the fare model; a time-based split (train January–April, test May) for the tip model, compared with a random split.
- **Tracking:** every model is logged to MLflow with parameters, metrics and tags; fitted pipelines are saved to a Unity Catalog volume and reloaded to confirm they reproduce the in-memory predictions.

---

## Project A: fare regression

Test set, scored once (RMSE and MAE in dollars):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Predict the training average | 18.32 | 12.21 | 0.000 |
| Zone-pair lookup (average fare of each pickup → drop-off pair) | 8.49 | 3.98 | 0.785 |
| Linear regression, pickup features, one-hot zones | 10.70 | 6.69 | 0.659 |
| GBT, pickup features, tuned, zones as native categories | 7.86 | 4.11 | 0.816 |
| Same, plus an out-of-fold zone-pair feature | 7.89 | 3.76 | 0.814 |
| GBT, drop-off features (distance and duration added) | 5.37 | 1.84 | 0.914 |

- **The zone encoding mattered more than the tuning.** A plain zone-pair lookup beat every one-hot model, because fares depend on the route and separate one-hot zone columns cannot express it (a likely reason, not tested separately). Letting the trees use zones as native categories (validation RMSE 9.114 against 11.660 for one-hot) and tuning with 3-fold cross-validation made the GBT the best pickup model.
- **The zone-pair feature improved the typical error, not the large errors.** Built from out-of-fold training averages and checked for leakage, it lowered MAE from $4.11 to $3.76 but left RMSE unchanged. Its cross-validated RMSE gain ($0.22) was slightly optimistic, so the test score is the reference.
- **Drop-off features are a ceiling, not a forecast:** the meter charges by distance and time. The largest errors are long trips ending in zone 265 ("Outside of NYC"), whose fares do not follow the meter.

## Project B: will the rider tip 20% or more?

Credit-card trips only, because cash tips are not recorded. Scored on May after training on January–April:

| Model | AUC-ROC | AUC-PR |
|---|---|---|
| Always predict a high tip | 0.500 | 0.716 |
| Random forest, hand-set | 0.651 | 0.796 |
| Logistic regression, hand-set | 0.665 | 0.804 |
| Logistic regression, tuned | 0.666 | 0.805 |
| GBT classifier, tuned | 0.670 | 0.809 |

- **The signal is real but modest.** The tuned GBT and the tuned logistic regression are effectively tied (a 0.004 difference is one fold-to-fold spread). A plausible reason, not tested here: whether a rider tips 20% or more depends mostly on the rider, which the trip data does not contain.
- **The class mix drove three choices.** High tips are the majority (71.7%), so AUC-ROC was the tuning metric (a no-skill AUC-PR is already about 0.72), class weights balanced training, and balanced accuracy chose the decision threshold, because F1 for the positive class would have flagged over 93% of trips.
- **Tuning used month folds** (one training month held out at a time). The one-standard-error rule chose the simplest of the tied settings, depth-2 trees with 40 rounds.
- **The scores were rescaled to be read as probabilities.** Class weights push the raw scores about 0.2 below the real rates; rescaling with `p / (p + w_pos * (1 - p))` gives a mean prediction of 0.717 against an actual 0.716 on May (Brier score 0.173 against 0.203 for the base rate).
- **A random split is slightly optimistic:** it scored 0.004 to 0.007 AUC higher than the time-based split for the tuned model, so the time-based number is the one reported. The decision threshold (0.50 on the raw score) was chosen on April, which was also one of the cross-validation months; May was used once.

## Project C: grouping pickup zones

Each of 216 zones (at least 500 pickups) becomes one row: log trip count, average distance, fare and duration, and the night, weekend and airport shares. Features are standardised, and k = 6 was chosen from the elbow and silhouette plots and from interpretability.

| Cluster | Zones | Avg distance (miles) | Avg fare ($) | Stable across seeds? |
|---|---|---|---|---|
| Manhattan core | 48 | 2.92 | 18.08 | yes |
| Downtown nightlife (night share 41.6%) | 12 | 3.77 | 22.69 | yes |
| Airports (94.2% airport trips) | 3 | 11.56 | 50.67 | yes |
| Residential neighbourhoods, medium volume | 58 | 6.38 | 30.17 | no (soft) |
| Outer-borough residential, very low volume | 85 | 10.42 | 39.60 | no (soft) |
| Outlying zones with many airport trips | 10 | 10.21 | 56.16 | no (soft) |

- **k = 3 has the higher silhouette (0.70 against 0.55)** but only isolates the three airport zones, which the trip type already identifies; k = 6 also separates the nightlife zones from the Manhattan core.
- **Seed stability:** the final model was compared with ten refits using other random seeds (adjusted Rand index 0.45 to 0.81). The airport, nightlife and Manhattan core groups hold; the split of the other 153 zones does not, so those three groups are a soft split. The cluster names are interpretations of the profiles and busiest zones.

---

## Saved models and tracked runs

- **Four fitted pipelines** in the Unity Catalog volume `/Volumes/taxi/taxi_ml/ml_models`: the tuned fare GBT, the fare GBT with the zone-pair feature, the tuned tip GBT (with a `decision_settings.json` holding the threshold and the rescaling weight) and the zone KMeans pipeline. Each was reloaded and checked against the in-memory model.
- **Twenty-five MLflow runs**, tagged by `task`: ten for fare, six for tips, nine for clustering.
- **Delta tables** in `taxi.taxi_ml`: the frozen samples and splits, `zone_profile`, `zone_clusters` (the assignment) and `zone_cluster_names`.

---

## Limitations

- Projects A and B are trained on a 2% sample (about 245,000 trips) and five months of data, so seasonality and long-term drift cannot be assessed.
- Project A's pickup models assume the destination zone is known, as in a fare estimate.
- Project B uses trip data only and shows modest skill.
- Project C: three of the six clusters have no stable boundaries across random seeds.

---

## Repository structure

```
nyc-yellow-taxi-machine-learning/
├── README.md
├── .gitignore
└── notebooks/
    └── taxi_dataset_mllib.ipynb     # all three projects, with outputs
```

## How to run

1. Build the Silver table `taxi.taxi_silver.yellow_taxi_trips_cleaned` first. The notebook that produces it is in the related repository below.
2. Use Databricks serverless compute with Unity Catalog. Create the schema `taxi.taxi_ml`; the notebook does not create it, but it creates the volumes `spark_temp` and `ml_models`.
3. Open `taxi_dataset_mllib.ipynb` and run all cells top to bottom (or Restart and Run All).

Serverless Spark ML needs a Unity Catalog volume for temporary files (`SPARKML_TEMP_DFS_PATH`, set in the setup cell) and limits a single saved model to 100 MB; the saved models here are well under that.

## Tools

Databricks (serverless), Apache Spark MLlib (`Pipeline`, `CrossValidator`, `GBTRegressor`, `GBTClassifier`, `KMeans`), MLflow, Delta Lake, Unity Catalog, pandas, matplotlib.

---

## Related repository

The data pipeline that builds the cleaned Silver table used here (Bronze → Silver → Gold, SQL and PySpark, plus the dashboard) is in **[nyc-yellow-taxi-2026](https://github.com/panagiotisflrs/nyc-yellow-taxi-2026.git)**.
