# Beginner's Walkthrough: Understand Before You Demo

## What problem are we solving?

Given a row of six operating inputs, we predict whether it belongs to the **machine-failure (1)** or **normal (0)** class. This is *supervised binary classification*: the CSV already tells us the correct class for training examples.

## What do inputs mean?

| Input | Plain-English meaning | Unit |
|---|---|---|
| Type | Product grade (L / M / H) | categorical |
| Air temperature | Temperature of the air around the process | Kelvin |
| Process temperature | Temperature at the manufacturing process | Kelvin |
| Rotational speed | Spindle / machine rotation rate | revolutions per minute (rpm) |
| Torque | Rotational force being applied | newton-metres (Nm) |
| Tool wear | Duration of tool wear during operation | minutes |

**Kelvin:** 300 K is about 26.85 °C. Use Kelvin when entering values in the app.

**Do not include** `TWF`, `HDF`, `PWF`, `OSF`, `RNF` as features! They describe failure reasons and strongly reveal the answer, which is called **target leakage**.

## Why do we divide the data?

We give the model 80% of the rows to **learn**, then reserve 20% to check whether it also classifies **unseen** rows. Stratification makes the failure ratio similar in both portions. Setting a random seed makes splits repeatable.

For most ML preprocessing, a simple rule applies: **fit preprocessing on training data only**. The `Pipeline` does this correctly. Otherwise, statistics from test data can influence training (**data leakage**).

## The three comparisons

**1. Majority-class baseline:** Always say *normal*. It can appear 96%+ accurate but miss **all** actual failures. It tells us how misleading the headline accuracy can be.

**2. Logistic regression:** A fairly simple, interpretable model that weighs each feature to calculate a probability. Inputs are standardized so differences in units (Kelvin vs rpm, etc.) don't dominate by scale.

**3. Random forest:** A collection of decision trees. Each tree checks conditions such as torque or speed ranges. The overall model aggregates many trees. It can discover non-linear relationships but risks overfitting and still needs testing.

Both trained models use `class_weight` to give the rare failure class more weight; that may increase recall at the cost of false alarms.

## The confusion matrix

| | Predicted normal (0) | Predicted failure (1) |
|---|---:|---:|
| Actually normal (0) | True negative (TN) | False positive (FP) |
| Actually failed (1) | False negative (FN) | True positive (TP) |

- **Precision = TP / (TP + FP)**: When an alarm goes off, how often is it right?
- **Recall = TP / (TP + FN)**: Of the actual failures, how many are detected?
- **F1 = 2 × precision × recall / (precision + recall)**: Balances the two.

**Interview example (illustrative numbers, NOT measured results):** There are 20 real failures, the model flags 25 records, and 15 are real failures. Then TP=15, FP=10, FN=5; precision=15/25=60%; recall=15/20=75%; F1≈66.7%. Use *your real results* from `reports/metrics.csv` when discussing this project.

## How does the Streamlit app work?

The model saves as a `.joblib` file after training. `app.py` loads it locally, builds one row from form entries, and calls `predict_proba()` to get the model's estimated probability of class 1. It shows an alert if the probability is at least the selected threshold (0.5 by default).

Changing the alert threshold changes the trade-off between false alarms and missed failures. It does **not** retrain the model or prove the probability is calibrated.

## What to actually do when running it

1. Open the source code and find `FEATURES` in `train.py`. Confirm there are only six features.
2. Run the official dataset download, and inspect the CSV header.
3. Run training. Copy down failure count, baseline accuracy, Logistic Regression precision/recall/F1, and Random Forest precision/recall/F1.
4. Open the random-forest confusion matrix PNG. Count the **false negatives**: why do they matter?
5. Launch the app. Try different speed, torque, and wear combinations, then upload `reports/sample_demo_inputs.csv` for a batch demonstration.
6. Open `reports/metrics.csv`. Compare each model's failure F1 and recall against the baseline, then explain the false-alarm tradeoff.
7. Only then replace résumé placeholders with your *measured* results.

## Frequent questions during interviews

**Why isn't accuracy enough?** Failures are rare; a model predicting normal every time has high accuracy but zero failure recall.

**Why use two models?** Logistic regression is a transparent linear benchmark; random forest models non-linear relations. Comparing them shows whether additional flexibility helps.

**Why not put failure modes into input features?** It would give the algorithm information that almost directly reveals the answer, causing unrealistic results.

**Does this predict failure 24 hours before it happens?** No. The published dataset labels individual operating records with their failure status; it does not provide the time-to-failure labels needed to promise advance warning.

**Is feature importance proof of a cause?** No, it shows how the fitted forest uses the feature; it does not prove mechanical causation.

**Why no deep learning?** This is a small, tabular dataset. A strong simple baseline and reliable evaluation make for a more understandable first project.
