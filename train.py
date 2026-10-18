import pandas as pd

#Step 1: Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

#Step 2: Explore the dataset
print("Database Shape")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMachine Failure Distributions:")
print(df["Machine failure"].value_counts())

print("\nFailure Percentages:") #normalize = True turns value_counts into percentages/porportions
print(df["Machine failure"].value_counts(normalize = True) * 100)


from sklearn.model_selection import train_test_split

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

#seperate FEATURES from TARGET
x = df[FEATURES]
y = df[TARGET]

#split training and test data
X_train, X_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = 0.20,
    random_state = 42,
    stratify = y
)

print("\nTraining Observations: ", len(X_train))
print("Testing Observations: ", len(X_test))

print("\nTraining Failure Distributions:")
print(y_train.value_counts())

print("\nTesting Failure Distributions:")
print(y_test.value_counts())





#MODEL 1 (Majority-Class Baseline)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score
baseline = DummyClassifier(strategy="most_frequent")
#Fit the model using training data
baseline.fit(X_train,y_train)

#Generate predictions using testing data
y_predict = baseline.predict(X_test)

#Calculate and print accuracy and recall
accuracy = accuracy_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
print("\nBaseline")
print(f"Accuracy: {accuracy:.2%}") 
print(f"Recall: {recall:.2%}")

#MODEL 2 (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Separate numerical and categorical features
numerical_features = FEATURES[1:]
catergorical_features = ["Type"]

#define preprocessing 
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numerical_features),
    ("catergoric", OneHotEncoder(handle_unknown="ignore"), catergorical_features)
])

# Create a Pipeline containing:
# 1. preprocessor
# 2. LogisticRegression

logistic_model = LogisticRegression(
    max_iter = 100,
    class_weight = "balanced",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("logistic_model", (logistic_model))
])

# Train the model using X_train and y_train
pipeline.fit(X_train, y_train)
# Predict using X_test
pred = pipeline.predict(X_test)

# Calculate accuracy and recall
accuracy_2 = accuracy_score(y_test, pred)
recall_2 = recall_score(y_test, pred)

# Print the results as percentages
print("\nLogistic Regression")
print(f"\nAccuracy: {accuracy_2:.2%}") 
print(f"Recall: {recall_2:.2%}")


from sklearn.metrics import precision_score, f1_score, confusion_matrix
#Additional Evaluation
# Calculate precision
precision = precision_score(y_test, pred)

# Calculate F1-score
score = f1_score(y_test, pred)

# Calculate the confusion matrix
matrix = confusion_matrix(y_test, pred)

# Print precision and F1-score as percentages
# Print the confusion matrix as a table
print(f"Precision: {precision:.2%}") 
print(f"f1_score: {score:.2%}")
print(matrix)

#MODEL 3 (Random Forest Classifier)
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier with:
# n_estimators=200
# min_samples_leaf=2
# class_weight="balanced"
# random_state=42
random_forest = RandomForestClassifier(n_estimators=200,
                                       min_samples_leaf=2,
                                       class_weight="balanced",
                                       random_state=42)

# Create a Pipeline using:
# 1. Your existing preprocessor
# 2. Your Random Forest model
pipeline_two = Pipeline([
    ("preprocessor", preprocessor),
    ("randomforest", random_forest)
])

# Train the Pipeline using X_train and y_train
pipeline_two.fit(X_train, y_train)

# Predict using X_test
# Store your predictions as rf_pred
rf_pred = pipeline_two.predict(X_test)

# Calculate accuracy and recall
accuracy_3 = accuracy_score(y_test, rf_pred)
recall_3 = recall_score(y_test, rf_pred)

# Print both results as percentages
print("\nRandom Forest Model")
print(f"\nAccuracy: {accuracy_3:.2%}")
print(f"Recall: {recall_3:.2%}")

rf_precision = precision_score(y_test, rf_pred)

rf_score = f1_score(y_test, rf_pred)

rf_matrix = confusion_matrix(y_test, rf_pred)
print(rf_matrix)
print(f"Precision: {rf_precision:.2%}") 
print(f"f1_score: {rf_score:.2%}")

# MODEL IMPROVEMENT: Validation Split
# Split X_train and y_train again into:
# X_build, X_val, y_build, y_val

# Use:
# test_size=0.20
# random_state=42
# stratify=y_train
X_build, X_val, y_build, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# Print the number of observations
# in X_build and X_val
print(len(X_build),len(X_val))

# MODEL IMPROVEMENT: Random Forest Validation

from sklearn.base import clone

# Create a fresh and unfitted model 
rf_val_pipeline = clone(pipeline_two)

# Train the model using X_build and y_build
rf_val_pipeline.fit(X_build, y_build)

# Predict the classes using X_val
# Store predictions in val_pred
rf_val_predictions = rf_val_pipeline.predict(X_val)

# Calculate validation accuracy and recall
rf_val_accuracy = accuracy_score(y_val, rf_val_predictions)
rf_val_recall = recall_score(y_val, rf_val_predictions)

print("\nRandom Forest Model Validation")
print(f"\nAccuracy: {rf_val_accuracy:.2%}")
print(f"Recall: {rf_val_recall:.2%}")

print("\nValidation Dataset Check")
print("Validation observations:", len(X_val))
print("Actual failure distribution:")
print(y_val.value_counts())

print("\nRandom Forest Validation Confusion Matrix:")
print(confusion_matrix(y_val, rf_val_predictions))

print("\nValidation Recall:")
print(recall_score(y_val, rf_val_predictions))

#Classification Threshold Tuning

# Get the estimated failure probabilities
rf_val_prob = rf_val_pipeline.predict_proba(X_val)[:, 1]

# Thresholds to experiment with
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

for threshold in thresholds:

    # Convert probabilities into 0 or 1
    # using the current threshold
    predictions = (rf_val_prob >= threshold).astype(int)
    
    # Calculate precision
    threshold_precision = precision_score(
        y_val, predictions, zero_division=0
    )   
    # Calculate recall
    threshold_recall = recall_score(y_val, predictions)

    # Calculate F1-score
    threshold_F1 = f1_score(y_val, predictions)

    # Print the threshold, precision,
    # recall and F1-score as percentages
    print(f"\nThreshold: {threshold:.2f}")
    print(f"Precision: {threshold_precision:.2%}")
    print(f"Recall: {threshold_recall:.2%}")
    print(f"F1-Score: {threshold_F1:.2%}")

# STEP 38: Automatically select the best threshold
best_threshold = None
best_f1 = -1
for threshold in thresholds:
    predictions = (rf_val_prob >= threshold).astype(int)
    current_f1 = f1_score(y_val, predictions)

    # Check whether current_f1 is greater than best_f1
    if current_f1 > best_f1:

    # If it is greater:
    # Update best_f1
    # Update best_threshold
        best_f1 = current_f1
        best_threshold = threshold

# Print the best threshold and its F1-score
print(f"\nThreshold: {best_threshold:.2f}")
print(f"Best F1-Score: {best_f1:.2%}")

#HYPERPARAMETER TUNING
from sklearn.model_selection import GridSearchCV
param_grid = {
    "randomforest__n_estimators": [100, 200],
    "randomforest__max_depth": [10, None],
    "randomforest__min_samples_leaf": [1,2]
}
grid_review = GridSearchCV(
    estimator=clone(pipeline_two),
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1
)
#train GridSearch
grid_review.fit(X_build, y_build)
print("Best Hyperparameters:", grid_review.best_params_)
print(f"Best F1:{grid_review.best_score_:.2%}")

# STEP 41: Evaluate Tuned Random Forest
# Retrieve the best trained model
tuned_rf = grid_review.best_estimator_

# Generate predictions using X_val
# Save as tuned_predictions
tuned_predictions = tuned_rf.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, tuned_predictions)

# Calculate precision
precision = precision_score(y_val, tuned_predictions)

# Calculate recall
recall = recall_score(y_val, tuned_predictions)

# Calculate F1-score
F1_score = f1_score(y_val, tuned_predictions)

# Print all four metrics as percentages
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-Score:  {F1_score:.2%}")

# STEP 44: Apply Selected Threshold
# Get estimated failure probabilities
tuned_probabilities = tuned_rf.predict_proba(X_val)[:, 1]
selected_threshold = 0.40

# Convert probabilities to class predictions
# using selected_threshold
tuned_predictions = (tuned_probabilities >= selected_threshold).astype(int)

# Calculate precision, recall and F1-score
precision = precision_score(y_val, tuned_predictions)
recall = recall_score(y_val, tuned_predictions)
f1 = f1_score(y_val, tuned_predictions)

# Print all three metrics as percentages
print(f"\nPrecision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-Score:  {f1:.2%}")

# STEP 46: Save Our Trained Model

import joblib
from pathlib import Path

# Create the directory for saved models
model_directory = Path("artifacts")
model_directory.mkdir(exist_ok=True)

# Define the file path
model_path = model_directory / "my_random_forest.joblib"

feature_ranges = {}
for feature in FEATURES[1:]:
    # Find the minimum value of this feature. Store it in minimum
    minimum = X_build[feature].min()

    # Find the maximum value of this feature. Store it in maximum
    maximum = X_build[feature].max()

    feature_ranges[feature] = {
        "min": minimum,
        "max": maximum
    }

# Package our trained model and threshold
model_bundle = {
    "model": tuned_rf,
    "threshold": selected_threshold,
    "features": FEATURES,
    "feature_ranges": feature_ranges
}

# Save model_bundle to model_path using joblib.dump()
joblib.dump(model_bundle, model_path)

# Print a message showing where the model was saved
print("Model was saved at ", model_path)


# STEP 77: Fair Model Comparison
models = {
    "Baseline": clone(baseline),
    "Logistic Regression": clone(pipeline),
    "Random Forest": clone(pipeline_two)
}

results = []

for name, model in models.items():
    model.fit(X_build, y_build)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)  # zero_division handles edge cases safely
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)