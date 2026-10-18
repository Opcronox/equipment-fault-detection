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
