
# STEP 47: Load Our Saved Model

import joblib
from pathlib import Path

model_path = Path("artifacts/my_random_forest.joblib")
# TODO 1:
# Load the saved file using joblib.load()
# Store the result in saved_bundle
saved_bundle = joblib.load(model_path)

# TODO 2:
# Retrieve the trained model from the dictionary
# Save it as loaded_model
loaded_model = saved_bundle["model"]

# TODO 3:
# Retrieve the selected threshold
# Save it as loaded_threshold
loaded_threshold = saved_bundle["threshold"]

# TODO 4:
# Print the selected threshold
print(loaded_threshold)
