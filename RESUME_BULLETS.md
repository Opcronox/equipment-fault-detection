# Résumé-ready wording (update after verifying your results)

**Project title:** Equipment Fault Detection Using Machine Learning | Python, scikit-learn, pandas, Streamlit

- Developed a machine-failure classification prototype using **10,000 synthetic industrial operating records** from the UCI AI4I 2020 dataset, including sensor measurements of temperature, torque, rotational speed and tool wear.
- Built reproducible preprocessing and **80/20 stratified train-test** pipelines, comparing **Logistic Regression** and **Random Forest** against a majority-class baseline while excluding failure-type labels to prevent data leakage.
- Evaluated rare-failure detection using **precision, recall, F1-score and confusion matrices**, and created a local **Streamlit interface** for individual and batch predictions.

**After you personally run the code on the official CSV:** Add a measured statement such as: "Achieved **[your actual value]** failure-class F1 and **[your actual value]** recall on **[actual number]** held-out test cases." The values must come directly from `reports/metrics.csv` and `reports/run_metadata.json`. Do **not** claim any score from the code fixture tests.

**Interview-friendly 20-second summary:**

"I built a machine-fault classification prototype using a 10,000-row synthetic industrial dataset. I used Python to clean operating measurements and compare logistic regression with a random forest against a baseline. Since machine failures were rare, I focused on precision, recall and F1 instead of just accuracy. I also built a simple Streamlit demo and documented the main limitation: this classifies recorded failure status, not how many hours remain before a future breakdown."
