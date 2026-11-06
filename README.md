# Human Activity Recognition — Streamlit Web App

## 1. Export model files from your training notebook

After all training and evaluation cells have run, add and execute the **Web App Export** cell supplied in the ChatGPT response. It creates an `artifacts` folder containing:
- three trained model files (`.pkl`)
- `metadata.json`
- `X_test.csv` and `y_test.csv`
- `feature_importance.csv`

Copy the generated `artifacts` folder into this project folder, beside `app.py`.

Expected layout:

```
HAR_Web_App/
  app.py
  requirements.txt
  artifacts/
    knn_har_model.pkl
    decision_tree_har_model.pkl
    random_forest_har_model.pkl
    metadata.json
    X_test.csv
    y_test.csv
    feature_importance.csv
```

## 2. Install dependencies

Open Anaconda Prompt / terminal in this folder:

```
pip install -r requirements.txt
```

## 3. Run the web app

```
streamlit run app.py
```

The app opens in your browser. If it does not open automatically, visit the local URL printed in the terminal (usually http://localhost:8501).

## Tabs included

1. Overview
2. Problem Statement
3. Solution Approach
4. Model Architecture
5. Results
6. Accuracy Test
7. Feature Insights

## Important notes

- The app uses the already-trained models from the notebook. It does not retrain models on each app launch.
- Official test-set accuracy is computed on the held-out UCI HAR test partition.
- Uploaded prediction CSVs must contain all 561 feature columns with the exact feature names created in the notebook. Include an `activity_id` column (1 to 6) if you want the app to calculate accuracy.
- This project compares KNN, Decision Tree and Random Forest. It does not establish superiority over all published HAR systems; external benchmark comparison requires implementing the corresponding published methods on the same split/protocol.
- The app works on engineered 561-feature rows. It does not collect live accelerometer/gyroscope data from a phone.
