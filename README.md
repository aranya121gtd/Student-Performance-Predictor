# Student Performance Predictor Dashboard

A Streamlit dashboard built around the uploaded notebook and CSV dataset.

## What it includes
- Overview of the 50,000-row dataset
- Math, reading and writing score distributions
- Score trends by study time and test-preparation course
- Individual student prediction using three TabNet regression models
- Test-set evaluation using R², MAE and RMSE
- Dataset preview and missing-value explorer

## Important project-specific details
- Targets: `math_score`, `reading_score`, `writing_score`
- Inputs: the 21 categorical and 16 numeric features defined in the notebook
- The performance band (A–F) is derived from the mean of the three predicted scores.
- The dashboard does not claim a separately trained pass/fail or calibrated risk classifier.
- `G1`, `G2`, and `G3` are retained because the notebook includes them as input features. The notebook notes that they are on a separate 0–20 scale and were independently merged.

## Setup on Windows (PowerShell)

1. Put these files in one folder:
   - `app.py`
   - `requirements.txt`
   - `stud_synthetic_50000_full.csv`

2. Open PowerShell in that folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

3. Start the dashboard:

```powershell
streamlit run app.py
```

4. Open the local URL shown in the terminal (usually http://localhost:8501).

## Train the model

- Open **Model Performance** in the sidebar.
- Click **Train / Retrain TabNet Models**.
- The dashboard trains three separate TabNet regressors, one for each subject score.
- Training may take several minutes and is best run on a machine with sufficient RAM. CPU training is supported; GPU availability depends on the installed PyTorch build.
- When training finishes, the models and preprocessing objects are saved in:
  `artifacts/student_tabnet_bundle.joblib`

The prediction page becomes available after training.

## Dataset path

By default, the app expects `stud_synthetic_50000_full.csv` in the same folder as `app.py`.

To use a different path in PowerShell:

```powershell
$env:STUDENT_DATA_PATH = "C:\full\path\to\your\dataset.csv"
streamlit run app.py
```

## Notes
- Model metrics are measured on the held-out test split. They are not hard-coded.
- The dashboard shows descriptive statistics before training, but it will not display fabricated model accuracy.
- Use the same dataset schema and preprocessing during training and prediction.
