# EV Battery Health Prediction

## Project description

This beginner project uses Python and supervised machine learning to estimate an electric-vehicle battery's State of Health (SoH) as a percentage. It follows the data-inspection, EDA, cleaning, train/test, regression, evaluation, and model-saving workflow taught in BCA Sem-III course 303-05.

> **Important dataset warning:** The Kaggle author describes these 10,000 records as **synthetic simulations**, not measurements collected from real EV batteries. This is suitable for learning the workflow, but its scores do not prove that a model can diagnose a real battery. Validate any future model with independent measured battery-test data.

> **Course requirement:** The teacher's project instructions require a real-world dataset and faculty approval. This supplied synthetic dataset does **not** meet that requirement unless your faculty approves it as a learning demonstration. Ask for approval or replace it with an approved measured dataset before submitting.

## Dataset and units

Source: [Electric Vehicle (EV) Battery Degradation & Charge on Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge). The supplied copy is `Dataset/EV_Battery_Dataset.csv`. Kaggle describes 10,000 simulated records for LFP and NMC batteries and lists an Apache 2.0 license.

The unit descriptions below follow the Kaggle dataset description and column names. A dash means the column is a label or identifier, not a physical measurement.

| Column | Meaning | Unit or kind | Project use |
|---|---|---|---|
| `Vehicle_ID` | Vehicle record identifier | ID; no physical unit | Excluded from model |
| `Car_Model` | Vehicle model name | Category | Reported in data inspection |
| `Battery_Type` | LFP or NMC chemistry | Category | Used for EDA bar chart |
| `Battery_Capacity_kWh` | Listed battery capacity | kilowatt-hours (kWh) | Excluded as a possible target-linked value |
| `Vehicle_Age_Months` | Vehicle age | months | Model input |
| `Total_Charging_Cycles` | Cumulative charge/discharge cycles | cycles | Model input and simple-regression input |
| `Avg_Temperature_C` | Average operating temperature | degrees Celsius (°C) | Model input |
| `Fast_Charge_Ratio` | Share of charging sessions using fast charging | ratio from 0 to 1; unitless | Model input |
| `Avg_Discharge_Rate_C` | Average battery discharge rate | C-rate | Model input |
| `Driving_Style` | Driving-style label | Category | Reported in data inspection |
| `Internal_Resistance_Ohm` | Battery internal resistance | ohms (Ω) | Excluded because Kaggle says it increases as SoH drops |
| `SoH_Percent` | State of Health; regression target | percent (%) | Target |
| `Battery_Status` | Healthy or Replace Required | Category | EDA only; not a model input |

## Folder structure

```text
EV_Battery_Health_Prediction/
├── Dataset/
│   └── EV_Battery_Dataset.csv
├── Python_Code/
│   ├── project.py
│   └── best_model.pkl
├── Reports/
│   ├── EDA_Report.html
│   ├── Project_Report.pdf
│   ├── soh_histogram.png
│   ├── soh_boxplot.png
│   ├── cycles_vs_soh.png
│   ├── correlation_heatmap.png
│   ├── battery_status_pie.png
│   ├── battery_type_bar.png
│   └── simple_regression_line.png
├── Fundamentals of Machine Learning/
│   └── README.md
├── main.py
├── requirements.txt
├── LEARNING_GUIDE.md
├── VIVA_QUESTIONS.md
└── .gitignore
```

The local `Fundamentals of Machine Learning` folder contains the course PDFs. The PDFs are not copied to this public project; the folder's README and `LEARNING_GUIDE.md` summarize the topics without redistributing the handouts. Share the original course files only if you have permission.

## Install and run

Tested with Python 3.12.10 and the package versions listed in `requirements.txt`.

1. Clone or download this repository.
2. Open a terminal in the project root.
3. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the required libraries:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Run the project from the repository root:

   ```powershell
   python Python_Code\project.py
   ```

   Or use the root launcher:

   ```powershell
   python main.py
   ```

The program prints the inspection and model results from the supplied CSV, saves seven chart images and `Project_Report.pdf` in `Reports/`, and saves the selected estimator in `Python_Code/best_model.pkl`. Running it again replaces those generated outputs.

## How to build this project yourself

Follow these steps in order. Times are estimates for a first-time student; your time may differ.

| Step | What you do | Estimated time |
|---|---|---:|
| 1 | Read the project instructions and confirm that the dataset matches the question | 20 minutes |
| 2 | Load the CSV and inspect its rows, columns, types, and units | 25 minutes |
| 3 | Check missing values, duplicates, and IQR outliers; write down cleaning decisions | 30 minutes |
| 4 | Make and label the six EDA charts | 35 minutes |
| 5 | Fit the simple line, explain its slope, R², and RSS | 25 minutes |
| 6 | Select features and the target, then split the data 80/20 | 20 minutes |
| 7 | Train Linear, Ridge (L2), and Lasso (L1) models with five-fold cross-validation | 45 minutes |
| 8 | Compare test-set MAE, MSE, RMSE, and R² | 25 minutes |
| 9 | Save the selected estimator and predict one held-out sample | 20 minutes |
| 10 | Generate the optional HTML report, write the report, and check the repository | 60 minutes |
| **Total** | **Estimated project time** | **About 5 hours** |

The script uses five numerical input measurements: vehicle age (months), charging cycles (cycles), temperature (°C), fast-charge ratio (0–1), and discharge rate (C-rate). It deliberately leaves out `Vehicle_ID` (an identifier), `Battery_Status` (a second target), `Internal_Resistance_Ohm` (strongly tied to SoH in the Kaggle description), and `Battery_Capacity_kWh` (a possible capacity/target leakage risk). It also leaves out text categories to keep the first model small and understandable.

### Optional automated EDA HTML report

This is a separate extra package; it is not imported by the main model code and is not included in the core `requirements.txt`. Use a separate environment so this optional package does not change the tested core dependencies. From the project root:

```powershell
py -3.12 -m venv .venv-eda
.\.venv-eda\Scripts\Activate.ps1
python -m pip install ydata-profiling==4.17.0
```

Then run this separate snippet from the project root:

```python
import warnings
from pyparsing.warnings import PyparsingDeprecationWarning

# Ignore only known deprecations in this optional package's dependencies.
warnings.filterwarnings(
    "ignore",
    message="'imghdr' is deprecated and slated for removal in Python 3.13",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="'oneOf' deprecated - use 'one_of'",
    category=PyparsingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"datetime.datetime.utcnow\(\) is deprecated.*",
    category=DeprecationWarning,
)

import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
profile = ProfileReport(df, title="EV Battery Dataset Profile", explorative=True)
profile.to_file("Reports/EDA_Report.html")
```

Open `Reports/EDA_Report.html` in a web browser. Automated EDA is a helpful extra, not a replacement for checking the columns, units, leakage risks, and generated results yourself.

## Results from the supplied CSV

These values are from running `Python_Code/project.py` on the supplied CSV with the recorded environment. The dataset has 10,000 rows and 13 columns, with 0 missing cells and 0 exact duplicate rows. The IQR rule flagged 101 input-feature values; the script capped those five input measurements at their IQR limits, leaving 10,000 rows and 0 flagged input outliers. It does not cap the SoH target.

The script used 8,000 training rows and 2,000 held-out testing rows. Five-fold cross-validation was performed on training rows only.

| Model | Mean five-fold CV R² (unitless) | Test MAE (%) | Test MSE (%²) | Test RMSE (%) | Test R² (unitless) |
|---|---:|---:|---:|---:|---:|
| Linear | 0.8004 | 1.0687 | 2.2163 | 1.4887 | 0.7916 |
| Ridge (L2) | 0.8004 | 1.0687 | 2.2163 | 1.4887 | 0.7916 |
| Lasso (L1) | 0.7976 | 1.0623 | 2.2665 | 1.5055 | 0.7868 |

Linear and Ridge tie at the displayed four-decimal precision for cross-validation R². The program selects Linear as the best model according to the sorted result table. These scores measure performance on this simulated dataset only.

The teaching-only simple regression uses charging cycles (cycles) to predict SoH (%):

```text
SOH (%) = -0.009001 (% per cycle) * cycles + 99.920179 (%)
Training R² (unitless): 0.762814
Training RSS (%²): 25094.825246
```

Because this line is fitted and scored on the same cleaned records, its R² is an in-sample teaching result, not an independent test score.

For one held-out sample, the input was vehicle age 68 months, 432 charging cycles, average temperature 24.3 °C, fast-charge ratio 0.91, and average discharge rate 0.61 C-rate. The actual SoH was 97.31%; the selected model predicted 95.41%.

## EDA charts

Charts are saved by the Python program and can be viewed here on GitHub:

![SoH histogram and density curve](Reports/soh_histogram.png)

![SoH box plot](Reports/soh_boxplot.png)

![Charging cycles versus SoH](Reports/cycles_vs_soh.png)

![Correlation heatmap](Reports/correlation_heatmap.png)

![Battery status pie chart](Reports/battery_status_pie.png)

![Battery type bar chart](Reports/battery_type_bar.png)

![Simple regression line](Reports/simple_regression_line.png)

## Common mistakes and how to avoid them

- **Calling the records real-world measurements:** The Kaggle description says they are synthetic; state that clearly.
- **Using the identifier as a predictor:** `Vehicle_ID` is not a physical battery measurement, so it is excluded.
- **Using a field derived from SoH:** `Battery_Status` and internal resistance may reveal the target; keep them out of the primary model.
- **Assuming capacity is harmless:** Battery capacity can be directly related to the SoH calculation. This project excludes it from model inputs and documents the risk.
- **Fitting a scaler before cross-validation:** The script puts scaling inside each scikit-learn pipeline so each fold learns scaling values only from its training portion.
- **Reporting training scores as final performance:** Use the held-out test scores for the model comparison; the simple-regression R² is clearly labelled in-sample.
- **Treating IQR flags as errors:** The script caps the five input measurements, not the target. Review unusual values before deciding whether they are mistakes.
- **Running from the wrong folder:** Run the command from the repository root so the relative CSV and output paths resolve.
- **Opening an untrusted pickle file:** Only load `best_model.pkl` from a source you trust.

## Limitations and future improvements

### Limitations

- The records are synthetic simulations, not battery-lab measurements or real fleet observations.
- A random row split may put very similar simulated records in both training and testing sets.
- The project does not test on a separate vehicle, battery pack, manufacturer, or time period.
- The first model omits categorical information such as battery chemistry and vehicle model.
- The displayed scores should not be used for maintenance, warranty, or safety decisions.

### Future improvements

- Obtain permission to use measured EV battery-test data and validate predictions on it.
- Split by vehicle or battery pack, or use a time-based test split, to test generalization to unseen batteries.
- Compare models with and without capacity and resistance, while treating target leakage carefully.
- Encode useful categories such as battery type, then compare them against the simple numeric baseline.
- Tune Ridge and Lasso strengths using cross-validation on training data only.
- Record dataset version, measurement procedure, and units in a data dictionary.

## Create and push a GitHub repository

For a new local project, open a terminal in the project root. The `.gitignore` in this project keeps virtual environments, generated Python cache files, and the original course PDFs out of Git.

```powershell
git init
git branch -M main
git add .
git commit -m "Add EV battery health mini project"
```

Create an empty repository on GitHub (do not add a second README there). Then connect and push:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/EV_Battery_Health_Prediction.git
git push -u origin main
```

If `git remote -v` already shows a remote, use that existing remote rather than adding another one. Review `git status` and `git diff --cached` before committing so you do not include unrelated files or course handouts.

### Clone and run as another student

```powershell
git clone https://github.com/YOUR_USERNAME/EV_Battery_Health_Prediction.git
cd EV_Battery_Health_Prediction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python Python_Code\project.py
```

## Final submission checklist

- [ ] Confirm faculty approval for the project topic and dataset; the supplied dataset is synthetic and does not meet the course's real-world-data requirement on its own.
- [ ] Add the student's name, roll number, academic year, and any required faculty details to the report before submission.
- [ ] Keep `Dataset/EV_Battery_Dataset.csv`, `Python_Code/project.py`, `Reports/EDA_Report.html`, and `Reports/Project_Report.pdf` in the required folders.
- [ ] Run the project from the repository root and confirm that the charts, saved model, and PDF are regenerated.
- [ ] Review the reported units, metrics, leakage exclusions, and synthetic-data limitation before explaining the work in the viva.
- [ ] Do not publish the original course PDFs unless redistribution is permitted.

## Author

**Author:** [Add your name]

**Course:** BCA Sem-III, 303-05 Fundamentals of Machine Learning

**Project:** EV Battery Health Prediction

Fill in your name before submitting the project.
