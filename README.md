<p align="center">
  <img src="https://placehold.co/900x220/12324b/ffffff?text=EV+Battery+Health+Prediction" alt="EV Battery Health Prediction banner">
</p>

<h1 align="center">EV Battery Health Prediction</h1>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-31210/"><img src="https://img.shields.io/badge/Python-3.12.10-blue" alt="Python 3.12.10"></a>
  <img src="https://img.shields.io/badge/Status-Learning%20project-informational" alt="Learning project">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://github.com/hitgalani/EV_Battery_Health_Prediction/commits/main"><img src="https://img.shields.io/github/last-commit/hitgalani/EV_Battery_Health_Prediction?label=last%20commit" alt="Last commit"></a>
</p>

<p align="center">A friendly machine-learning project that predicts battery State of Health (SoH, %) and classifies the dataset's battery status from five battery measurements.</p>

---

## Table of Contents

- [About the Project](#about-the-project)
- [Problem Statement and Objectives](#problem-statement-and-objectives)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Folder Structure](#folder-structure)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Results](#results)
- [Screenshots and Demo](#screenshots-and-demo)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author and Contact](#author-and-contact)
- [Acknowledgements](#acknowledgements)

## About the Project

This project demonstrates a basic supervised machine-learning workflow with an electric vehicle battery dataset. It predicts:

- A numeric **State of Health (SoH)** percentage using Linear Regression.
- The dataset's **battery status** using Logistic Regression.

It is intended for beginner students learning data preparation, exploratory data analysis, regression, classification, evaluation, and prediction.

> **Safety and data note:** The Kaggle dataset is synthetic. This project is for learning only. It cannot assess a real battery or provide safety advice. The course project instructions request real-world data, so get your teacher's approval before submitting this dataset.

Open the offline [Learning Website](Learning_Website/index.html) for the step-by-step guide. Its images are in [Learning_Website/images](Learning_Website/images).

## Problem Statement and Objectives

### Problem

Can five measurements in the supplied dataset help estimate SoH and predict the status label recorded in that dataset?

### Objectives

1. Inspect and prepare the supplied CSV.
2. Explore its values with tables and charts.
3. Train simple and multiple Linear Regression models for SoH.
4. Train Logistic Regression for the existing battery status label.
5. Evaluate the models with a held-out test set and five-fold cross-validation.
6. Ask the user for five measurements and display an educational prediction.

## Key Features

- Simple Linear Regression using charging cycles only.
- Multiple Linear Regression using five numeric inputs.
- Logistic Regression using the same five inputs to predict dataset status.
- Regression metrics: MAE, MSE, RMSE, R², and RSS.
- Classification metrics: accuracy, precision, recall, F1, and confusion matrix.
- Train/test score comparison and five-fold cross-validation.
- PNG charts and result tables saved in `Reports/`.
- Separate Sweetviz exploratory report saved as HTML.
- Console input with units and a simple example prediction.
- Offline, phone-friendly learning guide with project images.

## Tech Stack

| Tool | Purpose | Version |
|---|---|---:|
| Python | Programming language | 3.12.10 used for the recorded run |
| pandas | Read and prepare tabular data | 2.2.3 |
| NumPy | Numeric calculations | 1.26.4 |
| Matplotlib | Create and save charts | 3.11.2 |
| Seaborn | Statistical charts and heatmaps | 0.13.2 |
| scikit-learn | Regression, classification, splitting, and evaluation | 1.9.1 |
| Sweetviz | Create the separate EDA HTML report | 2.3.1 |
| setuptools | Compatibility dependency used by Sweetviz | 80.9.0 |

Exact dependency versions are listed in [requirements.txt](requirements.txt).

## Dataset Description

| Detail | Value |
|---|---|
| Dataset | Electric Vehicle (EV) Battery Degradation and Charge |
| Source | [Kaggle dataset page](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge) |
| Local file | `Dataset/EV_Battery_Dataset.csv` |
| Rows | 10,000 |
| Columns | 13 |
| Data type | Synthetic dataset, not verified real-world measurements |

### Columns and units

Units below are taken from the CSV column names and recorded values. IDs and categories do not have physical measurement units.

The CSV has no voltage (`V`), current (`A`), or ampere-hour capacity (`Ah`) columns. They are not model inputs and are not inferred.

| CSV column | Meaning | Unit or type |
|---|---|---|
| `Vehicle_ID` | Record identifier | ID; no physical unit |
| `Car_Model` | Vehicle model | Category |
| `Battery_Type` | Battery type | Category |
| `Battery_Capacity_kWh` | Battery capacity | kWh |
| `Vehicle_Age_Months` | Vehicle age | months |
| `Total_Charging_Cycles` | Total charging cycles | cycles |
| `Avg_Temperature_C` | Average temperature | °C |
| `Fast_Charge_Ratio` | Share of fast charging | ratio, 0–1; unitless |
| `Avg_Discharge_Rate_C` | Average discharge rate | C-rate |
| `Driving_Style` | Driving-style description | Category |
| `Internal_Resistance_Ohm` | Internal resistance | ohms (Ω) |
| `SoH_Percent` | State of Health | % |
| `Battery_Status` | Status provided by the dataset | Category |

### Targets and model inputs

| Task | Model | Inputs | Target |
|---|---|---|---|
| Simple regression | Linear Regression | `Total_Charging_Cycles` | `SoH_Percent` (%) |
| Multiple regression | Linear Regression | `Vehicle_Age_Months`, `Total_Charging_Cycles`, `Avg_Temperature_C`, `Fast_Charge_Ratio`, `Avg_Discharge_Rate_C` | `SoH_Percent` (%) |
| Classification | Logistic Regression | The same five numeric inputs above | `Battery_Status` |

The file contains **9,996 `Healthy`** rows and **4 `Replace Required`** rows. The program displays `Replace Required` as **Not Healthy**. It uses the existing `Battery_Status` labels; it does **not** create a label using the 80% SoH rule.

`SoH_Percent` is not a classification input. Including it would be **target leakage**: giving the model information that reveals the answer it is supposed to predict.

## Project Workflow

```text
Problem definition
        ↓
Load and inspect the CSV
        ↓
Prepare data and explore it (EDA)
        ↓
Select features and targets
        ↓
Split training and test data
        ↓
Train Linear Regression and Logistic Regression
        ↓
Evaluate test results and five-fold cross-validation
        ↓
Predict SoH and status from user input
        ↓
Save charts and monitor limitations
```

## Understand the Python Code

Open `Python_Code/project.py`. It is one script that runs from top to bottom. It does not define custom functions or classes, so follow the printed stage names in order.

| Stage | What the code does |
|---|---|
| 1. Problem definition | Prints the prediction goals and learning-only safety warning. |
| 2. Data collection | Reads `Dataset/EV_Battery_Dataset.csv` into `df`, then prints its size, columns, sample rows, summaries, and status counts. |
| 3. Data preparation | Checks required columns for missing values and duplicate rows. Counts IQR outliers but keeps them in the data. |
| 4. EDA | Prints SoH summaries, creates a separate Sweetviz HTML report, and displays dataset charts. |
| 5. Feature selection | Sets the five allowed columns in `feature_columns`; assigns regression inputs and target to `X` and `y`; converts the status target to 1 for Healthy and 0 for Not Healthy. |
| 6. Train/test split | Uses 80% of rows for training and 20% for testing. The classification split is stratified to preserve the label proportions. |
| 7. Model training | `model1` is cycles-only Linear Regression; `model2` is five-input Linear Regression; `model3` scales the inputs and trains Logistic Regression. |
| 8. Model evaluation | Calculates regression errors and R², classification scores and confusion matrix, train/test scores, and five-fold cross-validation. |
| 9. Prediction | Predicts SoH for one held-out test row and compares it with that row's known SoH. |
| 10. Deployment | Asks for five values, then prints predicted status, SoH, Healthy probability, and example advice. |
| 11. Monitoring | Prints reminders about checking future results and getting teacher approval; it does not monitor a live vehicle. |

### Important variable names

| Name | Simple meaning |
|---|---|
| `df` | The complete dataset table. |
| `feature_columns` | The five model input column names. |
| `X`, `y` | Regression inputs and numeric SoH target. |
| `X_class`, `y_class` | Classification inputs and numeric status target. |
| `X_train`, `X_test` | Input rows used for learning and final testing. |
| `model1`, `model2`, `model3` | Simple regression, multiple regression, and logistic classification models. |
| `new_battery` | A one-row table made from values entered by the user. |

The classification model never receives `SoH_Percent` as an input. That prevents target leakage. For Sweetviz only, the script makes a temporary `Healthy` True/False column on a copy of the data because Sweetviz requires a numeric or Boolean target. This does not change the CSV or model inputs.

### What happens when charts appear?

Charts are saved as PNG files in `Reports/` and displayed one at a time. Close each chart window to continue. After the charts, the console prompts for the five battery measurements. The program finishes after printing the result and closing the final chart.

## Folder Structure

```text
EV_Battery_Health_Prediction/
├── Dataset/
│   └── EV_Battery_Dataset.csv
├── Python_Code/
│   └── project.py
├── Reports/
│   ├── EV_Battery_Project_Report.html
│   ├── Sweetviz_EDA_Report.html       # generated when the program runs
│   └── *.png                          # saved charts and result tables
├── Learning_Website/
│   ├── index.html
│   └── images/
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Setup

The commands below are for **Windows PowerShell**. Install Git and Python 3.12 first. During Python setup, make sure the Python launcher (`py`) is available. Run these commands from the folder where you want the project.

### 1. Clone the repository

```powershell
git clone https://github.com/hitgalani/EV_Battery_Health_Prediction.git
cd EV_Battery_Health_Prediction
```

### 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the virtual environment's Python executable directly as shown in [How to Run](#how-to-run).

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How to Run

Run the command from the **project root**: the folder containing `Dataset`, `Python_Code`, and `Reports`. This is needed because the program reads and writes project-relative paths.

```powershell
python -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code\project.py
```

The program displays charts one at a time. Close each chart window to continue. It saves charts and result tables in `Reports/`, and creates `Reports/Sweetviz_EDA_Report.html`.

If your terminal prompt ends with `\Python_Code>`, return to the project root first:

```powershell
cd ..
```

If you did not activate the virtual environment, use:

```powershell
.\.venv\Scripts\python.exe -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code\project.py
```

When prompted, enter five numeric measurements. For example, the following inputs were used in a recorded run:

```text
Vehicle age (months): 48
Total charging cycles (cycles): 400
Average temperature (°C): 25
Fast-charge ratio (0 to 1): 0.49
Average discharge rate (C-rate): 1.49
```

The program output for those inputs was:

```text
Battery status: Healthy
Predicted State of Health: 96.34 %
Probability of Healthy: 100.00 %
Advice: keep using
Learning project on synthetic data; not for real safety decisions.
```

The status, probability, and advice are model outputs from synthetic data. Do not use them to make real-world battery safety or replacement decisions.

## Results

The following regression metrics were recorded on the supplied CSV using the project's held-out test split. Lower MAE, MSE, RMSE, and RSS indicate smaller prediction errors; higher R² indicates a closer fit.

| Regression model | MAE (%) | MSE (%²) | RMSE (%) | Test R² | RSS (%²) |
|---|---:|---:|---:|---:|---:|
| Simple Linear Regression | 1.1219 | 2.6577 | 1.6302 | 0.7500 | 5315.3974 |
| Multiple Linear Regression | 1.0683 | 2.2113 | 1.4870 | 0.7920 | 4422.5896 |

**Best regression result in this run:** Multiple Linear Regression, with test R² of **0.7920** and MAE of **1.0683 percentage points**. This result describes this synthetic dataset only.

### Classification test results

| Metric for Not Healthy | Test result |
|---|---:|
| Accuracy | 0.9965 |
| Precision | 0.1250 |
| Recall | 1.0000 |
| F1 score | 0.2222 |

The confusion matrix uses rows for actual values and columns for predicted values, ordered **Not Healthy, Healthy**:

| Actual \ Predicted | Not Healthy | Healthy |
|---|---:|---:|
| Not Healthy | 1 | 0 |
| Healthy | 7 | 1992 |

There were only four Not Healthy rows in the full CSV and one in the test set. High accuracy does not mean rare Not Healthy cases are classified reliably.

### Training and cross-validation checks

| Model | Train score | Test score |
|---|---:|---:|
| Simple Linear Regression (R²) | 0.7656 | 0.7500 |
| Multiple Linear Regression (R²) | 0.8009 | 0.7920 |
| Logistic Regression (accuracy) | 0.9951 | 0.9965 |

| Model and score | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Simple Linear Regression (R²) | 0.7500 | 0.7631 | 0.7599 | 0.7678 | 0.7713 | 0.7624 |
| Multiple Linear Regression (R²) | 0.7920 | 0.7960 | 0.7959 | 0.8065 | 0.8036 | 0.7988 |
| Logistic Regression (accuracy) | 0.9945 | 0.9975 | 0.9970 | 0.9965 | 0.9950 | 0.9961 |

Four rare Not Healthy records cannot be represented in every five-fold validation test group. Classification cross-validation accuracy may therefore hide poor performance on that rare label.

## Screenshots and Demo

### User-input SoH prediction

The bar is labelled with the numeric SoH prediction.

![User-input battery SoH prediction](Reports/deployment_prediction.png)

### Regression model results

![Regression test metrics](Reports/model_results.png)

### Classification confusion matrix

![Classification confusion matrix](Reports/confusion_matrix.png)

### Learning website

Open the offline [learning website](Learning_Website/index.html) for the full project walkthrough.

> **[ADD SCREENSHOT HERE: learning website on a phone or desktop]**

## Limitations

- The dataset is synthetic and is not verified real-world battery measurement data.
- The course project instructions request a real-world dataset. Ask the teacher to approve this dataset before submission.
- Only four records use the `Replace Required` status, which makes classification results uncertain.
- A high accuracy score can hide errors on a rare class.
- The 80% SoH rule is discussed as a common concept but is not used to create this dataset's classification target.
- The output is a student demonstration, not a battery inspection, engineering assessment, or safety recommendation.
- Results on this dataset do not prove performance on real batteries.

## Future Improvements

- Use an appropriately licensed, teacher-approved dataset with enough examples of each battery status.
- Test on a separate, representative dataset before making performance claims.
- Explore battery remaining useful life prediction if suitable labelled data is available.
- Add real-time monitoring only with validated sensors, suitable data, and safety review.
- Create a simple web app for entering values after the model and data have been responsibly validated.

## Contributing

Contributions that improve explanations, documentation, accessibility, or reproducibility are welcome.

1. Fork the repository.
2. Create a branch for your change.
3. Make a focused update and test it.
4. Commit your changes with a clear message.
5. Open a pull request and explain what changed.

Please do not present this learning project as a real battery safety tool.

## License

The project code and documentation are available under the [MIT License](LICENSE). The dataset may have separate terms; check the Kaggle dataset page before reusing or redistributing it.

## Author and Contact

| Contact | Details |
|---|---|
| Name | Hit Galani |
| GitHub | [hitgalani](https://github.com/hitgalani) |
| LinkedIn | [ADD LINKEDIN PROFILE URL] |
| Email | [ADD CONTACT EMAIL] |

## Acknowledgements

- Dataset: [Electric Vehicle (EV) Battery Degradation and Charge on Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge). The dataset is synthetic.
- Thanks to the course instructor and learning materials for introducing the machine-learning workflow.
- Python and open-source library maintainers whose tools support this educational project.

---

If this project helps you learn, please consider giving the repository a star ⭐
