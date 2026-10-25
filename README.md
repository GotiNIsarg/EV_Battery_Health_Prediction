# EV Battery Health Prediction

A beginner machine-learning project for BCA Semester III, course **303-05 Fundamentals of Machine Learning**.

The project uses:

- **Linear Regression** to predict battery State of Health (SoH, %).
- **Logistic Regression** to predict the status recorded in the dataset.
- **Python input** so a user can enter five battery measurements and see example predictions.

> **Important:** The Kaggle dataset is synthetic. This project is for learning only. It cannot determine whether a real battery is safe. The teacher's project instructions request a real-world dataset, so ask your teacher to approve this dataset before using it.

## Project specifications

| Part | Project specification |
|---|---|
| Dataset | `Dataset/EV_Battery_Dataset.csv`; 10,000 rows and 13 columns |
| Regression | Simple Linear Regression uses `Total_Charging_Cycles`; Multiple Linear Regression uses five numeric inputs. Both predict `SoH_Percent` (%) |
| Classification | Logistic Regression uses `Vehicle_Age_Months`, `Total_Charging_Cycles`, `Avg_Temperature_C`, `Fast_Charge_Ratio`, and `Avg_Discharge_Rate_C` to predict `Battery_Status` |
| Status labels | The CSV has `Healthy` and `Replace Required`; the program displays the second label as `Not Healthy`. It does not use the 80% rule for this target |
| User results | Predicted status, predicted SoH (%), probability of Healthy (%), and simple example advice |
| Workflow | Problem definition, data collection, preparation, EDA, feature selection, train/test split, training, evaluation, prediction, deployment, and monitoring |
| Evaluation | Regression: MAE, MSE, RMSE, R², RSS, and five-fold cross-validation. Classification: accuracy, precision, recall, F1, confusion matrix, and five-fold cross-validation |
| Saved reports | PNG charts and result tables in `Reports/`; learning guide and separate Sweetviz EDA HTML report |

The classification inputs do not include `SoH_Percent`. This avoids target leakage, which means giving the model information that reveals the answer. The CSV contains only four `Replace Required` rows, so classification results are uncertain.

## Learning website and report

- Open the offline [Learning Website](Learning_Website/index.html). Keep its `images` folder beside `index.html`.
- The same report is available at [Reports/EV_Battery_Project_Report.html](Reports/EV_Battery_Project_Report.html).
- Run the program to generate the separate [Sweetviz EDA report](Reports/Sweetviz_EDA_Report.html) in `Reports/`.
- The website explains the battery terms, course units, ML stages, code examples, real run results, limitations, viva questions, and project checklist.

## Dataset

**File:** `Dataset/EV_Battery_Dataset.csv`

**Source:** [Kaggle — Electric Vehicle (EV) Battery Degradation and Charge](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge)

The supplied CSV has **10,000 rows** and **13 columns**. Its recorded status labels are:

| `Battery_Status` value | Rows |
|---|---:|
| `Healthy` | 9,996 |
| `Replace Required` | 4 |

The classifier uses this existing status label. It displays `Replace Required` as **Not Healthy**. It does not create a classification label using the 80% SoH rule. Because there are only four Not Healthy rows, classification results are uncertain.

### Model inputs and targets

Both tasks use these five input columns:

| CSV column | Unit shown by the column name or data |
|---|---|
| `Vehicle_Age_Months` | months |
| `Total_Charging_Cycles` | cycles |
| `Avg_Temperature_C` | °C |
| `Fast_Charge_Ratio` | ratio from 0 to 1; unitless |
| `Avg_Discharge_Rate_C` | C-rate |

- **Simple Linear Regression:** `Total_Charging_Cycles` predicts `SoH_Percent`.
- **Multiple Linear Regression:** all five inputs predict `SoH_Percent`.
- **Logistic Regression:** all five inputs predict `Battery_Status`.

`SoH_Percent` is not used as a classifier input. This avoids **target leakage**: accidentally giving a model the answer it is meant to predict.

## Run the project

Use Python **3.12** or another compatible Python installation. Open a terminal in the main project folder (the folder containing `Dataset`, `Python_Code`, and `Reports`) and run:

```text
python -m pip install -r requirements.txt
python -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code/project.py
```

The program displays charts one at a time and saves them, with result tables, as PNG files in `Reports/`. It also creates `Reports/Sweetviz_EDA_Report.html`. Warnings are treated as errors, except Sweetviz's known `pkg_resources` deprecation warning. Sweetviz 2.3.1 needs a numeric or True/False target, so the report uses a temporary `Healthy` True/False column made from `Battery_Status`. This does not change the CSV label or the model inputs. The program then asks for battery age, charging cycles, average temperature, fast-charge ratio, and average discharge rate.

The displayed status, SoH, probability, and advice are educational examples from synthetic data. Do not use them for real safety or replacement decisions.

## Models and evaluation

The project evaluates regression with **MAE, MSE, RMSE, R², and RSS**. It evaluates classification with **accuracy, precision, recall, F1, and a confusion matrix**. It also compares train/test scores and runs five-fold cross-validation.

These are the results from the recorded run on the supplied CSV:

| Regression model | MAE (%) | MSE (%²) | RMSE (%) | Test R² | RSS (%²) |
|---|---:|---:|---:|---:|---:|
| Simple Linear | 1.1219 | 2.6577 | 1.6302 | 0.7500 | 5315.3974 |
| Multiple Linear | 1.0683 | 2.2113 | 1.4870 | 0.7920 | 4422.5896 |

| Classification metric for Not Healthy | Test result |
|---|---:|
| Accuracy | 0.9965 |
| Precision | 0.1250 |
| Recall | 1.0000 |
| F1 score | 0.2222 |

The high accuracy does **not** mean rare Not Healthy cases are predicted reliably. The test set contained only one Not Healthy row, so the classification results are particularly uncertain.

## Course connection

| Course unit | Topics used in this project |
|---|---|
| Unit 1 | ML life cycle, data types, data checks, EDA, and charts |
| Unit 2 | Pandas, features and targets, simple regression, `y = mx + b`, and R² |
| Unit 3 | Supervised learning, regression and classification, train/test split, and error metrics |
| Unit 4 | Linear Regression, Logistic Regression, model evaluation, and cross-validation |

Multiple Linear Regression is described as “not in syllabus / just for knowledge” in the Unit 4 notes. It is included here because the project asks for it.

## Folder structure

```text
.
├── Dataset/
│   └── EV_Battery_Dataset.csv
├── Python_Code/
│   └── project.py
├── Reports/
│   ├── EV_Battery_Project_Report.html
│   ├── Sweetviz_EDA_Report.html  (created when the program runs)
│   └── chart and result PNG files
├── Learning_Website/
│   ├── index.html
│   └── images/
├── README.md
├── requirements.txt
└── .gitignore
```

## Limitations and teacher approval

- Kaggle describes this dataset as **synthetic**, not measured real-world battery data.
- The teacher's instructions request a real-world dataset. Get approval before submitting this dataset.
- The four `Replace Required` examples are too few to support a confident classification conclusion.
- This is a college learning exercise, **not** a real battery health, safety, or replacement service.

## Libraries

The project uses pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Sweetviz. Install them with:

```text
python -m pip install -r requirements.txt
```
