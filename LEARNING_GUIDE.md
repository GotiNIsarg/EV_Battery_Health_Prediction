# Short Learning Guide

These four short lessons follow the topics in the course notes and journal. Each example points to a line in `Python_Code/project.py`.

## Unit 1 — Fundamentals of AI and Machine Learning

**Idea:** Machine learning finds patterns in examples. First, check what information the examples contain.

**Real-life example:** Before guessing how healthy a battery is, check its age, cycles, and temperature.

**In this project:** `df.info()` shows the columns and data types. `df.describe()` summarizes number columns.

```python
df.info()
print(df.describe())
```

## Unit 2 — Data Exploration and Simple Linear Regression

**Idea:** Charts help us see patterns. A simple regression line uses `y = mx + b`.

**Real-life example:** A teacher can draw a line to see whether more practice hours are linked to higher test scores.

**In this project:** The histogram shows SoH (%). The simple model compares cycles to SoH (%).

```python
sns.histplot(df["SoH_Percent"])
model.fit(df[["Total_Charging_Cycles"]], df["SoH_Percent"])
```

The line's slope is change in SoH percentage points per cycle. The simple model R² is unitless.

## Unit 3 — Understanding Machine Learning

**Idea:** Supervised learning learns from examples that already have answers. Test data checks predictions on held-out examples.

**Real-life example:** Practise with some solved questions, then check your learning with questions you did not practise.

**In this project:** `X` contains five inputs. `y` contains the known SoH (%) answers. The split keeps 20% for testing.

```python
X = df[feature_columns]
y = df["SoH_Percent"]
train_test_split(X, y, test_size=0.20, random_state=42)
```

MAE and RMSE use SoH percentage points (%). MSE uses %². R² has no unit.

## Unit 4 — Supervised Learning Algorithms

**Idea:** Different regression algorithms can learn from the same examples. Cross-validation compares them using several training/validation splits.

**Real-life example:** Try three ways of solving the same practice problem, then compare their scores.

**In this project:** Linear, Ridge (L2), and Lasso (L1) are trained separately. Five-fold CV R² helps choose one.

```python
model1 = LinearRegression()
model2 = Ridge(alpha=1.0)
model3 = Lasso(alpha=0.1, max_iter=10000)
cross_val_score(model1, X_train, y_train, cv=5)
```

This simple version does not scale inputs. Ridge and Lasso can be affected by inputs measured in different units.
