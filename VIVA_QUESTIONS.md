# Viva Questions and Simple Answers

1. **What does this project predict?**

   It predicts a battery's State of Health (SoH) as a percentage (%) using regression.

2. **Where did the dataset come from, and is it real measured data?**

   It came from the Kaggle dataset *Electric Vehicle (EV) Battery Degradation & Charge*. Kaggle says its 10,000 rows are synthetic simulations, not measurements from real EV batteries.

3. **What are the features and the target?**

   The target `SoH_Percent` is the value we predict. The five input features are vehicle age (months), total charging cycles (cycles), average temperature (°C), fast-charge ratio (unitless, 0–1), and average discharge rate (C-rate).

4. **Why is this a regression problem?**

   SoH is a numeric value, such as 94.60%, not a category. Regression predicts numbers; classification predicts labels such as Healthy or Replace Required.

5. **Why does the project leave some columns out of the model?**

   `Vehicle_ID` is only an identifier. `Battery_Status` is a second target. Internal resistance is strongly tied to SoH in the Kaggle description, and capacity could reveal information used to calculate SoH. Leaving these out helps reduce target leakage.

6. **Why split the data into training and test rows?**

   The model learns from the training rows (80%, or 8,000 rows). The held-out test rows (20%, or 2,000 rows) give a separate check of prediction performance.

7. **What is five-fold cross-validation?**

   The training rows are split into five parts. Each part is used once for validation while the other four parts train the model. The five R² scores are averaged to compare models more reliably than using only one split.

8. **What do MAE, MSE, RMSE, and R² tell us?**

   MAE is average absolute error in percentage points (%). MSE is average squared error in %² and penalizes large errors more. RMSE is the square root of MSE and is in %. R² is unitless and describes how much variation is explained on the scored data.

9. **What is the difference between Ridge and Lasso?**

   Ridge uses L2 regularization and shrinks model coefficients. Lasso uses L1 regularization and can shrink some coefficients to zero. Both can help control model complexity.

10. **What were the result and main limitation?**

    Linear Regression was selected by mean cross-validation R² (0.8008, unitless; tied with Ridge at four decimal places). Its held-out MAE was 1.0687 percentage points (%). The main limitation is that the data is synthetic, so these results do not validate performance on real batteries.
