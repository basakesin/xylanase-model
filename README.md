# Xylanase Activity Prediction with XGBoost

This project processes experimental design sheets from an Excel file to predict xylanase enzyme activity using an XGBoost regression model. The model's predictions are compared with existing design-of-experiment (DoE) predictions.

## Features

- Processes multiple sheets from an Excel file.
- Trains XGBoost regression models for each sheet.
- Compares model predictions with existing ones.
- Saves error metrics and feature importances to Excel.
- Generates scatter plots comparing experimental and predicted values.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
