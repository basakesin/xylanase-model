import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Modelin performansını hesapla

# Load the Excel file
file_path = "/Users/basakesin/Downloads/Tugba-Merve/Xylanase_exp_design_new.xlsx"
excel_file = pd.ExcelFile(file_path)

# Create an Excel writer for results
output_path = "Xylanase_results_with_feature_importance.xlsx"
writer = pd.ExcelWriter(output_path, engine="xlsxwriter")

# Initialize a list to store error metrics
error_metrics = []

# Process each sheet in the Excel file
for sheet_name in excel_file.sheet_names:
    df = excel_file.parse(sheet_name)
    print(f"Processing sheet: {sheet_name}")

    # Ensure there are enough columns to process
    if df.shape[1] > 2:
            # Define features (X) and target (y)
            X = df.iloc[:, 1:-2]
            y = df.iloc[:, -2]
            y_pred_existing = df.iloc[:, -1]

            # Split data into training and test sets
            X_train, X_test, y_train, y_test, y_pred_existing_train, y_pred_existing_test = train_test_split(
                X, y, y_pred_existing, test_size=0.2, random_state=7
            )

            # Mark test samples
            df["Test"] = False
            df.loc[X_test.index, "Test"] = True
            n_train = len(y_train)
            p = X_train.shape[1]
            # Train an XGBoost regressor model
            model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            df.loc[X_test.index, "Model Prediction"] = y_test_pred

            # Calculate errors
            train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
            test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
            train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
            test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

            # Calculate errors for existing predictions
            existing_train_rmse = mean_squared_error(y_train, y_pred_existing_train, squared=False)
            existing_test_rmse = mean_squared_error(y_test, y_pred_existing_test, squared=False)
            existing_train_mape = mean_absolute_percentage_error(y_train, y_pred_existing_train)
            existing_test_mape = mean_absolute_percentage_error(y_test, y_pred_existing_test)

            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            r2_existing_train = r2_score(y_train, y_pred_existing_train)
            r2_existing_test = r2_score(y_test, y_pred_existing_test)

            adj_r2_train = adjusted_r2(r2_train, n_train, p)
            adj_r2_test = adjusted_r2(r2_test, len(y_test), p)
            adj_r2_existing_train = adjusted_r2(r2_existing_train, n_train, p)
            adj_r2_existing_test = adjusted_r2(r2_existing_test, len(y_test), p)

            # Store error metrics
            error_metrics.append({
                "Sheet Name": sheet_name,
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
                "Train MAPE": train_mape,
                "Test MAPE": test_mape,
                "DoE Train RMSE": existing_train_rmse,
                "Existing Test RMSE": existing_test_rmse,
                "Existing Train MAPE": existing_train_mape,
                "Existing Test MAPE": existing_test_mape,
                "Train Adjusted R2": adj_r2_train,
                "Test Adjusted R2" :adj_r2_test,
                "DoE Train Adjusted R2" :adj_r2_existing_train,
            "DoE Test Adjusted R2" :adj_r2_existing_test
            })

            # Extract feature importance
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
            importance_df = importance_df.sort_values(by="Importance", ascending=False)

            # Save feature importance
            importance_sheet_name = f"{sheet_name[:28]}_Imp" if len(sheet_name) > 28 else f"{sheet_name}_I"
            importance_df.to_excel(writer, sheet_name=importance_sheet_name, index=False)

            # Save the updated dataset
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            # Plot results for Test samples
            df_true = df[df["Test"] == True]
            df_train = df[df["Test"] == False]

            if not df_true.empty:
                plt.figure(figsize=(6, 6))
                plt.scatter(df_train.iloc[:, -4], df_train.iloc[:, -3],
                            color='blue', label="Model Prediction for Train Data", alpha=0.5, s=10)
                plt.scatter(df_true.iloc[:, -4], df_true["Model Prediction"],
                            color='blue', label="Model Prediction for Test Data", alpha=0.7, marker='*', s=50)
                plt.scatter(df_train.iloc[:, -4], y_pred_existing_train,
                            color='red', label="DoE Prediction for Train Data", alpha=0.5, s=10)
                plt.scatter(df_true.iloc[:, -4], y_pred_existing_test,
                            color='red', label="DoE Prediction for Test Data", alpha=0.7, marker='*', s=50)

                # Draw the x=y reference line
                x_min, x_max = min(df.iloc[:, -4]), max(df.iloc[:, -4])
                plt.plot([x_min, x_max], [x_min, x_max], color="black", linestyle="--", label="Experimental result")

                plt.xlabel("Experimental Xylanase Activity")
                plt.ylabel("Predicted/Model Xylanase Activity")
                plt.title(f"Prediction Results for {sheet_name}")

                # Update legend location and opacity
                plt.legend(loc='upper left', framealpha=0.6)
                plt.grid(True)

                # Save the figure
                fig_filename = f"{sheet_name}_results.png"
                plt.savefig(fig_filename, dpi=300, bbox_inches="tight")
                plt.show()

    else:
        print(f"Sheet {sheet_name} does not have enough columns for processing.\n")

# Save and close the Excel writer
writer.close()

# Convert the error metrics list to a DataFrame
error_df = pd.DataFrame(error_metrics)

# Save the error metrics table
error_df.to_excel("Xylanase_Model_Error_Metrics.xlsx", index=False)
print(f"Results written to {output_path}")
print("Error metrics saved as Xylanase_Model_Error_Metrics.xlsx")
