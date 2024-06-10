# Rossmann Time Series Analysis

## Project Overview

The goal of this project was to develop and train a model that predicts the average daily sales values for Rossmann stores based on data from the week preceding the forecast date. The project is organized into several folders:

- **scripts**: This folder contains the main scripts for training and prediction.
  - `main.py`: Trains the best model identified during experimentation, generates predictions and creates plots.
  - `train_models.py`: Trains various models, from which the best-performing one was selected for `main.py`. The performance of all models is documented in the `model_results.md` file.
  - `decompose_time_series.py`: Decomposes the time series data, revealing weekly seasonality. This information was used in the ARIMA model, one of the models evaluated in `train_models.py`.

- **src**: This folder includes utility functions.
  - `data.py`: Contains functions for preparing the data for analysis.
  - `plots.py`: Contains functions for generating plots.

- **tests**: This folder includes tests for the functions in `data.py`.

- **results**: This folder includes saved plots, a markdown file comparing model performances, and the saved best-trained model.

## How to Run the Project

1. **Clone the Project**: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/raodz/Rossmann-Time-Series-Analysis.git
   ```

2. **Install Dependencies**: Ensure you have all necessary packages installed. You can use `requirements.txt` if provided or install packages manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Data**: Create a folder named `data` in the project directory and add the `train.csv` file. You can download the dataset from [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales).

4. **Train Models**: Run `train_models.py` to train and evaluate different models. This script will save the results in `model_results.md`.
   ```bash
   python scripts/train_models.py
   ```

5. **Decompose Time Series**: Run `decompose_time_series.py` to perform and visualize time series decomposition.
   ```bash
   python scripts/decompose_time_series.py
   ```

6. **Generate Final Predictions**: Run `main.py` to train the selected best model, generate predictions, and create plots.
   ```bash
   python scripts/main.py
   ```

## Results and Conclusions

### Model Performance

Despite considering seasonality, the ARIMA model did not yield the expected results. The Lasso model outperformed ARIMA significantly and was therefore used in `main.py` for final predictions.

### Potential Issues with ARIMA

The ARIMA model might have underperformed because it considered only one type of seasonality (weekly), while the data exhibited both weekly and yearly seasonality. This dual-seasonality is evident in the trend plots saved in `decomposition_plot.png` in the `results` folder.

### Why Lasso Performed Well

Lasso Regression (Least Absolute Shrinkage and Selection Operator) can handle high-dimensional data effectively and perform feature selection by driving some coefficients to zero. This ability to eliminate irrelevant features likely helped in reducing overfitting and improving the model's predictive performance, making it a better choice for this time series prediction task.

### Summary of Key Findings

- **ARIMA**: Struggled with capturing the complexities of the sales data due to its single-seasonality focus.
- **Lasso Regression**: Excelled by selecting the most relevant features and mitigating overfitting, thus delivering better predictions.

## Folder Structure

```bash
Rossmann-Time-Series-Analysis/
├── scripts/
│   ├── main.py
│   ├── train_models.py
│   ├── decompose_time_series.py
├── src/
│   ├── data.py
│   ├── plots.py
├── tests/
│   ├── test_data.py
├── results/
│   ├── decomposition_plot.png
│   ├── model_results.md
│   ├── lasso_model.joblib
│   ├── test_predictions.png
│   ├── train_test_predictions.png
```

By following this structure and methodology, the project systematically identifies the best model for predicting Rossmann store sales and provides insights into the underlying data patterns, enhancing the accuracy and reliability of the forecasts.
