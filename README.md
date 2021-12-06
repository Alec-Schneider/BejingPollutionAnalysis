# Air Pollution Forecasting Using A Transfomer Neural Network
Semester long IST707 - Data Analytics project

### Project Goals
- The objective of the project is to use the main skills taught in this class to solve a real data mining problem
- For this project, you must choose your own dataset. It can be one that you created yourself or found from other resources, such as the Kaggle competitions and the [UCI repository](http://archive.ics.uci.edu/ml/).
- Define a problem on the dataset and describe it in terms of its real-world organizational or business application. The complexity level of the problem should be comparable to homework assignments.

Data provdided by [UCI](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data#).

### See [full_pipeline_model.ipynb](./full_pipeline_model.ipynb) for the following:
- Data pipeline transformation
- Splitting training and test files
- Build Transformer model using [PyTorch](https://pytorch.org/)
- Train model
- Save model
- Forecast entire test set
- Run cost projection on air pollution fines

### Other Files and Scripts:
- ts_transformer.py: Time Series Transformer neural network architecture
- torch_utils.py: Custom module of PyTorch helper functions, PyTorch class of the pollution dataset, and wrapper class to allow ts_transformer to be used on a Scikit-learn pipeline
- sklearn_utils.py: Custom Transformer steps for a Scikit-learn pipeline and a pipeline creation function (bejing_pipeline)
- preprocessor.py: Kalman Filtering (Preprocessor class) class to impute missing data in the datasets during the processing of data
- process_data.py: command line script to impute missing data and clean the data
- tseries.R: Time Series plotting and analysis of sample data
- tseries_eda.Rmd: extensive EDA and time series plotting and forecasting 

### Software Requirements of the Project
For all Python files (.ipynb abd .py extensions):
- Custom modules:
    - torch_utils 
    - ts_transformer
    - preprocessor
- PyTorch
- PathLib
- NumPy
- Pandas
- PyKalman
- Scikit-learn
- Matplotlib
- Multiprocessing
- os
- glob
- click

For all R scripts:
- tseries
- TSstudio
- forecast
- xts
- tidyverse


## Daily Heatmap of Particle Matter 2.5 Concentrations

![Daily Heatmap](/.images/daily_heatmap.png)

## Transformer's Forecasted Rolling Mean Particle Matter 2.5 Concentrations

![Forecasted Rolling Means](/.images/rolling_mean_preds_vs_thresholds.png)
