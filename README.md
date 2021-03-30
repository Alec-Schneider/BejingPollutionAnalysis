
# Air Pollution Forecasting Using A Tranfomer Neural Network

Data provdided by [UCI](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data#).

See [full_pipeline_model.ipynb](./full_pipeline_model.ipynb) for the following:
- Data pipeline transformation
- Splitting training and test files
- Build Transformer model using [PyTorch](https://pytorch.org/)
- Train model
- Save model
- Forecast entire test set
- Run cost projection on air pollution fines


## Daily Heatmap of Particle Matter 2.5 Concentrations
![Daily Heatmap](/.images/daily_heatmap.png)


## Transformer's Forecasted Rolling Mean Particle Matter 2.5 Concentrations
![Forecasted Rolling Means](/.images/rolling_mean_preds_vs_thresholds.png)

