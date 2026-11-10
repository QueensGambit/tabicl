from tabicl import TabICLRegressor
import numpy as np

X_train = np.zeros((20,20))
y_train = np.zeros((20,))

model = TabICLRegressor(model_path="~/Programming/forks/tabicl/checkpoints/official_regression/tabicl-regressor-v2-20260212.ckpt")

model.fit(X_train, y_train)

