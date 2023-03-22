# import pandas as pd
# from numpy import asarray
# from xgboost import XGBRegressor

# from dvclive import Live

# data = pd.read_csv('data/housing.csv')
# data = data.values


# with Live(save_dvc_exp=True) as live:
#     live.log_metric("r2", .89898977)

# X, y = data[:, :-1], data[:, -1]
# # define model
# model = XGBRegressor()
# # fit model
# model.fit(X, y)
# # define new data
# row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
# new_data = asarray([row])
# # make a prediction
# yhat = model.predict(new_data)
# # summarize prediction

# print('Predicted: %.3f' % yhat)