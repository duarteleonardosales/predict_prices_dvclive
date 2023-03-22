import numpy as np
import pandas as pd
from dvclive import Live
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

data = pd.read_csv('data/housing.csv')
data = data.values

X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# define model
model = XGBRegressor()
# fit model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
corr_pearson = np.corrcoef(y_test, y_pred)

with Live(save_dvc_exp=True) as live:
    live.log_metric("val/r2_score", r2)
    live.log_metric("val/r", corr_pearson[0][1])