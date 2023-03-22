import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from dvclive import Live

data = pd.read_csv('data/housing.csv')
data = data.values
TEST_SIZE = 0.28
SEED = 12
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

# define model
model = XGBRegressor()
# fit model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
corr_pearson = np.corrcoef(y_test, y_pred)

with Live(save_dvc_exp=True) as live:
    live.log_param('test_size', TEST_SIZE)
    live.log_param('seed', SEED)
    live.log_metric("val/r2_score", r2)
    live.log_metric("val/r", corr_pearson[0][1])
live.end()