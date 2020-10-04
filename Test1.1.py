# -*- coding: utf-8 -*-
"""Test 1:
This module performs a forecasting operation on the dataset 1,
we use a boosted tree as the other models
performed much worse."""

import math
import logging
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import xgboost as xgb

#Import and set up our logger.
logging.root.setLevel(logging.INFO)
# Establish the constants.
PATIENCE = 10
RANDOM_SEED = 42

#Import and confirm the data:
months = ["Month "+ str(x) for x in range(1,26)]
#Label the data according to the 25 monthly observations (for human ease).
testset1A_df = pd.read_csv("./Data/testset1A.csv", names= months)
testset1B_df = pd.read_csv("./Data/testset1B.csv", names= months)
testset1C_df = pd.read_csv("./Data/testset1C.csv", names= months)
testset1D_df = pd.read_csv("./Data/testset1D.csv", names= months)
testset1E_df = pd.read_csv("./Data/testset1E.csv", names= months)
testset1_df = pd.concat([testset1A_df, testset1B_df, testset1C_df, testset1D_df,
                        testset1E_df], ignore_index=True)
#Check that the data is according to spec:
#    having 16000 lines, merged to 80000 lines with 25 columns.
if not (len(testset1A_df) == len(testset1B_df) == len(testset1C_df) ==
        len(testset1D_df) == len(testset1E_df) == 16000
        and len(testset1_df.columns) == 25 and len(testset1_df) == 80000):
    logging.warning('Please make sure the dataset is as expected.')

#Remove the 'true' label
y = testset1_df['Month 25']
del testset1_df['Month 25']

# Preprocess the data and split into subsets.
X_norm = preprocessing.normalize(testset1_df.to_numpy())
X_stand = preprocessing.scale(X_norm)

X_train, X_test, y_train, y_test = train_test_split(X_stand, y,
                                                    train_size=0.7,
                                                    random_state=RANDOM_SEED)
# Create Kfolds to train our model on subsets
# of training instead of a validation set.
# then gridearch the best variables, train and evaluate our model.
logging.info("Parameter optimization:")
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2, 4, 6],
                    'n_estimators': [50, 100, 200]}, verbose=1, n_jobs=4)
kf = KFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
for train_index, test_index in kf.split(X_train):
    xgb_model = clf.fit(np.array(X_train)[train_index],
                                      np.array(y_train)[train_index])
    y_pred = xgb_model.predict(np.array(X_train)[test_index])
    y_true = np.array(y_train)[test_index]
    logging.info('Base RMSE for fold: %.2f',
                 math.sqrt(mean_squared_error(y_true, y_pred)))

logging.info('Best score: %.2f', clf.best_score_)
logging.info('Best parameters %s', clf.best_params_)
pred_test = clf.predict(np.array(X_test))
logging.info('RMSE Final Test: %.2f', mean_squared_error(y_test, pred_test,
                                                         squared=True))
