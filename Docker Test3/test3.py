# -*- coding: utf-8 -*-
"""Test 3:
This module performs a forecasting operation on the dataset 3,
we use a boosted tree as a baseline model as it worked with
the previous data, and then create an LSTM to predict based on
the correlation between recent features as shown in the
correlation matrix."""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import tensorflow as tf
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense

#Import and set up our logger.
logging.root.setLevel(logging.INFO)

# Establish the constants.
N_FEAT = 1
N_STEPS = 24
PATIENCE = 10

#Import and confirm the data:
months = ["Month "+ str(x) for x in range(1,26)]
#Label the data according to the 25 monthly observations (for human ease).
testset3A_df = pd.read_csv("./Data/testset3A.csv", names= months)
testset3B_df = pd.read_csv("./Data/testset3B.csv", names= months)
testset3C_df = pd.read_csv("./Data/testset3C.csv", names= months)
testset3D_df = pd.read_csv("./Data/testset3D.csv", names= months)
testset3E_df = pd.read_csv("./Data/testset3E.csv", names= months)
testset3_df = pd.concat([testset3A_df,testset3B_df, testset3C_df,
                         testset3D_df,testset3E_df], ignore_index=True)
#Check that the data is according to spec:
#    having 16000 lines, merged to 80000 lines with 25 columns.
if not (len(testset3A_df) == len(testset3B_df) == len(testset3C_df) ==
        len(testset3D_df) == len(testset3E_df) == 16000
        and len(testset3_df.columns) == 25 and len(testset3_df) == 80000):
    logging.warning('Please make sure the dataset is as expected.')

#Remove the 'true' label
y = testset3_df['Month 25']
del testset3_df['Month 25']

def evaluate_prediction(true_train, pred_train, true_val, pred_val,
                        true_test = None, pred_test = None):
    """Returns the evaluation metric for the given predicions vs true values.
    Parameters:
    true_train (numpy.ndarray): The true values of y for the training data.
    pred_train (numpy.ndarray): The predicted values of y for the
        training data as given by the network.
    true_val (numpy.ndarray): The true values of y for the validation data.
    pred_val (numpy.ndarray): The predicted values of y for the
        validation data as given by the network.
    true_test (numpy.ndarray): Optional: The true values of y for the unseen data.
    pred_test (numpy.ndarray): The predicted values of y for the
        unseen data as given by the network.
    """

    logging.info('RMSE Train: %.2f', mean_squared_error(true_train, pred_train,
                                                        squared=True))
    logging.info('RMSE Validation: %.2f', mean_squared_error(true_val, pred_val,
                                                             squared=True))
    if not true_test is None and not pred_test is None:
        logging.info('RMSE Final Test: %.2f', mean_squared_error(true_test,
                                                                 pred_test,
                                                                 squared=True))

def format_input_for_network(data, features):
    """Function to convert the default data into the shape required by the LSTM.

    Parameters:
        data (numpy.ndarray): The data we want to use as input for the LSTM.
        features (int): The number of features we have to work with.

    Returns:
        (numpy.ndarray): The reshaped array for the LTSM.
    """
    return data.reshape((data.shape[0], data.shape[1], features))

# Preprocess the data and split into subsets.
X_norm = preprocessing.normalize(testset3_df.to_numpy())
X_stand = preprocessing.scale(X_norm)
X_train, X_test, y_train, y_test = train_test_split(X_stand, y,
                                                    train_size=0.7,
                                                    random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                train_size=0.5,
                                                random_state=42)
X = format_input_for_network(X_train, N_FEAT)
X_eval = format_input_for_network(X_val, N_FEAT)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=PATIENCE,
                                            restore_best_weights=True)

# Create our baseline model: a gradient boosted tree.
dtrain = xgb.DMatrix(X_train, label=y_train, silent=True)
param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
param['nthread'] = 4
param['eval_metric'] = 'rmse'
dtest = xgb.DMatrix(X_val, label=y_val, silent=True)
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, PATIENCE, evallist, early_stopping_rounds=10, verbose_eval=10)
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

#For ease of use and reproducability I have saved the model,
# but here is the code to create it from scratch.
'''
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(N_STEPS, N_FEAT), dropout=0.25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')

# fit model
model.fit(X, y_train, epochs=200, verbose=1, callbacks=[callback])
model.save('SimpleLSTM.h5')
'''
model = tf.keras.models.load_model('./3.SimpleLSTM.h5')
# use prediction to see how well trained and generalizabel the model is.
yhat_train = model.predict(X, verbose=0)
yhat_val = model.predict(X_eval, verbose=0)

# when satisfied with the metrics above,
# complete the unseen prediction as a final evaluation.
X_final_test = format_input_for_network(X_test, N_FEAT)
yhat_test = model.predict(X_final_test, verbose=0)

evaluate_prediction(y_train, yhat_train, y_val, yhat_val, y_test, yhat_test)
