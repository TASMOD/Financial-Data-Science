import logging

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import tensorflow as tf

import seaborn as sns
import xgboost as xgb

logging.root.setLevel(logging.INFO)

months = ["Month "+ str(x) for x in range(1,26)]
#Label the data according to the 25 monthly observations.
testset1A_df = pd.read_csv("../Data/testset3A.csv", names= months)
testset1B_df = pd.read_csv("../Data/testset3B.csv", names= months)
testset1C_df = pd.read_csv("../Data/testset3C.csv", names= months)
testset1D_df = pd.read_csv("../Data/testset3D.csv", names= months)
testset1E_df = pd.read_csv("../Data/testset3E.csv", names= months)
testset1_df = pd.concat([testset1A_df,testset1B_df, testset1C_df,testset1D_df,testset1E_df], ignore_index=True)
# Check that the data is according to spec. Each file has 16000 lines, merged to 80000 lines with 25 columns.
if not (len(testset1A_df) == len(testset1B_df) == len(testset1C_df) ==
      len(testset1D_df) == len(testset1E_df) == 16000
      and len(testset1_df.columns) == 25 and len(testset1_df) == 80000):
      logging.warning('Please make sure the dataset is as expected.')
#check for null values
logging.info(testset1_df.head())

#Remove the 'true' label
y = testset1_df['Month 25']
del testset1_df['Month 25']

plt.hist(testset1_df.to_numpy(), bins=25)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
corr = testset1_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, vmax=1, annot=True, square=True)
plt.title('feature correlations')

def evaluate_prediction(y_train, yhat_train, y_val, yhat_val):
    try:
        logging.info('RMSE Train: %.2f'
              % mean_squared_error(y_train, yhat_train, squared=True))
        logging.info('RMSE Validation: %.2f'
              % mean_squared_error(y_val, yhat_val, squared=True))
        return str(mean_squared_error(y_train, yhat_train, squared=True)), str(mean_squared_error(y_val, yhat_val, squared=True))
    except:
        logging.warning('Something went wrong' + sys.exc_info()[0])


# feature selectionn to check if any features just introduce noise.
X_norm = preprocessing.normalize(testset1_df.to_numpy())
X_stand = preprocessing.scale(X_norm)
X_train, X_test, y_train, y_test = train_test_split(X_stand, y,
                                                    train_size=0.7,
                                                    random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                    train_size=0.5,
                                                    random_state=42)
n_features = 1
n_steps = 24
patience = 10
X = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_eval = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)

#print(X_train[0], y_train[0])
dtrain = xgb.DMatrix(X_train, label=y_train, silent=True)
param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
param['nthread'] = 4
param['eval_metric'] = 'rmse'
dtest = xgb.DMatrix(X_val, label=y_val, silent=True)
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = patience
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10, verbose_eval=10)
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

''''
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features), dropout=0.25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')

# fit model
model.fit(X, y_train, epochs=200, verbose=1, callbacks=[callback])
model.save('SimpleLSTM.h5')
'''
model = tf.keras.models.load_model('/Users/vinushka/Dropbox/Work/N_test/Submission/3.SimpleLSTM.h5')
# demonstrate prediction
yhat_train = model.predict(X, verbose=0)
yhat_val = model.predict(X_eval, verbose=0)
trainmetric, evalmetric = evaluate_prediction(y_train, yhat_train, y_val, yhat_val)
