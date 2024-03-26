# Data processing libraries
import datetime
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

# Data processing
from sklearn.preprocessing import MinMaxScaler

# Model and measure metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, f1_score

#Model
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

import yfinance as yf

import os
import json

# LSTM model grouping together all the preprocessing,
# model building and evaluating techniques found in model research.
# Generalisation to fit any historical stock data gotten from yahoo finance.
# interesting to see how different stocks differ under same preprocessing and modelling
# techniques.
# Very ad hoc implementation, could improve design using SOLID principles to make more
# robust, scalable and ready for change.

class LSTM_V3_Model:

    sc_single = MinMaxScaler(feature_range=(0,1))
    sc_mult = MinMaxScaler(feature_range=(0,1))

    def __init__(self, stock_ticker, start, end, n_epochs, batch_size, look_back):
        self.stock_ticker = stock_ticker
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.look_back = look_back
        self.df = LSTM_V3_Model.get_data(stock_ticker, start, end)
        self.preprocess_data()
        
        if os.path.isfile("./saved_models/LSTMV3/"+stock_ticker+".keras"):
            self.model = load_model("./saved_models/LSTMV2/"+stock_ticker+".keras")
            self.history = self.load_history()
        else:
            self._build_model()

    def get_data(ticker, start, end):
        data =  yf.download(ticker, progress=True, actions=True,start=start, end=end)
        data = pd.DataFrame(data)
        data.rename(columns = {'Adj Close':ticker}, inplace=True)
        data.drop(['Close', 'Dividends', 'Stock Splits'], axis=1, inplace=True)
        data.dropna(inplace=True)
        
        return data
    
    
    def preprocess_data(self):
        self.test = self.df[self.stock_ticker][-261:]
        LSTM_V3_Model.sc_single.fit_transform(np.array(self.test).reshape(-1, 1))

        for leng in [10,20,30]:
            sma = ta.sma(self.df[self.stock_ticker], length=leng)
            self.df['SMA_'+f'{leng}'] = [1 if sma[i] < self.df[self.stock_ticker][i] else 0 for i in range(len(sma))]

        self.df['RSI_'+f'{14}'] = ta.rsi(self.df[self.stock_ticker])
        stoch = ta.stoch(self.df['High'], self.df['Low'], self.df[self.stock_ticker])
        stochk = stoch['STOCHk_14_3_3']
        self.df = self.df[13:]
        self.df['Stoch_Oscillator'] = stochk

        williams_r = ta.willr(self.df['High'], self.df['Low'], self.df[self.stock_ticker])
        self.df['Williams_%R'] = williams_r

        adx = ta.adx(self.df['High'], self.df['Low'], self.df[self.stock_ticker])
        self.df['ADX'] = adx.ADX_14
        self.df['returns'] = np.log(self.df[self.stock_ticker] / self.df[self.stock_ticker].shift(1))
        self.df.dropna(inplace=True)
        self.df['direction'] = [1 if self.df.returns[i]>0 else 0 for i in range(len(self.df))]
        self.features = [col for col in self.df.columns if col not in [self.stock_ticker, 'returns', 'Open', 'High', 'Low', 'Volume']]
        self.num_features = len(self.features)
        self.train_dates = self.df[self.features][:-261].index
        self.test_dates = self.df[self.features][-261:].index
        df_np = np.array(self.df[self.features].dropna())
        df_scaled = LSTM_V3_Model.sc_mult.fit_transform(df_np)
        X, Y = self.shape_data(df_scaled)
        self.X_train, self.y_train = X[:-261], Y[:-261]
        self.X_test, self.y_test = X[-261:], Y[-261:]

    def shape_data(self, data):
        X = []
        Y = []
        for i in range(self.look_back, len(data)):
            X.append([arr[:self.num_features] for arr in data[i-self.look_back:i]])
            Y.append(data[i][len(data[i])-1])

        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], self.num_features))
        return X, Y
    
    def _build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=100,return_sequences=True,batch_input_shape=(self.batch_size, self.look_back, self.num_features)))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(units=100,return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(units=100,return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(units=100))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.fit()

    def fit(self):
        self.impl = self.model.fit( 
                        self.X_train,
                        self.y_train,
                        epochs=self.n_epochs
                    )
        self.history = self.impl.history
        self.save_history()
        self.model.save("./saved_models/LSTMV3/"+self.stock_ticker+".keras")
    
    def predict(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.array([1 if prob >= 0.5 else 0 for prob in y_pred])
        y_true = self.y_test
        self.y_pred = y_pred
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)


    def measure_classification(self):
        target_names = ['direction_up', 'direction_down']

        y_true = self.y_test
        y_pred = self.y_pred
        print(f'====== {self.stock_ticker} Classification report ======')
        print(classification_report(y_true, y_pred, target_names=target_names))
        print(confusion_matrix(y_true, y_pred, labels=[1, 0]))

    
    def load_history(self):
        history_dir = "./saved_models/LSTMV3/"
        history_file_path = history_dir + self.stock_ticker + "_history.json"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        with open(history_file_path, 'r') as f:
            loaded_history = json.load(f)
        return loaded_history

    def save_history(self):
        history_dir = "./saved_models/LSTMV3/"
        history_file_path = history_dir + self.stock_ticker + "_history.json"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        with open(history_file_path, 'w') as f:
            json.dump(self.history, f)
    
