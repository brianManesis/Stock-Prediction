# Data processing libraries
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data processing
from sklearn.preprocessing import MinMaxScaler

# Model and measure metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

class LSTM_Model:

    sc = MinMaxScaler(feature_range=(0,1))

    def __init__(self, stock_ticker, start, end, n_epochs, batch_size, look_back, num_features):
        self.stock_ticker = stock_ticker
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.look_back = look_back
        self.num_features = num_features
        self.df = LSTM_Model.get_data(stock_ticker, start, end)
        self.preprocess_data()
        
        if os.path.isfile("./saved_models/LSTMV1/"+stock_ticker+".keras"):
            self.model = load_model("./saved_models/LSTMV1/"+stock_ticker+".keras")
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
        self.train_dates = self.df[self.stock_ticker][:-261].index
        self.test_dates = self.df[self.stock_ticker][-261:].index
        df_np = np.array(self.df[self.stock_ticker])
        df_np = df_np.reshape(-1,1)

        df_scaled = LSTM_Model.sc.fit_transform(df_np)

        X, Y = self.shape_data(df_scaled)
        self.X_train, self.y_train = X[:-261], Y[:-261]
        self.X_test, self.y_test = X[-261:], Y[-261:]

    def shape_data(self, data):
        X = []
        Y = []
        for i in range(self.look_back, len(data)):
            X.append(data[i-self.look_back:i, 0])
            Y.append(data[i, 0])
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
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
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam',loss='mean_squared_error')
        self.fit()

    def fit(self):
        self.impl = self.model.fit( 
                        self.X_train,
                        self.y_train,
                        validation_split=0.33,
                        epochs=self.n_epochs,
                    )
        self.history = self.impl.history
        self.save_history()
        self.model.save("./saved_models/LSTMV1/"+self.stock_ticker+".keras")
    
    def predict(self):
        predictions_scaled = self.model.predict(self.X_test)
        predictions = LSTM_Model.sc.inverse_transform(predictions_scaled)
        self.predictions = pd.DataFrame(predictions, index=self.test_dates)[0]
        rmse = np.sqrt(np.mean(((np.array(self.predictions) - np.array(self.test)) ** 2)))
        mae = mean_absolute_error(self.test, self.predictions)
        self.mae = mae
        self.rmse = rmse
        return mae, rmse

    def load_history(self):
        history_dir = "./saved_models/LSTMV1/"
        history_file_path = history_dir + self.stock_ticker + "_history.json"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        with open(history_file_path, 'r') as f:
            loaded_history = json.load(f)
        return loaded_history

    def save_history(self):
        history_dir = "./saved_models/LSTMV1/"
        history_file_path = history_dir + self.stock_ticker + "_history.json"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        with open(history_file_path, 'w') as f:
            json.dump(self.history, f)
    
    def get_validation_loss(self):
        val_loss_df = pd.DataFrame({"loss":self.history['loss'], "val_loss": self.history['val_loss']})
        return val_loss_df

    def get_test(self):
        test_df = pd.DataFrame({"test":self.test, "predictions":self.predictions})
        return test_df
