# Data processing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from datetime import datetime

# Data processing
from sklearn.preprocessing import MinMaxScaler

# Model and measure metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Model
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

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

    def __init__(self, stock_ticker, n_epochs, batch_size, look_back, num_features):
        self.stock_ticker = stock_ticker
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.look_back = look_back
        self.num_features = num_features
        self.df = pd.read_csv("./datasets/"+stock_ticker+".csv")
        self.preprocess_data()
        
        if os.path.isfile("./models/"+stock_ticker+".keras"):
            self.model = load_model("./models/"+stock_ticker+".keras")
            self.history = self.load_history()
        else:
            self._build_model()
    
    def preprocess_data(self):
        self.df.Date = pd.to_datetime(self.df['Date'])
        self.df = self.df.set_index('Date')
        self.train = self.df.Close[:-261]
        self.test = self.df.Close[-261:]
        self.train_dates = self.train.index
        self.test_dates = self.test.index
        df_np = np.array(self.df.Close)
        df_np = df_np.reshape(-1,1)

        df_scaled = LSTM_Model.sc.fit_transform(df_np)

        X, Y = self.shape_data(df_scaled)
        self.X_train, self.y_train = X[:-261], Y[:-261]
        self.X_test, self.y_test = X[-261:], Y[-261:]

    def shape_data(self, scaled_np):
        X = []
        Y = []
        for i in range(self.look_back, len(scaled_np)):
            X.append(scaled_np[i-self.look_back:i, 0])
            Y.append(scaled_np[i, 0])
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, Y
    
    def _build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=100,return_sequences=True,input_shape=(self.look_back, self.num_features)))
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
                        batch_size=self.batch_size
                    )
        self.history = self.impl.history
        self.save_history()
        self.model.save("./models/"+self.stock_ticker+".keras")
    
        
    def predict(self):
        predictions_scaled = self.model.predict(self.X_test)
        predictions = LSTM_Model.sc.inverse_transform(predictions_scaled)

        self.predictions = pd.DataFrame(predictions, index=self.test_dates)
        rmse = np.sqrt(np.mean(((self.predictions - self.test) ** 2)))
        print("RMSE: ", rmse)

    def load_history(self):
        history_file_path = "./models/"+self.stock_ticker+"_history.json"
        with open(history_file_path, 'r') as f:
            loaded_history = json.load(f)
        return loaded_history

    def save_history(self):
        history_file_path = "./models/"+self.stock_ticker+"_history.json"
        with open(history_file_path, 'w') as f:
            json.dump(self.history, f)
    
    def plot_validation_loss(self):
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def plot_test(self):
        figure, axes = plt.subplots( 1 ) 

        axes.plot(self.test, color = 'black', label = 'Google Stock Price')
        axes.plot(self.predictions, color = 'red', label = 'Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')

        plt.legend()
        plt.show()


model = LSTM_Model('NFLX', n_epochs=100, batch_size=60, look_back=32, num_features=1)


model.predict()

model.plot_validation_loss()
model.plot_test()