# self.train processing libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# self.train processing
from sklearn.preprocessing import StandardScaler

# Model and measure metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

#Model
from sklearn import linear_model
import os
import pickle


#   The only change from Logistic_Regression_Model is that the target variable "direction"
#   is shifted by -1 i.e all days shifted to previous day so that now "direction" is
#   1 if the price will go up tomorrow and 0 otherwise instead of 1 if the price
#   went up from yesterday and 0 otherwise

class Logistic_Regression_Model_V2:

    sc = StandardScaler(with_mean=True)

    def __init__(self, ticker, start, end):
        self.ticker = ticker
        data = Logistic_Regression_Model_V2.get_data(ticker, start, end)
        split_index = int(0.7*len(data))
        self.train = data[0:split_index]
        self.test = data[split_index:]
        self.features = []
        self.train = self.preprocess_data(self.train)
        self.test = self.preprocess_data(self.test)
        self.strategy_rtn = []
        if os.path.isfile("./saved_models/LogRegV2/"+ticker+".pkl"):
            with open("./saved_models/LogRegV2/"+ticker+".pkl", "rb") as file:
                self.model = pickle.load(file)
            with open("./saved_models/LogRegV2/"+ticker+"_features.pkl", "rb") as file:
                self.features = pickle.load(file)
        else:
            self.feature_selection()
            self._build_model()
            self.fit()

        self.predict()

    def get_data(ticker, start, end):
        data =  yf.download(ticker, progress=True, actions=True,start=start, end=end)
        data = pd.DataFrame(data)
        data.rename(columns = {'Adj Close':ticker}, inplace=True)
        data.drop(['Close', 'Dividends', 'Stock Splits'], axis=1, inplace=True)
        data.dropna(inplace=True)
        
        return data
    
    def preprocess_data(self, data):
        for leng in [10,20,30]:
            sma = ta.sma(data[self.ticker], length=leng)
            data['SMA_'+f'{leng}'] = sma

        for leng in [10,20,30]:
            ema = ta.ema(data[self.ticker], length=leng)
            data['EMA_'+f'{leng}'] = ema

        for leng in [14, 28]:
            data['RSI_'+f'{leng}'] = ta.rsi(data[self.ticker], length=leng)

        stoch = ta.stoch(data['High'], data['Low'], data[self.ticker])
        stochk = stoch['STOCHk_14_3_3']
        data = data[13:]
        data['Stoch_Oscillator'] = stochk

        williams_r = ta.willr(data['High'], data['Low'], data[self.ticker])
        data['Williams_%R'] = williams_r

        adx = ta.adx(data['High'], data['Low'], data[self.ticker])
        data['ADX'] = adx['ADX_14']

        data['returns'] = np.log(data[self.ticker] / data[self.ticker].shift(1)).shift(-1)
        data.dropna(inplace=True)
        data['direction'] = [1 if data.returns[i]>0 else 0 for i in range(len(data))]

        
        features_for_scaling = [col for col in data.columns if col not in ['returns', 'direction']]
        self.features = features_for_scaling
        data[features_for_scaling] = Logistic_Regression_Model_V2.sc.fit_transform(data[features_for_scaling])
        
        return data.dropna(axis=0)

    
    def _build_model(self):
        x = self.train[self.features]
        y = self.train['direction']
        param_selector = linear_model.LogisticRegression()
        param_grid = [    
            {'penalty' : ['l1', 'l2', 'none'],
            'C' : np.logspace(-4, 4, 20),
            'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
            }
        ]
        model_selector = GridSearchCV(param_selector, param_grid = param_grid, cv = 5, n_jobs=-1)
        model_selector.fit(x,y)
        self.model = model_selector.best_estimator_

    def fit(self):
        self.model.fit(self.train[self.features], self.train['direction'])
        with open("./saved_models/LogRegV2/"+self.ticker+".pkl", 'wb') as file:
            pickle.dump(self.model, file)

    def feature_selection(self):
        x = self.train[self.features]
        y = self.train['direction']
        feature_selector = linear_model.LogisticRegression()

        efs = EFS(estimator=feature_selector, 
                min_features=1,
                max_features=5,
                scoring='accuracy',
                cv=5).fit(x, y)

        print('Best features:', efs.best_feature_names_)
        self.features = list(efs.best_feature_names_)        
        with open("./saved_models/LogRegV2/"+self.ticker+"_features.pkl", 'wb') as file:
            pickle.dump(self.features, file)

    def predict(self):
        self.test['preds'] = self.model.predict(self.test[self.features])
    
    def get_metrics(self):
        y_true = self.test['direction']
        y_pred = self.test['preds']
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)

    def measure_classification(self):
        target_names = ['direction_up', 'direction_down']

        y_true = self.test['direction']
        y_pred = self.test['preds']
        print(f'====== {self.ticker} Classification report ======')
        print(classification_report(y_true, y_pred, target_names=target_names))
        print(confusion_matrix(y_true, y_pred, labels=[1, 0]))

    def get_roc_curve(self):
        y_true = self.test['direction']
        y_pred_proba = self.model.predict_proba(self.test[self.features])[::,1]

        fpr, tpr, _ = roc_curve(y_true,  y_pred_proba)
        return pd.DataFrame({"fpr":fpr, "tpr":tpr})