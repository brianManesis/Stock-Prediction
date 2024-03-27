# Final_Year_Project

## Name
Stock price and price movement prediction with Machine and Deep Learning

## Description
Stock price and price movement prediction using LSTM neural networks for stock price prediction and Logistic Regression and LSTM for predicting stock price movements of various stocks.

## Folder Contents
The top level of the repository contains 5 directories: 

The directory called analysis is where the initial analysis on the data gotten from the yahoo finance API was done. The Initial_Analysis.ipynb notebook collects stock data for Google, Amazon, Meta, Apple and Microsoft stocks and extracts simple statistics such as correlation between stock prices and descriptive statistics. The Google_Analysis.ipynb notebook performs further exploratory analysis on the Google stock data, including daily returns and stationarity analysis. 

The directory called model_analysis is where the modelling and feature engineering investigation was performed for each of the models implemented. The models were all trained on Google stock data. The LSTM_Analysis.ipynb notebook contains a single feature, stacked LSTM for predicting Googles stock price. The LSTM_V2_Analysis.ipynb notebook contains a multi-feature feature, stacked LSTM for predicting Googles stock price. The Logisitic_Regression_Analysis.ipynb notebook contains a logistic regression model for predicting the stock price movements of the Google stock price. The LSTM_V3_Analysis.ipynb notebook contains an LSTM classifier model to predict the stock price movements of the Google stock price. The Trading_As_Classification_Problem.ipynb notebook was used to investigate classification models for predicting stock price movements.

The directory called models contains class wrappers with generalizations of the models from the model_analysis directory. They can be generalized to any stock data from a yahoo finance API endpoint. 

The results directory contains the measurements gotten from each of the models in the model's directory trained on each of the stocks selected in the inital_analysis.ipynb notebook. 

The tests directory contains tests for the shape_data function used in the LSTM models. 

The linux_environment.yml file is used during the installation process on a Linux machine with anaconda installed to recreate the environment used for this project. 
All the notebooks can be run in jupyter notebooks with the environment specified in the linux_environment.yml

## Installation
Linux:
conda env create -n project_env -f=linux_environment.yml

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Authors and acknowledgment


## Disclaimer
Use at your own Risk The stock price and price movement prediction models used are provided for educational and research purposes. The models used may contain inaccuracies or uncertainties. Please be aware of the risks associated with stock market investments, and exercise caution when using predictive models for financial decision-making.
