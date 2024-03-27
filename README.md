# Final_Year_Project

## Name
Stock price and price movement prediction with Machine and Deep Learning

## Description
Stock price and price movement prediction using LSTM neural networks for stock price prediction and Logistic Regression and LSTM for predicting stock price movements of various stocks.

## Contents
The top level of the repository contains 5 directories: 

The directory called analysis is where the initial analysis on the data gotten from the yahoo finance API was done. The Initial_Analysis.ipynb notebook collects stock data for Google, Amazon, Meta, Apple and Microsoft stocks and extracts simple statistics such as correlation between stock prices and descriptive statistics. The Google_Analysis.ipynb notebook performs further exploratory analysis on the Google stock data, including daily returns and stationarity analysis. 

The directory called model_analysis is where the modelling and feature engineering investigation was performed for each of the models implemented. The models were all trained on Google stock data. The LSTM_Analysis.ipynb notebook contains a single feature, stacked LSTM for predicting Googles stock price. The LSTM_V2_Analysis.ipynb notebook contains a multi-feature feature, stacked LSTM for predicting Googles stock price. The Logisitic_Regression_Analysis.ipynb notebook contains a logistic regression model for predicting the stock price movements of the Google stock price. The LSTM_V3_Analysis.ipynb notebook contains an LSTM classifier model to predict the stock price movements of the Google stock price. The Trading_As_Classification_Problem.ipynb notebook was used to investigate classification models for predicting stock price movements.

The directory called models contains class wrappers with generalizations of the models from the model_analysis directory. They can be generalized to any stock data from a yahoo finance API endpoint. 

The results directory contains the measurements gotten from each of the models in the model's directory trained on each of the stocks selected in the inital_analysis.ipynb notebook. Stocks to train the models on can be selected by altering the tickers list to the stocks you wish.

The tests directory contains tests for the shape_data function used in the LSTM models. 

The linux_environment.yml file is used during the installation process on a Linux machine with anaconda installed to recreate the environment used for this project. 
All the notebooks can be run in jupyter notebooks with the environment specified in the linux_environment.yml

## Installation
Linux is required as tensorflow on windows lacks some of the functionality used in this project.
Anaconda is required for recreating the environment used in this project.
To install need to open an anaconda terminal with the conda activate command and create a new conda environment from the linux_environment.yml file. need to activate this anaconda environment and open jupyter notebook to use the new project_env environment as a kernal.

To install enter following commands:

conda activate
conda env create -n project_env -f=linux_environment.yml
conda activate project_env
jupyter notebook

Open the project directory in jupyter notebook to run any of the notebooks.

## Usage
Model wrapper classes in the models directory can be used on any stock data from a yahoo finance API endpoint. The functionality of these classes can be seen in the results directory. Click on the *_Results.ipynb notebook, where * is the model you wish to use. Select the stocks you wish to test the model on by altering the tickers list to the stocks you wish. Each of the notebooks in the model_analysis directory can be run independently once the correct environment is set up (See installation).  

## Author
Brian Manesis

## Acknowledgments
I, Brian Manesis, would not have been able Project would not have been possible without the help and guidance of my supervisor. I want to thank Prof Rozenn Dahyot steering the project in the right direction and giving me feedback on what I was doing right and where to focus my efforts.  

I would like to express my appreciation to my family and friends for their support, encouragement and understanding. Their belief in my abilities have been a source of motivation during the course of this project. Thank you for being there to lift my spirits. 

I would also like to acknowledge Maynooth university for providing resources and facilities which proved to be critical to the development of this project. 

Thank you all for being a part of this project. 

## Disclaimer
Use at your own Risk The stock price and price movement prediction models used are provided for educational and research purposes. The models used may contain inaccuracies or uncertainties. Please be aware of the risks associated with stock market investments, and exercise caution when using predictive models for financial decision-making.
