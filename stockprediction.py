__author__ = 'Prathmesh'

import os
from os import path
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

print("Welcome to stock prediction!!")
dates = []
prices = []

def get_data(file_path):
    fp = open(file_path)
    csvFileReader = csv.reader(fp)
    next(csvFileReader)

    for row in csvFileReader:
        #reads row as string
        dates.append(int(row[0].split('-')[0]))
        prices.append(float(row[1]))

    fp.close()
    return

def predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))

    svr_lin = SVR(kernel="linear",C=1e3)
    svr_poly = SVR(kernel="poly",C=1e3,degree=2)
    svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)

    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)

    plt.scatter(dates,prices,color='black',label='Data')

    plt.plot(dates,svr_lin.predict(dates),color='red',label='SVR_Linear')
    plt.plot(dates,svr_poly.predict(dates),color='green',label='SVR_poly')
    plt.plot(dates,svr_rbf.predict(dates),color='blue',label='SVR_rbf')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('SVR')

    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0],svr_rbf.predict(x)[0],svr_poly.predict(x)[0]

FILE_PATH = path.join(os.pardir, "Stockprediction/Resources/aapl.csv")

get_data(FILE_PATH)

print(predict_prices(dates,prices,10))






