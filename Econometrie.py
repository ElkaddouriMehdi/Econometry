import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from arch import arch_model
import numpy as np
from pmdarima.arima import ADFTest, auto_arima

""" please correct the imported data before introducing it , it's done in the jupyter notebook 
"""
class my_data:
    """ k is the percentage given to training data"""
    def __init__(self,data,k):
        self.my_data=data
        self.k = k
    def train_data(self):

        return(self.my_data[:int(self.k*len(self.my_data))])

    def test_data(self):

        return(self.my_data[int(self.k*len(self.my_data)):])
    """ the length of my training data is the int of the percentage given to it"""

    def length_train(self):

        return(int(self.k*len(self.my_data)))

    """ the length of my test data is the length of my_data ( whole data) - the length of the training data"""
    def length_test(self):

        return(len(self.my_data)-int(self.k*len(self.my_data)))

    def plot_my_data(self):
        return(plt.plot(self.my_data))

    def plot_acf(self):

        return(plot_acf(self.train_data()))

    def plot_pacf(self):

        return(plot_pacf(self.train_data()))

    def adf_test(self):
        adf_test = ADFTest(alpha=0.05) # adf test for alpha = 0.05

        return(adf_test.should_diff(self.my_data))

    def std_data(self):

        return(self.my_data.std())

    def arima_auto(self):
        return(auto_arima(self.train_data(),start_p=1, d=0, start_q=1,
                          max_p=5, max_d=2, max_q=5, start_P=0,
                          D=0, start_Q=0, max_P=3, max_D=3,
                          max_Q=3, m=1, seasonal=True,
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = True,
                          random_state=20,n_fits = 10 ))
    def arima_summary(self):
        return(self.arima_auto().summary())


class models(my_data):

    def __init__(self,data,k):
        my_data.__init__(self,data,k)




    """ i forecast for each i
        the p and q given from the graph of pacf and acf must be the input of my classes
        the heritage allow me the explore the data caracteristics for each model
    """

    def model_forcast(self,p,q):
        garch_forecast=[]      #list to inialize
        for i in range (self.length_test()):
            my_data_train= self.my_data[:self.length_train()+i]  # data [train_data:end]
            my_model= arch_model(my_data_train,p= p,q=q,vol='Garch')
            my_fit = my_model.fit(disp='off') # fitting my model garch for p and q ( in my case p=2 & q = 2)
            my_prediction=my_fit.forecast() # let's apply the prediction
            garch_forecast.append(np.sqrt(my_prediction.variance.values[-1:][0]))  # the prediction results
        garch_forecast=pd.Series(garch_forecast,index=self.my_data.index[self.length_train():]) # construct of my serie using my_data index  in order to plot it
        return(garch_forecast)
    """ i can see that my garch(2,2) model explains perfectly the variation of returns
        
    """
    def model_sarimax(self,p,d,q,P,D,Q,s):
        ts=self.my_data
        sarima_model=sm.tsa.statespace.SARIMAX(ts,order=(p, d, q),seasonal_order=(P, D, Q, s),enforce_stationarity=True,enforce_invertibility=True)
        sarima_model_fit=sarima_model.fit() # fit model to data
        sarima_forecast=sarima_model_fit.get_prediction(start=self.length_train(),end=len(self.my_data))
        # return in-sample-prediction
        return(sarima_forecast)

    def model_arima(self):
        model_arima=self.arima_auto()
        prediction = pd.DataFrame(model_arima.predict(n_periods=727), index=self.test_data().index)
        prediction.columns = ['predicted_price']
        return(prediction)














