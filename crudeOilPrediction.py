#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:30:32 2020

@author: tom
"""
import quandl
import eia
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def retrieve_time_series(api, series_ID):
    series_search = api.data_by_series(series=series_ID)
    df = pd.DataFrame(series_search)
    return df

def getEIAData(series_ID):
    api_key = "776d3a3fe9bf6dfbd47b9141d0059f79"
    api = eia.API(api_key)
    df = retrieve_time_series(api, series_ID)
    return df

def preprocessing(crudeOilMonthlyAmount, crudeOilDailyPrice):
    crudeOilDailyPrice.reset_index(level=0, inplace=True)
    crudeOilDailyPrice['Date'] = crudeOilDailyPrice['Date'].astype('datetime64[ns]')
    crudeOilMonthlyPrice = crudeOilDailyPrice.resample('M', on="Date").mean()
    
    crudeOilMonthlyAmount.reset_index(level=0, inplace=True)
    crudeOilMonthlyAmount['index'] = crudeOilMonthlyAmount['index'].astype('datetime64[ns]')
    mask = crudeOilMonthlyAmount['index'] <= '2020-04-30'
    crudeOilMonthlyAmount = crudeOilMonthlyAmount.loc[mask]
    
    training_set_x = crudeOilMonthlyAmount['Crude Oil Production, OPEC Total, Monthly (million barrels per day)'][:int(len(crudeOilMonthlyAmount) * 0.6)]
    training_set_y = crudeOilMonthlyPrice['Value'][:int(len(crudeOilMonthlyAmount) * 0.6)]
    
    validation_set_x = crudeOilMonthlyAmount['Crude Oil Production, OPEC Total, Monthly (million barrels per day)'][int(len(crudeOilMonthlyAmount) * 0.6):int(len(crudeOilMonthlyAmount) * 0.8)]
    validation_set_y = crudeOilMonthlyPrice['Value'][int(len(crudeOilMonthlyAmount) * 0.6):int(len(crudeOilMonthlyAmount) * 0.8)]
    
    return training_set_x, training_set_y, validation_set_x, validation_set_y

def MultiLayerPerceptronRegressor(training_set_x, training_set_y, validation_set_x, validation_set_y):
    mlp = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(10,),activation='logistic',
                                     tol=1e-2, max_iter=5000, random_state=0))
    mlp.fit(pd.DataFrame(training_set_x).values, pd.DataFrame(training_set_y).values)
    return mlp.score(pd.DataFrame(validation_set_x).values, pd.DataFrame(validation_set_y).values)
    
def main():
    crudeOilMonthlyAmount = getEIAData("STEO.COPR_OPEC.M")
    crudeOilDailyPrice = quandl.get("EIA/PET_RWTC_D", start_date="1994-01-01", end_date="2020-04-15")
    training_set_x, training_set_y, validation_set_x, validation_set_y = preprocessing(crudeOilMonthlyAmount, crudeOilDailyPrice)
    score = MultiLayerPerceptronRegressor(training_set_x, training_set_y, validation_set_x, validation_set_y)
    print(score)
    
main()