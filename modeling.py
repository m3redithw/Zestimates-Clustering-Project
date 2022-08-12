# Essential libraries
import os
import json
import requests

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Data hanlding
import pandas as pd
import numpy as np
import pydataset
import scipy.stats as stats
import statistics as s

# Sklearn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor

from sklearn.model_selection import train_test_split

# Local files
import env
import acquire
import prepare
import clustering

# Text display
import colorama
from colorama import Fore

# This function creates 3 dataframes for model's predictions, makes baseline prediction for train, validate, test.
def baseline_prediction(train, validate, test):
    # Initialized dataframe to hold preditions from different models
    train_predictions = pd.DataFrame({
        'actual': train.logerror}) 
    validate_predictions = pd.DataFrame({
        'actual': validate.logerror})
    test_predictions = pd.DataFrame({
        'actual': test.logerror})
    
    # Adding baseline predictions to dataframes
    train_predictions['baseline'] = train.logerror.mean()
    validate_predictions['basline']=validate.logerror.mean()
    test_predictions['baseline']=test.logerror.mean()
    
    # Print baseline rmse for train dataset's baseline
    rmse = mean_squared_error(train_predictions.actual, train_predictions.baseline, squared = False)
    print(Fore.BLUE+ "\nRoot mean of squared error of baseline prediction is: ", "{:10.3f}".format(rmse))
    return train_predictions, validate_predictions, test_predictions

# This function uses the first feature combination to make prediction using multiple regression & polynomial features.
# It will then add the predictions to the dataframes that we initiated.
def feature_a_models(train, validate, test, train_predictions, validate_predictions, test_predictions):
    cols = ['age_clusters_0', 'age_clusters_1', 'age_clusters_2','location_clusters_0',
            'location_clusters_1',  'location_clusters_2', 'location_clusters_3',
            'total_sqft','lot_sqft','size_clusters_0','size_clusters_1', 'size_clusters_2',
            'value_clusters_0', 'value_clusters_1', 'value_clusters_2', 'transaction_month']
    X_train = train[cols]
    y_train = train.logerror

    X_validate = validate[cols]
    y_validate = validate.logerror

    X_test = test[cols]
    y_test = test.logerror
    # Notes: I looped through k and found out the model performs the best when k=15
    # Initiate the linear regression model
    lm = LinearRegression()

    # Transform our X
    rfe = RFE(lm, n_features_to_select=7)
    rfe.fit(X_train, y_train)

    # Use the transformed x in our model
    X_train_rfe = rfe.transform(X_train)
    X_validate_rfe = rfe.transform(X_validate)
    X_test_rfe = rfe.transform(X_test)
    lm.fit(X_train_rfe, y_train)

    # Make predictions and add that to the predictions dataframe
    train_predictions['feature a multiple rfe k=15'] = lm.predict(X_train_rfe)
    validate_predictions['feature a multiple rfe k=15'] = lm.predict(X_validate_rfe)
    test_predictions['feature a multiple rfe k=15'] = lm.predict(X_test_rfe)
    
    # Polynomial degree 2 interaction terms only
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly.fit(X_train)
    X_train_poly = pd.DataFrame(
        poly.transform(X_train),
        columns=poly.get_feature_names(X_train.columns),
        index=X_train.index,
    )
    lm = LinearRegression()
    lm.fit(X_train_poly, y_train)

    X_validate_poly = poly.transform(X_validate)
    X_test_poly = poly.transform(X_test)

    # Make predictions and add that to the predictions dataframe
    train_predictions['feature a polynomial degree 2 only interaction'] = lm.predict(X_train_poly)
    validate_predictions['feature a polynomial degree 2 only interaction'] = lm.predict(X_validate_poly)
    test_predictions['feature a polynomial degree 2 only interaction'] = lm.predict(X_test_poly)
    return train_predictions, validate_predictions, test_predictions

# This function uses the second feature combination to make prediction using multiple regression & polynomial features.
# It will then add the predictions to the dataframes that we initiated.
def feature_b_models(train, validate, test, train_predictions, validate_predictions, test_predictions):
    cols = ['living_sqft', 'structure_value', 'assessed_value', 'land_value',
       'zip_bin_insgfnt high', 'zip_bin_sgfnt high', 'location_clusters_0',
       'location_clusters_2', 'location_clusters_3', 'value_clusters_0']
    X_train = train[cols]
    y_train = train.logerror

    X_validate = validate[cols]
    y_validate = validate.logerror

    X_test = test[cols]
    y_test = test.logerror
    
    # Notes: I looped through k and found out the model performs the best when k=9
    # Initiate the linear regression model
    lm = LinearRegression()

    # Transform our X
    rfe = RFE(lm, n_features_to_select=9)
    rfe.fit(X_train, y_train)

    # Use the transformed x in our model
    X_train_rfe = rfe.transform(X_train)
    X_validate_rfe = rfe.transform(X_validate)
    X_test_rfe = rfe.transform(X_test)
    lm.fit(X_train_rfe, y_train)

    # Make predictions and add that to the predictions dataframe
    train_predictions['feature b multiple rfe k=9'] = lm.predict(X_train_rfe)
    validate_predictions['feature b multiple rfe k=9'] = lm.predict(X_validate_rfe)
    test_predictions['feature b multiple rfe k=9'] = lm.predict(X_test_rfe)
    
    # Polynomial degree 2
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly.fit(X_train)
    X_train_poly = pd.DataFrame(
        poly.transform(X_train),
        columns=poly.get_feature_names(X_train.columns),
        index=X_train.index,
    )
    lm = LinearRegression()
    lm.fit(X_train_poly, y_train)

    X_validate_poly = poly.transform(X_validate)
    X_test_poly = poly.transform(X_test)
    
    # Make predictions and add that to the predictions dataframe
    train_predictions['feature b polynomial degree 2'] = lm.predict(X_train_poly)
    validate_predictions['feature b polynomial degree 2'] = lm.predict(X_validate_poly)
    test_predictions['feature b polynomial degree 2'] = lm.predict(X_test_poly)
    return train_predictions, validate_predictions, test_predictions

# This function uses the thrid feature combination to make prediction using multiple regression & polynomial features.
# It will then add the predictions to the dataframes that we initiated.
def feature_c_models(train, validate, test, train_predictions, validate_predictions, test_predictions):
    cols = ['living_sqft', 'structure_value', 'assessed_value', 'land_value',
       'county_Los Angeles', 'county_Orange', 'county_Ventura',
       'zip_bin_insgfnt high', 'zip_bin_sgfnt high', 'location_clusters_0',
       'location_clusters_2', 'location_clusters_3', 'age_clusters_0',
       'area_clusters_1', 'area_clusters_2', 'size_clusters_0',
       'value_clusters_0']
    X_train = train[cols]
    y_train = train.logerror

    X_validate = validate[cols]
    y_validate = validate.logerror

    X_test = test[cols]
    y_test = test.logerror
    # Notes: I looped through k and found out the model performs the best when k=9
    # Initiate the linear regression model
    lm = LinearRegression()

    # Transform our X
    rfe = RFE(lm, n_features_to_select=6)
    rfe.fit(X_train, y_train)

    # Use the transformed x in our model
    X_train_rfe = rfe.transform(X_train)
    X_validate_rfe = rfe.transform(X_validate)
    X_test_rfe = rfe.transform(X_test)
    lm.fit(X_train_rfe, y_train)

    # Make predictions and add that to the train_predictions dataframe
    train_predictions['feature c multiple rfe k=6'] = lm.predict(X_train_rfe)
    validate_predictions['feature c multiple rfe k=6'] = lm.predict(X_validate_rfe)
    test_predictions['feature c multiple rfe k=6'] = lm.predict(X_test_rfe)
    
    # Polynomial degree 2
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly.fit(X_train)
    X_train_poly = pd.DataFrame(
        poly.transform(X_train),
        columns=poly.get_feature_names(X_train.columns),
        index=X_train.index,
    )
    lm = LinearRegression()
    lm.fit(X_train_poly, y_train)

    X_validate_poly = poly.transform(X_validate)
    X_test_poly = poly.transform(X_test)

    # Make predictions and add that to the predictions dataframe
    train_predictions['feature c polynomial degree 2'] = lm.predict(X_train_poly)
    validate_predictions['feature c polynomial degree 2'] = lm.predict(X_validate_poly)
    test_predictions['feature c polynomial degree 2'] = lm.predict(X_test_poly)
    return train_predictions, validate_predictions, test_predictions
