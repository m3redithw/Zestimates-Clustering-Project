# Essential libraries
import os
import json
import requests

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

# Local files
import env
import acquire
import prepare

def location_clusters(train, validate, test):
    cols = ['latitude', 'longitude', 'zip_bin_insgfnt high', 'zip_bin_insgfnt low',
       'zip_bin_sgfnt high']
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(train[cols])
    train['location_clusters'] = kmeans.predict(train[cols])
    validate['location_clusters'] = kmeans.predict(validate[cols])
    test['location_clusters'] = kmeans.predict(test[cols])
    train.location_clusters = train.location_clusters.astype('str')
    validate.location_clusters = validate.location_clusters.astype('str')
    test.location_clusters  = test.location_clusters.astype('str')
    return train, validate, test

def age_clusters(train, validate, test):
    cols = ['age']
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(train[cols])

    train['age_clusters'] = kmeans.predict(train[cols])
    validate['age_clusters'] = kmeans.predict(validate[cols])
    test['age_clusters'] = kmeans.predict(test[cols])
    
    train.age_clusters = train.age_clusters.astype('str')
    validate.age_clusters = validate.age_clusters.astype('str')
    test.age_clusters  = test.age_clusters.astype('str')
    return train, validate, test

def area_clusters(train, validate, test):
    cols =['total_sqft', 'lot_sqft', 'living_sqft']
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(train[cols])

    train['area_clusters'] = kmeans.predict(train[cols])
    validate['area_clusters'] = kmeans.predict(validate[cols])
    test['area_clusters'] = kmeans.predict(test[cols])
    
    train.area_clusters = train.area_clusters.astype('str')
    validate.area_clusters = validate.area_clusters.astype('str')
    test.area_clusters  = test.area_clusters.astype('str')
    return train, validate, test

def size_clusters(train, validate, test):
    cols = ['bedrooms', 'bathrooms', 'full_bath']
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(train[cols])
    
    train['size_clusters'] = kmeans.predict(train[cols])
    validate['size_clusters'] = kmeans.predict(validate[cols])
    test['size_clusters'] = kmeans.predict(test[cols])
    
    train.size_clusters = train.size_clusters.astype('str')
    validate.size_clusters = validate.size_clusters.astype('str')
    test.size_clusters  = test.size_clusters.astype('str')
    
    return train, validate, test

def value_clusters(train, validate, test):
    # adding value clusters to dataframe - not significant
    cols = ['structure_value', 'assessed_value', 'land_value','taxamount']
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(train[cols])
    train['value_clusters'] = kmeans.predict(train[cols])
    validate['value_clusters'] = kmeans.predict(validate[cols])
    test['value_clusters'] = kmeans.predict(test[cols])
    train.value_clusters = train.value_clusters.astype('str')
    validate.value_clusters = validate.value_clusters.astype('str')
    test.value_clusters  = test.value_clusters.astype('str')
    
    return train, validate, test
    
def clusters_dummy(train, validate, test):
    cols = ['location_clusters', 'age_clusters', 'area_clusters', 'size_clusters', 'value_clusters']
    train_dummy = pd.get_dummies(train[cols], dummy_na=False, drop_first=False)
    train = pd.concat([train, train_dummy], axis=1)

    validate_dummy = pd.get_dummies(validate[cols], dummy_na=False, drop_first=False)
    validate = pd.concat([validate, validate_dummy], axis=1)

    test_dummy = pd.get_dummies(test[cols], dummy_na=False, drop_first=False)
    test = pd.concat([test, test_dummy], axis=1)
    
    return train, validate, test