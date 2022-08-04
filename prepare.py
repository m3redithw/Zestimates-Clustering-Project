# import essential libraries
import pandas as pd
import numpy as np

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# import data acquisition
import acquire

def handle_missing_values(df, prop_required_columns, prop_required_row):
    # This function filter out columns and rows that have more or equal to a certian percentage of data, and drop the rest that have 1-percentage missing value.
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold) #1, or ‘columns’ : Drop columns which contain missing value
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold) #0, or ‘index’ : Drop rows which contain missing values.
    return df

def prep_zillow(df):
    # Drop columns and rows that are missing 25% or more data
    df = handle_missing_values(df, 0.75, 0.75)

    # Drop missing values
    df = df.dropna()
    
    # Rename columns
    df.rename(columns = {'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms',
                              'calculatedfinishedsquarefeet':'total_sqft', 'finishedsquarefeet12': 'living_sqft', 'fullbathcnt':'full_bath', 'lotsizesquarefeet':'lot_sqft', 'regionidcity':'city', 'regionidzip':'zips', 'structuretaxvaluedollarcnt': 'structure_value', 'taxvaluedollarcnt':'assessed_value', 'landtaxvaluedollarcnt':'land_value'}, inplace = True)
    
    # Impute   `yearbuilt` to `age`
    df['age'] = 2017 - df['yearbuilt']
    
    # Data mapping using fips code
    df['county'] = df.fips.map({6037: 'Los Angeles', 6059: 'Orange', 6111: 'Ventura'})
    # Impute latitude & longitude
    df.latitude = df.latitude/1000000
    df.longitude = df.longitude/1000000

    # Drop unuseful columns
    col = ['parcelid','assessmentyear','yearbuilt','fips','propertycountylandusecode', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcounty', 'censustractandblock', 'propertylandusedesc']
    df.drop(columns = col, inplace = True)
    
    # Drop incorrect zips
    df = df['zips']<=99999
    
    # Change data types
    df['city'] = df['city'].astype(int)
    df['zips'] = df['zips'].astype(int)
    df['age'] = df['age'].astype(int)

    return df

def split(df):
    '''
    This function splits a dataframe into 
    train, validate, and test in order to explore the data and to create and validate models. 
    It takes in a dataframe and contains an integer for setting a seed for replication. 
    Test is 20% of the original dataset. The remaining 80% of the dataset is 
    divided between valiidate and train, with validate being .30*.80= 24% of 
    the original dataset, and train being .70*.80= 56% of the original dataset. 
    The function returns, train, validate and test dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123)   
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test