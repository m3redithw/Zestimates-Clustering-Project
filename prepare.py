# import essential libraries
import pandas as pd
import numpy as np

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

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
                              'calculatedfinishedsquarefeet':'total_sqft', 'finishedsquarefeet12': 'living_sqft', 'fullbathcnt':'full_bath', 'lotsizesquarefeet':'lot_sqft', 'structuretaxvaluedollarcnt': 'structure_value', 'taxvaluedollarcnt':'assessed_value', 'landtaxvaluedollarcnt':'land_value'}, inplace = True)
    
    # Impute   `yearbuilt` to `age`
    df['age'] = 2017 - df['yearbuilt']

    # Impute tax amount to tax rate
    df['taxrate'] = df.taxamount/df.assessed_value*100

    # Data mapping using fips code
    df['county'] = df.fips.map({6037: 'Los Angeles', 6059: 'Orange', 6111: 'Ventura'})

    # Create column for transaction month
    df['transaction_month'] = df['transactiondate'].str.slice(5, 7)

    # Create dummy variables for county and add dummies to original dataframe
    dummy_df = pd.get_dummies(df[['county']], dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis=1)

    

    # Impute latitude & longitude
    df.latitude = df.latitude/1000000
    df.longitude = df.longitude/1000000

    # Drop unuseful columns
    col = ['transactiondate','regionidcity','regionidzip','calculatedbathnbr','assessmentyear','yearbuilt','fips','propertycountylandusecode', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcounty', 'censustractandblock', 'propertylandusedesc']
    df.drop(columns = col, inplace = True)
    
    # Join table to get correct zipcode
    geo = pd.read_csv('address.csv')
    df = pd.merge(df, geo, on='parcelid', how='inner')

    # Drop new nulls from zipcode
    df = df.dropna()    

    # Change data types
    df['age'] = df['age'].astype(int)
    df['zip_code'] = df['zip_code'].astype(int)
    df['transaction_month']=df['transaction_month'].astype(int)

    # Handle Outliers:
    # The general rule for outliers are:
    ## Upper bond: Q3 + 1.5*IQR
    ## Lower bund: Q1 - 1.5*IQR
    # Bonds are manually adjusted for each feature
    
    df = df[df.zip_code >= 90000]
    df = df[df.zip_code < 100000]
    df = df[df.bedrooms <= 7]
    df = df[df.bedrooms >= 1]

    df = df[df.bathrooms <= 7]
    df = df[df.bathrooms >= 0.5]

    df = df[df.total_sqft <= 7500]
    df = df[df.total_sqft >= 500]

    df = df[df.lot_sqft <= 50000]
    df = df[df.lot_sqft >= 900]

    df = df[df.assessed_value <= 1200000]
    df = df[df.assessed_value >= 45500]

    df = df[df.taxrate < 10]

    # Join zip groups by logerror
    zip_error = pd.read_csv('logerror_zip.csv')
    df = pd.merge(df, zip_error, on='zip_code', how='left')
    df['zip_bin'] = df.zip_group.map({1: 'sgfnt high', 2: 'sgfnt low', 3: 'insgfnt high', 4: 'insgfnt low'})
    zipdummy = pd.get_dummies(df[['zip_bin']], dummy_na=False, drop_first=False)
    df = pd.concat([df, zipdummy], axis=1)
    df.drop(columns = 'zip_group', inplace = True)
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

# This function scales the data using MinMaxScaler(), then splits the data into train, validate, test.
def split_scale(df):
    # Copy a new dataframe to perform feature engineering
    scaled_df = df.copy()

    # Initiate MinMaxScaler
    scaler = MinMaxScaler()

    # Split the scaled data into train, validate, test
    train, validate, test = split(scaled_df)

    # Columns to scale
    cols = ['bathrooms', 'bedrooms', 'total_sqft', 'living_sqft', 'full_bath',
       'latitude', 'longitude', 'lot_sqft', 'roomcnt',
       'structure_value', 'assessed_value', 'land_value', 'taxamount', 'age', 'transaction_month']

    # Fit numerical features to scaler
    scaler.fit(train[cols])

    # Set the features to transformed value
    train[cols] = scaler.transform(train[cols])
    validate[cols] = scaler.transform(validate[cols])
    test[cols] = scaler.transform(test[cols])


    return train, validate, test