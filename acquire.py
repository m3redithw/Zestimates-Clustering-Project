import pandas as pd
import os
import env
import requests

# Getting conncection to mySQL database, and acquiring data
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Loading raw data from Zillow database
def new_zillow_data():
    '''
    This function reads the Zillow data from the mySQL database into a df.
    '''
    # Create SQL query.
    sql_query = '''
    SELECT 
    prop.*,
    pred.logerror,
    pred.transactiondate,
    air.airconditioningdesc,
    arch.architecturalstyledesc,
    build.buildingclassdesc,
    heat.heatingorsystemdesc,
    landuse.propertylandusedesc,
    story.storydesc,
    construct.typeconstructiondesc
FROM
    properties_2017 prop
        INNER JOIN
    (SELECT 
        parcelid, logerror, MAX(transactiondate) AS transactiondate
    FROM
        predictions_2017
    GROUP BY parcelid ,  logerror) pred USING (parcelid)
        LEFT JOIN
    airconditioningtype air USING (airconditioningtypeid)
        LEFT JOIN
    architecturalstyletype arch USING (architecturalstyletypeid)
        LEFT JOIN
    buildingclasstype build USING (buildingclasstypeid)
        LEFT JOIN
    heatingorsystemtype heat USING (heatingorsystemtypeid)
        LEFT JOIN
    propertylandusetype landuse USING (propertylandusetypeid)
        LEFT JOIN
    storytype story USING (storytypeid)
        LEFT JOIN
    typeconstructiontype construct USING (typeconstructiontypeid)
WHERE prop.propertylandusetypeid = 261 AND
        prop.latitude IS NOT NULL
        AND prop.longitude IS NOT NULL
        AND transactiondate <= '2017-12-31';
    '''
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df = df.drop(columns = 'id')
    
    return df

def get_zillow_data():
    '''
    This function reads in zillow data from Zillow database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_data.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_data.csv')
        
    return df