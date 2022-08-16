# How Accurate Is Your Zestimates
by **Meredith Wang**

<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-013243.svg?logo=python&logoColor=white"></a>
<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-2a4d69.svg?logo=numpy&logoColor=white"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-8DF9C1.svg?logo=matplotlib&logoColor=white"></a>
<a href="#"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-65A9A8.svg?logo=pandas&logoColor=white"></a>
<a href="#"><img alt="plotly" src="https://img.shields.io/badge/plotly-adcbe3.svg?logo=plotly&logoColor=white"></a>
<a href="#"><img alt="sklearn" src="https://img.shields.io/badge/sklearn-4b86b4.svg?logo=scikitlearn&logoColor=white"></a>
<a href="#"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1560bd.svg?logo=scipy&logoColor=white"></a>
<a href="#"><img alt="GeoPandas" src="https://img.shields.io/badge/GeoPandas-1faecf.svg?logo=python-geopandas&logoColor=white"></a>

![Zillow-Zestimates](https://user-images.githubusercontent.com/105242871/181685714-7b738d62-e43f-4a4d-b026-e68d736ebdff.jpeg)




**Zillow's ZestimateⓇ** is an estimate of value using a proprietary formula created by the online real estate database company. Zestimates cover more than 100 million homes across the United States. A Zestimate is calculated from physical attributes, tax records, and user submitted data.

In this project, we will use statistical analysis to analyze the key drivers of logerror, which is defined as the differnece between the predicted log error and the actual log error. We will incorporate clustering methodologies, and develop a ML regression model to predict the log error, and provide recommendations on making more accurate prediction on log error which further leads to better prediction on home value predictions.
```
logerror = log(Zestimate) − log(SalePrice)
```

## :house:   Project Goals
▪️ Find the key drivers of log error for **single family properties** in 2017.

▪️ Use clustering methodologies to explore and understand the relationship between features better.

▪️ Construct an ML Regression model that predict **log error** ('logerror') of Single Family Properties using attributes of the properties and the useful labels we discovered from clustering.

▪️ Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.

▪️ Make recommendations on what works or doesn't work in prediction log error.

## :memo:   Initial Questions
▪️ Are any of the location features associated with `logerror`? Is `logerror` significantly different across different counties? What about latitude, longtitude? What about zip code?

▪️ Are any of the area (in square feet) features, including `total_sqft`, `lot_sqft`, `living_sqft`, of the property associated with `logerror`?

▪️ Are any of the size features, including `bedrroms`, `bathrooms`, `full_bath`, `roomcnt`, of the property associated with `logerror`?

▪️ Is the `age` of the house correlated with `logerror`?

▪️ Is `transaction_month` correlated with `logerror`?

## :open_file_folder:   Data Dictionary
**Variable** |    **Value**    | **Meaning**
---|---|---
*Latitude* | Float | Latitude of the middle of the parcel
*Longitude* | Float | Longitude of the middle of the parcel
*Zip Code* | Integer | Zip code in which the property is located
*County* | 1) Ventura 2) Los Angeles 3) Orange | County in which the property is located
*Bedrooms* | Integer | Number of bedrooms in home 
*Bathrooms* | Float | Number of bathrooms in home including fractional bathrooms
*Full Bath* | Interger |  Number of full bathrooms (sink, shower + bathtub, and toilet) present in home
*Room Count* | Float |  Total number of rooms in the principal residence
*Total Sqft* | Float | Calculated total finished living area of the home
*Living Sqft* | Float | Finished living area
*Lot Sqft* | Float | Area of the lot in square feet
*Assessed Value* | Float | The total tax assessed value of the parcel
*Structure Value* | Float | The assessed value of the built structure on the parcel
*Land Value* | Float | The assessed value of the land area of the parcel
*Tax Amount* | Float | The total property tax assessed for that assessment year
*Age* | Integer | This indicate the age of the property in 2017, calculated using the year the principal residence was built 
*Transaction Month* | Integer | The month in 2017 that the property is sold
Note: Full dictionary please reference [zillow_data_dictionary](zillow_data_dictionary.xlsx)

## :jigsaw:   Data Overview
![data_overview](https://user-images.githubusercontent.com/105242871/183556966-ea49f052-b409-415d-83d9-7c4aba868b03.jpg)


## :placard:   Project Plan / Process
#### :one:   Data Acquisition

<details>
<summary> Gather data from mySQL database</summary>

- Create env.py file to establish connection to mySQL server

- Use **zillow** database in the mySQL server

- Read data dictionary and extract meaningful columns from table in the **zillow** database

- Write query to join useful tables to gather all data about the houses in the region:  <u>properties_2017, predictions_2017, propertylandusetype </u>
     ```sh
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
    WHERE
	    prop.propertylandusetypeid = 261 AND
	    prop.latitude IS NOT NULL
	    AND prop.longitude IS NOT NULL
	    AND transactiondate <= '2017-12-31';
     ```
</details>

<details>
<summary> acqure.py</summary>

- Create acquire.py and user-defined function `get_zillow_data()` to gather data from mySQL
     ```sh
     def get_zillow_data():
     
     if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        df = new_zillow_data()
        df.to_csv('zillow.csv')
        
    return df
    ```
- Import [acquire.py](acquire.py)

- Test acquire function

- Calling the function, and store the table in the form of dataframe
    ```sh
    df = acquire.get_zillow_data()
    ```
</details>

#### :two:   Data Preparation

<details>
<summary> Missing Value Analysis</summary>

- Visualize the percentage of missing data of each variable

- Create a function that removes columns and rows that have more than a certian percentage of missing values

	```sh
	    def handle_missing_values(df, prop_required_columns, prop_required_row):
		    threshold = int(round(prop_required_columns * len(df.index), 0))
		    df = df.dropna(axis=1, thresh=threshold) #1, or ‘columns’ : Drop columns which contain missing values
		    threshold = int(round(prop_required_row * len(df.columns), 0))
		    df = df.dropna(axis=0, thresh=threshold) #0, or ‘index’ : Drop rows which contain missing values
		    return df
	``` 
</details>

<details>
<summary> Data Cleaning</summary>

- **Missing values:**
Null values are dropped for entire dataset
	```sh
	df = df.dropna()
	```

- **Rename Columns**
     ```sh
     df.rename(columns = {'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms',
                              'calculatedfinishedsquarefeet':'total_sqft', 'finishedsquarefeet12': 'living_sqft',
			      'fullbathcnt':'full_bath', 'lotsizesquarefeet':'lot_sqft', 'structuretaxvaluedollarcnt': 'structure_value',
			      'taxvaluedollarcnt':'assessed_value', 'landtaxvaluedollarcnt':'land_value'}, inplace = True)
     ```
     
- **Data Conversion**
	- Convert `yearbuilt` to `age`
	 ```sh
	 df['age'] = 2017 - df['yearbuilt']
	 ```
	 
	 - Convert `taxamount` to `taxrate`
	 ```sh
	 df['taxrate'] = df.taxamount/df.assessed_value*100
	 ```
	 
	 - Extract month from `transaction_date`
	 ```sh
	 df['transaction_month'] = df['transactiondate'].str.slice(5, 7)
	 ```
	
	 - Convert `latitude` and `longitude` to correct digit
	 ```sh
	 df.latitude = df.latitude/1000000
 	 df.longitude = df.longitude/1000000
	 ```
	
- **Join Tables**
	- Join table **address.csv** which has the correct zip code for properties (derived from geo engineering)
	```sh
	geo = pd.read_csv('address.csv')
	df = pd.merge(df, geo, on='parcelid', how='inner')
	```
	
	- Join table **logerror_zip.csv** which utilized T-test to decide the significancy of logerrors corresponding to each zip code
	```sh
	zip_error = pd.read_csv('logerror_zip.csv')
	df = pd.merge(df, zip_error, on='zip_code', how='left')
	```
	
- **Data Mapping**
    - Created new `county` column with county name corresponding to `fips_code`
    ```sh
    df['county'] = df.fips.map({6037: 'Los Angeles', 6059: 'Orange', 6111: 'Ventura'})
    ```
    
    - Create new `zip_bin` column with category name corresponding to each `zip_group`
    ```sh
    df['zip_bin'] = df.zip_group.map({1: 'sgfnt high', 2: 'sgfnt low', 3: 'insgfnt high', 4: 'insgfnt low'})
    ```
    
- **Dummy Variables:**
    - Created dummy variables for categorical feature `county`
    ```sh
    dummy_df = pd.get_dummies(df[['county']], dummy_na=False, drop_first=False)
    ```
    
    - Concatenated all `county` dummy variables onto original dataframe
    ```sh
    df = pd.concat([df, dummy_df], axis=1)
    ```
    
    - Create dummy variables for categorical feature `zip_group`
    ```sh
    zipdummy = pd.get_dummies(df[['zip_bin']], dummy_na=False, drop_first=False)
    ```
    
    - Concatenated all `county` dummy variables onto original dataframe
    ```sh
    df = pd.concat([df, zipdummy], axis=1)
    ```
    
- **Data types:**
`float` is converted to `int` datatype
     ```sh
     df['age'] = df['age'].astype(int)
     df['zip_code'] = df['zip_code'].astype(int)
     df['transaction_month']=df['transaction_month'].astype(int)
     ```
     
- **Outliers**
    - General rull for handling outliers:
        - Upper bond: Q3 + 1.5 * IQR
        - Lower bond: Q1 - 1.5 * IQR
    
        **Note:** each feature has minor adjustment based on data distribution
    - Outliers for each feature are dropped
        ```sh
        df = df[df.bedrooms <= 7]
        df = df[df.bedrooms >= 1]

        df = df[df.bathrooms <= 7]
        df = df[df.bathrooms >= 0.5]

        df = df[df.square_feet <= 7500]
        df = df[df.square_feet >= 500]

        df = df[df.lot_size <= 50000]
        df = df[df.lot_size >= 900]

        df = df[df.assessed_value <= 1200000]
        df = df[df.assessed_value >= 45500]
        ```
- **Drop Columns**
Unuseful columns are dropped
	```sh
	col = ['transactiondate','regionidcity','regionidzip','calculatedbathnbr','assessmentyear','yearbuilt','fips','propertycountylandusecode', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcounty', 'censustractandblock', 'propertylandusedesc']
	df.drop(columns = col, inplace = True)
	```
	
- Create function `prep_zillow` to clean and prepare data with steps above

- Import [prepare.py](prepare.py)

- Test prepare function

- Call the function, and store the cleaned data in the form of dataframe
</details>

<details>
<summary> Data Splitting</summary>

- Create function `split()` to split data into **train, validate, test**

- Test split function

- Check the size of each dataset
     ```sh
     train.shape, validate.shape, test.shape
     ```
- Call the function, and store the 3 data samples separately in the form of dataframe
     ```sh
     train, validate, test = prepare.split(df)
     ```
</details>

<details>
<summary> Data Scaling</summary>

- Scaling numerical features using `MinMaxScaler()`

- Create a function that copies the original dataframe, split the data into train, validate, and test, then scale the data of each dataset

	```sh
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
		'structure_value', 'assessed_value', 'land_value', 'taxamount', 'age']

		# Fit numerical features to scaler
		scaler.fit(train[cols])

		# Set the features to transformed value
		train[cols] = scaler.transform(train[cols])
		validate[cols] = scaler.transform(validate[cols])
		test[cols] = scaler.transform(test[cols])
	
    	return train, validate, test
	``` 
</details>
	
#### :three:   Exploratory Analysis
- Ask questions to find what are the key features that are associated with property assessed value

- Explore each feature's correlation with assessed value

- Use visualizations to better understand the relationship between features

- Use centroid-based clustering method to find patterns in data, and use the meaningful clusters to group data
![plotly 3d scatter](https://user-images.githubusercontent.com/105242871/183792869-a1b6cbf0-009e-413c-b326-77a6f8d2f3ee.GIF)


#### :four:    Statistical Testing & Modeling	
- Conduct T-Test for categorical variable vs. numerical variable

- Conduct Pearson R for numerical variable vs. numerical variable
	
- Conduct Chi^2 Test for categorical variable vs. categorical variable

- Conclude hypothesis and address the initial questions
	
#### :five:    Modeling Evaluation
- Create multiple regression model and use Recursive Feature Elimination (RFE) to select features

- Find the amount of features that can gerenate the highest performance (evaluated using Root Mean Squared Error)

- Generate polynomial model, fit and tranform the train dataset into feature

- Find the degree that generates the best performing model (evaluated using RMSE)

- Create lasso-lars model object, fit the model to our training data, and use the model to make predictions

- Create generalized linear model `(TweedieRegressor)` and fit train dataset

- Pick the top 3 models among all the models and evaluate performance on validate dataset
    
- Stored model performance on train and validate in [model_performance.csv](model_performance.csv)

- Pick the model with highest accuracy and evaluate on test dataset

## :repeat:   Steps to Reproduce
- [x] You will need an **env.py** file that contains the hostname, username and password of the mySQL database that contains the telco table. Store that env file locally in the repository.
- [x] Clone my repo (including the **imports.py**, **acquire.py**, **prepare.py**, **clustering.py**, **modeling.py**, **address.csv**, **logerror_zip.csv**) 
- [x] Confirm **.gitignore** is hiding your env.py file
- [x] Ensure you have imported `*` from  **imports.py** before running anything else (Libraries used are pandas, matplotlib, seaborn, plotly, sklearn, scipy which are all included in the imports file)
- [x] Follow instructions in [eda](eda.ipynb) workbook on steps for data exploration
- [x] Reference [feature_engineering](feature_engineering.ipynb) for steps took to select clusters
- [x] Reference in [modeling](modeling.ipynb) for experimentations that drove to the final models
- [x] Follow instructions in [final_report](final_report.ipynb) and README file
- [x] Good to run final report :smile_cat:

## :key:    Key Findings

▪️ Logerror is dependent on location features. <span style="color: blue"> Location clusters </span> are relevant and useful for estimating logerror.

▪️ Area features have <span style="color: blue"> weak </span> correlation with logerror. Area clusters seem relevant judging from the bra graph, but the ANOVA test concludes there's no significant difference between clusters.

▪️ Size features have <span style="color: blue"> weak </span> correlation with logerror. We can see the relationship from visualizing clusters but the ANOVA test concludes there's no significant difference between clusters.

▪️ Value features have <span style="color: blue"> weak </span> correlation with logerror. Although the visualization tells us there's a negative correlation between them, the ANOVA test concludes there's no significant difference between clusters.

(Each feature is a driver of property tax assessed value, supported by visualization and statistical testing. For lot size and age analysis please reference [zillow_eda](zillow_eda.ipynb))

<img width="1167" alt="final_model" src="https://user-images.githubusercontent.com/105242871/183556801-ed23751f-a7f1-44a9-a0b3-2d3c2b616b4d.png">

## :high_brightness:    Recommendations
▪️ Impute null values instead of dropping them.

▪️ Handle outliers differently, or save the outliers so that the final model will have better prediction on future onseen data.

▪️ Experiment with more feature combinations and different algorithms.

▪️ To make better predictions, we need to gather more accurate **geographical data**. In this dataset we're given, the `regionidcity`, `regionidzip` etc. are not accurate.


## 🔜  Next Steps
▪️ Collect more **geographic** data on the property(e.g. local school, surrounding properties, distance from downtown, city population, etc.)

▪️ Develop machine learning models with higher accuracy (lower RMSE) with these additonal data and make better predictions.

▪️ Collect data on previous years (e.g. historical time to close data) to analyze the general trend of each area, and determine what features drive the logerror the most.
