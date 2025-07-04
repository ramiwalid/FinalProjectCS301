import pandas as pd

df = pd.read_csv('USA Housing Dataset.csv')

def pre_process(data):

    # NaN check
    if data.isnull().values.any():
        print("There are null values.")
    else:
        print("No missing values")

    # Houses with a price of 0 are most likely bad data. Removed for sake of keeping things predictable
    zero_price_count = len(data[data['price'] == 0])
    print(f"Houses with zero price count: {zero_price_count}")
    data = data[data['price'] > 0].copy()

    """
    Feature engineering
    """
    # Check city count
    print(f"There are {len(data['city'].value_counts())} unique cities.")

    # Age of house is more interpretable than year built
    data['house_age'] = 2014 - data['yr_built']
    
    # Binary flag for renovation is more useful than year
    data['renovated'] = (data['yr_renovated'] > 0).astype(int)
    
    # Square footage per room
    total_rooms_temp = data['bedrooms'] + data['bathrooms']
    data['sqft_per_room'] = data['sqft_living'] / total_rooms_temp.replace(0, 1)
    
    # Basement ratio - what proportion of living space is basement
    data['basement_ratio'] = data['sqft_basement'] / data['sqft_living'].replace(0, 1)
    
    # Total rooms
    data['total_rooms'] = data['bedrooms'] + data['bathrooms']
    
    # Above ground ratio
    data['above_ground_ratio'] = data['sqft_above'] / data['sqft_living'].replace(0, 1)
    
    # Get city counts
    city_counts = data['city'].value_counts()
    
    # Keep top 10 cities, group rest as 'Other'
    top_cities = city_counts.head(10).index.tolist()
    data['city_grouped'] = data['city'].apply(lambda x: x if x in top_cities else 'Other')
        
    # Encode the grouped cities
    data = pd.get_dummies(data, columns=['city_grouped'], prefix='city')
    
    # Define columns to check for outliers
    outlier_columns = ['price', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms']
    
    for col in outlier_columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
                        
            # Remove outliers
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
    """
    Dropped these columns because of redundancy/feature space consideration. The timeframe is short, only 2 months, and
    the market is reflected in the data anyway. Streets are unique values that blow up the feature space when encoded, and
    statezip is redundant with city. Countries contributes nothing because every house is in the USA. Also dropping yr built 
    and yr renovated since we extracted the useful information into house age and renovated features. To prevent cities from blowing
    up the feature space, I encoded them into the top 10 cities and then an "other" group.
    """
    columns_to_drop = ['date', 'street', 'statezip', 'country', 'city', 'yr_built', 'yr_renovated']
    data = data.drop(columns=columns_to_drop)
    
    return data

results = pre_process(df)