import pandas as pd

df = pd.read_csv('USA Housing Dataset.csv')

def pre_process(data):
    # NaN check
    if data.isnull().values.any():
        print("There are null values.")
    else:
        print("No missing values")

    # Houses with a price of 0 are most likely bad data. Removed for sake of keeping things predictable.
    zero_price_count = len(data[data['price'] == 0])
    print(f"Houses with zero price count: {zero_price_count}")
    data = data[data['price'] > 0]

    # Check city count.
    print(f"There are {len(data['city'].value_counts())} unique cities.")

    """
    Dropped these columns because of redundancy/feature space consideration. The timeframe is short, only 2 months, and
    the market is reflected in the data anyway. Streets are unique values that blow up the feature space when encoded, and
    statezip is redundant with city. Countries contributes nothing because every house is in the USA. There are also 43 different
    cities and the feature space is still rich without it, so I dropped that as well for more important info.
    """
    data = data.drop(columns=['date', 'street', 'statezip', 'country', 'city'])

    return data

results = pre_process(df)