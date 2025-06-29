import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize dataframe
df = pd.read_csv('genaidataset.csv')

def preprocessing(data):

    # NaN value check (There aren't any)
    if data.isnull().values.any():

        print(data.isnull().sum())

    else:
        print("No missing values.")

    """
    One-hot encoding for our categorical features. No inherent order between the columns, and this will allow our models to
    account for each column with equal importance. Dropped employee sentiment because encoding it loses all the informtion
    that it offers.
    """
    df_encoded = pd.get_dummies(data, columns=['Industry', 'Country', 'GenAI Tool'])
    df_encoded = df_encoded.drop('Employee Sentiment', axis=1)

    # self-explanatory
    df_scaled = df_encoded.copy()

    # The name of the companies doesn't matter in the actual training
    df_scaled = df_scaled.drop('Company Name', axis=1)

    # self-explanatory
    scaler = StandardScaler()

    # numeric columns
    numerics = ['Adoption Year', 'Number of Employees Impacted', 
                    'New Roles Created', 'Training Hours Provided', 
                    'Productivity Change (%)']

    # Scaling
    df_scaled[numerics] = scaler.fit_transform(df_encoded[numerics])

    return {'original': data, 'encoded': df_encoded, 'scaled': df_scaled}
