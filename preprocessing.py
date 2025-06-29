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
    df_encoded = pd.get_dummies(df, columns=['Industry', 'Country']) 
    df_encoded = df_encoded.drop('Employee Sentiment', axis=1)

    df_encoded['GenAI_Tool_Encoded'] = df['GenAI Tool'].astype('category').cat.codes

    df_scaled = df_encoded.copy()
    df_scaled = df_scaled.drop('Company Name', axis=1)

    scaler = StandardScaler()

    numerics = ['Adoption Year', 'Number of Employees Impacted', 
                'New Roles Created', 'Training Hours Provided', 
                'Productivity Change (%)']

    df_scaled[numerics] = scaler.fit_transform(df_encoded[numerics])

    return {'original': data, 'encoded': df_encoded, 'scaled': df_scaled}
