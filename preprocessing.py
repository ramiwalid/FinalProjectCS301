import numpy as np
import pandas as pd


# Initialize dataframe
df = pd.read_csv('genaidataset.csv')

# NaN value check (There aren't any)
if df.isnull().values.any():

    print(df.isnull().sum())

else:
    print("No missing values.")

"""
One-hot encoding for our categorical features. No inherent order between the columns, and this will allow our models to
account for each column with equal importance.
"""

df_encoded = pd.get_dummies(df, columns=['Industry', 'Country', 'GenAI Tool'])

# This is just text, and more often than not it isn't unique. This approach prevents blowing up the feature space.
df_encoded['Encoded Employee Sentiment'] = df['Employee Sentiment'].astype('category').cat.codes
df_encoded = df_encoded.drop('Employee Sentiment', axis=1)