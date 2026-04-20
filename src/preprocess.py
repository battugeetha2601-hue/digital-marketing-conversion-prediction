from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def preprocess_data(df):
    df = df.drop_duplicates()

    enc = OrdinalEncoder()
    df[['CampaignChannelV2','CampaignTypeV2']] = enc.fit_transform(
        df[['CampaignChannel','CampaignType']]
    )
    
    # Drop the original string columns after encoding to avoid XGBoost errors
    df = df.drop(columns=['CampaignChannel', 'CampaignType'])

    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    return df