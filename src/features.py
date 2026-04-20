def create_features(df):
    df["engagement_score"] = (
        df["ClickThroughRate"] * df["TimeOnSite"]
    )

    return df