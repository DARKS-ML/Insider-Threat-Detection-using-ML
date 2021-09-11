import pandas as pd


def ReadDataset(filename):
    '''Path ===> ../Dataset/'''
    df = pd.read_csv(r'../../Dataset/'+filename+'.csv')
    return df


def ChooseDataBetweenTwoDate(df,start_date,end_date):
    new_df = df[df.date.between(start_date,end_date)]
    return new_df
