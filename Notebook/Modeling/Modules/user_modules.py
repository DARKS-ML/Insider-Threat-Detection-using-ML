import sj_library as sj
lib =sj.lb

def Show_infor(msg):
    print(f'{msg} Dataset : ')

def ReadCsv(filename):
    '''Path => ../../Dataset/'''
    df = lib.pd.read_csv(f'../../Dataset/'+filename+'.csv')
    return df

def Show_Shape(msg,df):
    '''Show Dataset Shape'''
    print(f'{msg} Dataset Shape: {df.shape}')


# def Merge_Df(df1,df2,on:str):
#     df = lib.pd.merge(df1,df2,on=on)
#     return df

def Convert_To_DT(df,column_name):
    '''
        pass dataframe and columnname which
        you want to conver to datetime
        eg;
        convertToDateTime(dataframeName,column_name)

    '''
    df[column_name] = lib.pd.to_datetime(df[column_name])
    return df