import pandas as pd

def dividerLine(msg):
    '''
        Divider Information
        eg;
        dividerLine('Message ')

    '''
    print('\n************************************\n')
    print(msg)
    print('\n************************************\n')

def simpleLine():
    print('\n************************************\n')

def convertToDateTime(df,column_name):
    '''
        pass dataframe and columnname which
        you want to conver to datetime
        eg;
        convertToDateTime(dataframeName,column_name)

    '''
    df[column_name] = pd.to_datetime(df[column_name])
    print(df.head())

def exportFile(df,file_name):
    '''
        pass dataframe and file name which
        you want !
        eg;
        exportFile(dataframeName,file_name)

    '''
    df.to_csv(r'/home/iamdpk/Project Work/Insider-Threat-Detection-using-ML/Dataset/'+file_name+'.csv',index=False)
#     /home/iamdpk/Project Work/Insider-Threat-Detection-using-ML/Dataset/r3.2/
    print(f'Exporting {file_name}_copy.csv is Done !')


def informationAboutDF(df):
    '''
        Details about Dataframe
        eg;
        informationAboutDF(dataframeName)

    '''
    simpleLine()
    print(f'Shape\n {df.shape}\n')
    simpleLine()
    print(f'Null value Information \n {df.isnull().sum()}\n')
    simpleLine()
    print(f'Dataframe Types\n {df.dtypes}\n')
    simpleLine()
    print(f'Column Information\n {df.columns}\n')
    simpleLine()
    print(f'Information\n {df.info()}\n')
    simpleLine()
    print(f'\n Sample\n {df.head()}\n')
    simpleLine()
