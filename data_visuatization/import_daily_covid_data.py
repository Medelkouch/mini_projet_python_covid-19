import sqlite3
from sqlite3 import Error 
import pandas as pd
import numpy as np


def create_connection(db_file):
    """create a database connection to a SQLite database"""
    conn= None
    try:
        conn=sqlite3.connect(db_file)
        print(sqlite3.version)
        
            
    except Error as e:
        print(e)
        
    return conn

    
def convert_data_types(df):

    df['date'] = pd.to_datetime(df['date'])
    df['total_cases']= df['total_cases'].astype("int")
    df['new_cases']= df['new_cases'].astype("int")
    df['total_deaths']= df['total_deaths'].astype("int")
    df['new_deaths']= df['new_deaths'].astype("int")


def delete_columns_from_dataframe(df):
        df.drop(columns= ['total_cases_per_million', 'new_cases_per_million', 'total_deaths_per_million', 'new_deaths_per_million', 'total_tests', 'new_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'tests_units', 'population', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cvd_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_100k'],inplace= True)


def import_file_to_database(conn):
    df= pd.read_csv('owid-covid-data.csv')
    data= pd.read_csv('country-and-continent-codes-list-csv_csv.csv')
    pd.dataframe(data)
    
    # clean_characters(df)
    df= df.replace(np.nan,0)
    df= df.set_index('iso_code')
    df= df.drop(0)
    convert_data_types(df)
    delete_columns_from_dataframe(df)
    df.to_sql('DailyCOVID192020', conn, if_exists='replace', index=True)



    
                                       
    
if __name__=='__main__':
    conn=create_connection(r"data_base.db")
    import_file_to_database(conn)
    conn.close()