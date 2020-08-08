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


def clean_characters(df):
    df['New Cases']= df['New Cases'].str.replace("+","")
    df['New Cases']= df['New Cases'].str.replace(",","")                                             
    df['Total Cases']= df['Total Cases'].str.replace(",","")
    df['Total Deaths']= df['Total Deaths'].str.replace(",","")
    df['Total Deaths']= df['Total Deaths'].str.replace("+","")
    df['Total Deaths']= df['Total Deaths'].str.replace(" ","0")
    df['Total Recovered']= df['Total Recovered'].str.replace(",","")
    df['Total Recovered']= df['Total Recovered'].str.replace("N/A","0")

    
def convert_data_types(df):
    df['New Cases']= df['New Cases'].astype("int")
    df['Total Cases']= df['Total Cases'].astype("int")
    df['New Deaths']= df['New Deaths'].astype("int")
    df['Total Deaths']= df['Total Deaths'].astype("int")
    df['Total Recovered']= df['Total Recovered'].astype("int")


def import_file_to_database(conn):
    df= pd.read_csv('base_data.csv')  
    
    clean_characters(df)
    df= df.replace(np.nan,0)
    df= df.set_index('Country')
    df= df.drop(0)
    convert_data_types(df)
    df.to_sql('RapportCOVID192020', conn, if_exists='replace', index=True)


if __name__=='__main__':
    conn=create_connection(r"data_base.db")
    import_file_to_database(conn)
    conn.close()