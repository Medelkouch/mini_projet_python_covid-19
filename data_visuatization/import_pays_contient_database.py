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

    df['iso_code1']= df['iso_code1'].astype(str)
    df['Continent_Name']= df['Continent_Name'].astype(str)
    df['Country_Name']= df['Country_Name'].astype(str)


def delete_columns_from_dataframe(df):
        df.drop(columns= ['Country_Name'],inplace= True)
        


def select_all_to_database(conn):
    df = pd.read_sql_query("select * from ContinentCovid192020 ct inner join RapportCovid192020 rap on ct.Country_Name = rap.Country", conn)
    print (list(df.columns))
    delete_columns_from_dataframe(df)
    df.to_sql('ResumeCOVID192020', conn, if_exists='replace', index=True)


def import_file_to_database(conn):
    df= pd.read_csv('Pays_continent.csv')
    convert_data_types(df)
    df.to_sql('ContinentCOVID192020', conn, if_exists='replace', index=False)



    
                                       
    
if __name__=='__main__':
    conn=create_connection(r"data_base.db")
    #import_file_to_database(conn)
    select_all_to_database(conn)
    conn.close()