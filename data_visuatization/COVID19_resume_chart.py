import sqlite3
from sqlite3 import Error 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_connection(db_file):
    """create a database connection to a SQLite database"""
    conn= None
    try:
        conn=sqlite3.connect(db_file)
        print(sqlite3.version)
        
            
    except Error as e:
        print(e)
        
    return conn
        
    
def select_all_to_database(conn):
    df = pd.read_sql_query("select Country, [Total Deaths] from ResumeCOVID192020 order by [Total Deaths] DESC LIMIT 10", conn)
    df.to_sql('ResumeCOVID192020', conn, if_exists='replace', index=True)
    print (list(df.columns))
    #df.plot.barh(x='Continent_Name')
    #plt.show()


if __name__=='__main__':
    conn=create_connection(r"data_base.db")
    select_all_to_database(conn)
    conn.close()

    #select Continent_Name, Sum([Total Deaths]) from ResumeCOVID192020 Group By Continent_Name Order by Sum([Total Deaths]) DESC;
#select Continent_Name, Sum([Total Deaths]) from ResumeCOVID192020 Group By Continent_Name Order by Sum([Total Deaths]) DESC;
#select Country, [Total Deaths] from ResumeCOVID192020 order by [Total Deaths] DESC LIMIT 10;
