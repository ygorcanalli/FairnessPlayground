import sqlite3
import pandas as pd
from pprint import pprint
from sqlite3 import Error
 
 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
 
    return conn
 
def create_nnar_table(conn, table_name):
    """ create a table nnar_table statement
    :param conn: Connection object
    :param table_name: table name
    :param pred_size: size of predicted y
    :return:
    """
    sql = "CREATE TABLE " + table_name + """(
        id integer PRIMARY KEY,
        fp_male real,
        fn_male real,
        fp_female real,
        fn_female real,
        test_loss real,
        test_acc real);"""

    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)

def persist_nnar(conn, table_name, values):
    """
    Persist a nnar result into the choosen table
    :param conn: sqlite connection
    :param table_name: table name
    :param values: values to insert
    :return:
    """
    sql = "INSERT INTO " + table_name + \
        " (fp_male,fn_male,fp_female,fn_female,test_loss,test_acc) VALUES (?,?,?,?,?,?);"      
            
    try:
        cur = conn.cursor()
        cur.execute(sql, values)
    except Error as e:
        print(e)

def load_table(conn, table_name):
    """
    Read a table with pandas
    :param conn: sqlite connection
    :param table_name: table name
    :param values: values to insert
    :return: pandas dataframe with whole table
    """
    return pd.read_sql_query("SELECT * FROM " + table_name, conn)

""" how to
conn = create_connection("test.db")
with conn:
    create_nnar_table(conn, "baseline")
    row = [0.0,0.0,0.0,0.0,0.35288344998465093,0.8366594910621643]
    persist_nnar(conn, "baseline", row)
"""