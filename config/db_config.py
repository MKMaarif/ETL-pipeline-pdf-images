from dotenv import load_dotenv
import os
import psycopg2 
import pandas as pd 
from sqlalchemy import create_engine

load_dotenv()

def initialize_db():
    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database="postgres" # Connect to the default database
    )
    connection.autocommit = True
    cursor = connection.cursor()

    # Check if database exists
    db_name = os.getenv("DB_NAME")
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute(f"CREATE DATABASE {db_name}")

    cursor.close()
    connection.close()


def create_connection():
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        engine = create_engine("postgresql://{}:{}@{}:{}/{}"
                               .format(os.getenv("DB_USER"), 
                                       os.getenv("DB_PASSWORD"), 
                                       os.getenv("DB_HOST"), 
                                       os.getenv("DB_PORT"), 
                                       os.getenv("DB_NAME")))
        return connection, engine
    except Exception as e:
        initialize_db()
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        engine = create_engine("postgresql://{}:{}@{}:{}/{}"
                               .format(os.getenv("DB_USER"), 
                                       os.getenv("DB_PASSWORD"), 
                                       os.getenv("DB_HOST"), 
                                       os.getenv("DB_PORT"), 
                                       os.getenv("DB_NAME")))
        return connection, engine

# reset the database
def reset_db():
    # Connect to 'postgres' instead of the target DB
    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database="postgres"  # ✅ Connect to 'postgres' to drop another DB
    )
    connection.autocommit = True
    cursor = connection.cursor()

    db_name = os.getenv("DB_NAME")

    # Drop the database if it exists
    cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")

    # Re-create the database
    cursor.execute(f"CREATE DATABASE {db_name}")

    cursor.close()
    connection.close()

    # Reconnect to the newly created database
    connection, engine = create_connection()
    return connection, engine

# crete a table in from dataframe
def create_table_from_df(table_name, data):
    df = pd.DataFrame(data)
    connection, engine = create_connection()
    df.to_sql(table_name, engine, if_exists="append", index=False)
    connection.commit()
    connection.close()

# read a table from the database
def read_table(table_name):
    connection, engine = create_connection()
    df = pd.read_sql("SELECT * FROM {}".format(table_name), connection)
    connection.commit()
    connection.close()
    return df

# drop a table from the database
def drop_table(table_name):
    connection, engine = create_connection()
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
    cursor.close()
    connection.close()


    