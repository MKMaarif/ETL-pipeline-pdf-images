import cv2
from img2table.document import Image
from img2table.ocr import TesseractOCR
import pandas as pd
import re
import os
import streamlit as st
import config.db_config as db

def extract_table_data(tables, implicit_rows, implicit_columns, borderless_tables):
    """Extracts table data from the images using img2table."""
    ocr = TesseractOCR(n_threads=4, lang="eng")
    title = []
    data = []
    for table in tables:
        table_image = Image(src=table)
        extracted_data = table_image.extract_tables(ocr=ocr, 
                                                implicit_rows=implicit_rows,
                                                implicit_columns=implicit_columns,
                                                borderless_tables=borderless_tables)
        if len(extracted_data) == 0:
            table_title = ""
            table_data = ""
        else:
            if extracted_data[0].title is None:
                table_title = "Table"
            else:
                table_title = extracted_data[0].title.replace("\n", "_").replace(" ", "_").replace(".", "")
            table_title = table_title[:63]
            table_data = extracted_data[0].df
            table_data.columns = table_data.iloc[0]
            table_data = table_data[1:]
            table_data.columns = [str(col) + "  " for col in table_data.columns]
            table_data = table_data.map(lambda x: str(x) + "  ")
            table_data = table_data.to_string(index=False)

        title.append(table_title)
        data.append(table_data)
    return title, data

# process the table data
def process_table_data(table_name, table_data):
    
    if len(table_data) == 0:
        return table_name, None
    
    # Given table name
    table_name = table_name.replace("\n", "_").replace(" ", "_").replace(".", "")
    # cut the table name to 63 characters
    table_name = table_name[:63]
    
    # Given table string
    table_data = table_data

    # Convert table string into a structured list
    lines = table_data.split("\n")
    parsed_data = [re.split(r"\s{2,}", line.strip()) for line in lines]

    # Extract column headers and rows
    headers = parsed_data[0]  # Use the first row as headers
    data_rows = parsed_data[1:]  # Remaining rows

    # Handle duplicate headers
    headers = [f"{header}_{i}" if headers.count(header) > 1 else header for i, header in enumerate(headers)]

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data_rows, columns=headers)

    # Replace "None" string with actual None values
    df.replace("None", None, inplace=True)

    return table_name, df

def read_data(file_path, file_name):
    if file_path.endswith(".csv"):
        table_data = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        table_data = pd.read_excel(file_path)
    else:
        raise ValueError("Invalid file format. Please provide a CSV or Excel file.")
    
    # Given table name
    table_name = file_name.split(".")[0]
    table_name = table_name.replace("\n", "_").replace(" ", "_").replace(".", "")
    table_name = table_name[:63]
    
    # Handle data
    table_data = table_data.fillna("None")
    # round the float columns
    for col in table_data.select_dtypes(include=["float"]).columns:
        table_data[col] = table_data[col].round(2)
    # Convert all columns to string for visualization
    table_data.columns = [str(col) + "  " for col in table_data.columns]
    table_data = table_data.map(lambda x: str(x) + "  ")
    table_data = table_data.to_string(index=False)

    return table_name, table_data

def download_csv(table_name, table_df):
    name, df = process_table_data(table_name, table_df)
    df = pd.DataFrame(df)
    csv_dest = f"upload/res/{name}.csv"
    df.to_csv(csv_dest, index=False)
    csv = df.to_csv(index=False).encode("utf-8")
    return name, csv

def save_table(table_name, table_df):
    """Save table to database."""
    name, df = process_table_data(table_name, table_df)
    db.create_table_from_df(name, df)

def delete_table(table_name):
    """Deletes the table file."""
    csv_dest = f"upload/res/{table_name}.csv"
    try:
        os.remove(csv_dest)
    except FileNotFoundError:
        pass