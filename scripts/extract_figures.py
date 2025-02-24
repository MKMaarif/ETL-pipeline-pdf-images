import pandas as pd
import config.db_config as db
import re

# Read data from the CSV/Excel file
def read_data(file_path, file_name):
    if file_path.endswith(".csv"):
        fig_data = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        fig_data = pd.read_excel(file_path)
    else:
        raise ValueError("Invalid file format. Please provide a CSV or Excel file.")
    
    # Given figure name
    figure_name = file_name.split(".")[0]
    figure_name = figure_name.replace("\n", "_").replace(" ", "_").replace(".", "")
    figure_name = figure_name[:63]
    
    # Handle data
    fig_data = fig_data.fillna("None")
    # round the float columns
    for col in fig_data.select_dtypes(include=["float"]).columns:
        fig_data[col] = fig_data[col].round(2)
    # Convert all columns to string for visualization
    fig_data.columns = [str(col) + "  " for col in fig_data.columns]
    fig_data = fig_data.map(lambda x: str(x) + "  ")
    fig_data = fig_data.to_string(index=False)

    return figure_name, fig_data

# process the figure data
def process_figure_data(figure_name, figure_data):
    
    if len(figure_data) == 0:
        return figure_name, None
    
    # Given figure name
    figure_name = figure_name.replace("\n", "_").replace(" ", "_").replace(".", "")
    # cut the figure name to 63 characters
    figure_name = figure_name[:63]
    
    # Given figure string
    figure_data = figure_data

    # Convert figure string into a structured list
    lines = figure_data.split("\n")
    parsed_data = [re.split(r"\s{2,}", line.strip()) for line in lines]

    # Extract column headers and rows
    headers = parsed_data[0]  # Use the first row as headers
    data_rows = parsed_data[1:]  # Remaining rows

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data_rows, columns=headers)

    # Replace "None" string with actual None values
    df.replace("None", None, inplace=True)

    return figure_name, df


def save_figure(figure_name, figure_df):
    """Save figure to database."""
    name, df = process_figure_data(figure_name, figure_df)
    # create table
    db.create_table_from_df(name, df)
