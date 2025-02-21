import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
# from inference import get_model
import functools
from PIL import Image
import time

# Import custom scripts
import config.db_config as db
import config.vector_db_config as vdb
import scripts.handle_files as hfiles
import scripts.extract_text as etxt
import scripts.extract_tables as etables
import scripts.extract_figures as efigs

print("Streamlit app running...")
st.set_page_config(layout="wide")

# Import the YOLOv11 model and load the pre-trained weights
# MODEL_ID = 'tft-id-lmwzu/1'
# ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
# model = get_model(MODEL_ID, ROBOFLOW_API_KEY)

# Folder structure setup
BASE_DIR = "upload"
DIRECTORIES = ["files", "res", "img/pages", "img/annotated", "img/texts", "img/tables", "img/figures"]
for dir in DIRECTORIES:
    os.makedirs(os.path.join(BASE_DIR, dir), exist_ok=True)

# clear the directories if they are not empty
def clear_directories():
    for directory in DIRECTORIES:
        dir = f'{BASE_DIR}/{directory}'
        for file in os.listdir(dir):
            os.remove(f'{dir}/{file}')

# Initialize session state variables
if "session_started" not in st.session_state:
    st.session_state.session_started = False

# Initialize the session state variables
if not st.session_state.session_started:
    # reset db
    db.reset_db()
    
    # set the session state variables
    st.session_state.session_started = True

if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False
    st.session_state.files_processed = False
    st.session_state.file_path = ""
    st.session_state.home_page = True
    st.session_state.text_page = False
    st.session_state.table_page = False
    st.session_state.figure_page = False
    st.session_state.data_page = False
    st.session_state.vector_data_saved = False

# Helper function to update session state variables
def complete_file_upload():
    st.session_state.files_uploaded = True

def update_page_state(page):
    st.session_state.home_page = False
    st.session_state.text_page = False
    st.session_state.table_page = False
    st.session_state.figure_page = False
    st.session_state.data_page = False
    if page == "Home":
        st.session_state.home_page = True
    elif page == "Text":
        st.session_state.text_page = True
    elif page == "Tables":
        st.session_state.table_page = True
    elif page == "Figures":
        st.session_state.figure_page = True
    elif page == "All Data":
        st.session_state.data_page = True

def upload_new_file():
    # destroy all session state variables
    st.session_state.clear()
    st.session_state.session_started = True

# Streamlit UI
if not st.session_state.files_uploaded:
    print("Files not uploaded")
    # clear the directories if they are not empty
    clear_directories()

    st.title("📄 ETL Pipeline for Document Processing")
    uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        # Save the uploaded file
        file_extension = uploaded_file.name.split(".")[-1].lower()
        save_path = f"upload/files/{uploaded_file.name}"
        hfiles.save_uploaded_file(uploaded_file, save_path)
        st.session_state.file_path = save_path

        # Button to start processing the uploaded file
        if st.button("Process File", on_click=complete_file_upload):
            st.write("Processing...")

if st.session_state.files_uploaded and not st.session_state.files_processed:
    print("Files uploaded")

    # retrieve the file path
    save_path = st.session_state.file_path
    
    # Split the PDF into images
    if save_path.endswith(".pdf"):
        page_files = hfiles.split_pdf(save_path)
    else:
        page_files = [save_path]
    st.session_state.page_files = page_files

    # Extract text, tables, and figures and save them to the session state
    annotated_images, texts, tables, figures = hfiles.detect_text(page_files)
    st.session_state.annotated_images = annotated_images
    st.session_state.texts = texts
    st.session_state.tables = tables
    st.session_state.figures = figures    

    # initialize text_data session state variable
    if "text_data" not in st.session_state:
        st.session_state.text_data = ""
    if "table_name" not in st.session_state:
        st.session_state.table_name = []
    if "table_data" not in st.session_state:
            st.session_state.table_data = []
    if "figure_name" not in st.session_state:
        st.session_state.figure_name = [None] * len(figures)
    if "figure_data" not in st.session_state:
        st.session_state.figure_data = [None] * len(figures)
        

    # Extract text from the images
    st.session_state.text_data = etxt.extract_text(texts)

    # Update session state variables
    st.session_state.files_processed = True

# SIDEBAR
if st.session_state.files_processed:
    print("Files processed")
    # Retrieve the extracted text, tables, and figures from the session state
    save_path = st.session_state.file_path
    annotated_images = st.session_state.annotated_images
    texts = st.session_state.texts
    text_data = st.session_state.text_data
    tables = st.session_state.tables
    figures = st.session_state.figures

    # Initialize the session state variables

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Text", "Tables", "Figures", "All Data"])
    update_page_state(page)
        
    # sidebar file name
    st.sidebar.title("File Name")
    st.sidebar.write(os.path.basename(save_path))

    # sidebar upload new file
    if st.sidebar.button("Upload New File", on_click=upload_new_file, key="upload_new_file"):
        st.write("Upload a new file")

# Display home page
if st.session_state.files_processed and st.session_state.home_page:
    st.title("📌 Extracted Pages")
    st.write(
        """
        This app extracts text, tables, and figures from a PDF or image file.
        """
    )
    if len(annotated_images) == 0:
        st.write("No pages detected in the document.")
    elif len(annotated_images) < 3:
        # loop to display the extracted pages in 3 columns
        cols = st.columns(len(annotated_images))
        for i, col in enumerate(cols):
            # resize the image to fit the container width
            img = cv2.imread(annotated_images[i])
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            col.image(img, caption=f"Page {i + 1}", use_container_width=True)
    else:
        # loop to display the extracted pages in 3 columns
        for i in range(0, len(annotated_images), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(annotated_images):
                    # resize the image to fit the container width
                    img = cv2.imread(annotated_images[i + j])
                    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    col.image(img, caption=f"Page {i + j + 1}", use_container_width=True)

# Text Page
# functions to process the extracted text
def process_text(text):
    processed_text = etxt.clean_text(text)
    st.session_state.text_data = processed_text

def save_text(editable_text):
    file_path = st.session_state.file_path
    file_name = os.path.basename(file_path)
    etxt.save_text(file_name, editable_text)

if st.session_state.files_processed and st.session_state.text_page:
    # Display the extracted text
    st.title("📝 Extracted Text")
    st.write(
        """
        This page displays the extracted text from the uploaded file.
        """
    )
    # Display the extracted text
    col1, col2 = st.columns([2, 3])

    # coloumn to display the extracted images
    col1.subheader("🖼️ Extracted Text Images")
    with col1.container(height=600):
        # loop to display the extracted text images in 3 columns inside a container
        for i in range(0, len(texts), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(texts):
                    # resize the image to fit the container width
                    img = cv2.imread(texts[i + j])
                    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    col.image(img, caption=f"Text {i + j + 1}", use_container_width=True)

    # coloumn to display the extracted text
    col2_1, col2_2 = col2.columns([1, 1])
    
    # coloumn to display the extracted text
    col2_1.subheader("✍️ Extracted Text (Editable)")
    editable_text = col2_1.text_area("Edit the extracted text", text_data, height=500)

    # coloumn to display the ner table
    entities = etxt.extract_entities(text_data)
    col2_2.subheader("🔍 Top 10 Named Entities")
    col2_2.table(entities.sort_values("Count", ascending=False).reset_index(drop=True).head(15))

    # buttons to process and save the extracted text
    if col2_1.button("Process Text", on_click=lambda: process_text(editable_text), use_container_width=True):
        st.success("Text processed successfully")

    text_file_path = "upload/res/extracted_text.txt"
    if col2_2.button("Save Extracted Text", on_click=lambda: save_text(editable_text), use_container_width=True):
        st.success(f"Text saved in database successfully")

# Tables Page
# Extract figures function
def read_tab_data(i, data_path, name):
    table_name, table_data = etables.read_data(data_path, name)
    # update the session state
    st.session_state.table_name[i] = table_name
    st.session_state.table_data[i] = table_data
    # rerun the page
    time.sleep(0.1)  
    st.rerun()

# Extract tables function
def extract_tables(tables, implicit_rows, implicit_columns, borderless_tables):
    table_name, table_data = etables.extract_table_data(tables, implicit_rows, implicit_columns, borderless_tables)
    st.session_state.table_name = table_name
    st.session_state.table_data = table_data

# Delete table function
def delete_table(i):
    # delete table from the file system
    table_path = st.session_state.table_name[i]
    etables.delete_table(table_path)
    # delete table from session state
    st.session_state.tables.pop(i)
    st.session_state.table_name.pop(i)
    st.session_state.table_data.pop(i)

# Save table function
def save_table(table_name, table_data):
    etables.save_table(table_name, table_data)

# display tables page
if st.session_state.files_processed and st.session_state.table_page:
    tables = st.session_state.tables
    st.title("📊 Extracted Tables")
    st.write(
        """
        This page displays the extracted tables from the uploaded file.
        """
    )
    if len(tables) == 0:
        st.write("No tables detected in the document.")
    else:

        if len(st.session_state.table_data) == 0:
            # Extract the table data
            with st.container(border=True):
                st.subheader("Table Extraction Settings")

                col1, col2, col3 = st.columns(3)
                implicit_rows = col1.checkbox("Implicit Rows", value=False)
                implicit_columns = col2.checkbox("Implicit Columns", value=False)
                borderless_tables = col3.checkbox("Borderless Tables", value=False)

                # Extract the table data
                if st.button("Extract Tables", on_click=lambda: extract_tables(tables, implicit_rows, implicit_columns, borderless_tables)):
                    st.success("Tables extracted successfully")

            # Display the extracted tables
            st.subheader("📊 Extracted Tables")
            for i in range(0, len(tables), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(tables):
                        # resize the image to fit the container width
                        img = cv2.imread(tables[i + j])
                        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        col.image(img, caption=os.path.basename(tables[i + j]), use_container_width=True)
        else:
            # Display the extracted tables
            st.subheader("📊 Extracted Tables")
            uploaded_tab_data = [False] * len(tables)
            upploaded_table_path = [None] * len(figures)
            table_name = [None] * len(tables)
            editable_table = [None] * len(tables)
            csv_name = [None] * len(tables)
            csv_table = [None] * len(tables)

            for i, table in enumerate(tables):
                # container to display the extracted table images and data
                with st.container(border=True):
                    col1, col2 = st.columns([3, 2])
                    col1.image(table, caption=f"Table {i + 1}", use_container_width=True)

                    # col2 to display the extracted table data
                    # col2 to display the extracted figure data
                    col2_1, col2_2 = col2.columns([3, 1], vertical_alignment="center")
                    # upload csv / excel file
                    uploaded_tab_data[i] = col2_1.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key=f"Table_{i+1}_Upload")
                    if uploaded_tab_data[i]:
                        # Save the uploaded file
                        file_extension = uploaded_tab_data[i].name.split(".")[-1].lower()
                        save_path = f"upload/files/{uploaded_tab_data[i].name}"
                        hfiles.save_uploaded_file(uploaded_tab_data[i], save_path)
                        upploaded_table_path[i] = save_path
                        # button to read the uploaded data
                        if col2_2.button("Read Data", key=f"Figure_{i+1}_Read", use_container_width=True):
                            if uploaded_tab_data[i] and uploaded_tab_data[i]:
                                read_tab_data(i, upploaded_table_path[i], uploaded_tab_data[i].name)
                                st.success(f"Data read successfully")
                            else:
                                st.warning("Please upload a file first!")

                    table_name[i] = col2.text_input("Table Name", st.session_state.table_name[i], key=f"Table_{i+1}_Name", max_chars=63)
                    if len(st.session_state.table_data[i]) > 0:
                        table_data = st.session_state.table_data[i]
                        editable_table[i] = col2.text_area("Edit the extracted table", table_data, height=400, key=f"Table_{i+1}_Data")
                    else:
                        editable_table[i] = col2.text_area("Edit the extracted table", "", height=400, key=f"Table_{i+1}_Data")

                    # Buttons
                    col2_1, col2_2, col2_3 = col2.columns([1, 1, 1])
                    # buttons to delete tables with confirmation
                    if col2_1.button("Delete Table", on_click=functools.partial(delete_table, i), key=f"Table_{i+1}_Delete", use_container_width=True):
                        st.success(f"Table deleted successfully")
                    # buttons to download the extracted tables
                    csv_name[i], csv_table[i] = etables.download_csv(table_name[i], editable_table[i])
                    col2_2.download_button(label="Download Table", data=csv_table[i], file_name=f"{table_name[i]}.csv", mime="text/csv", key=f"Table_{i+1}_Download", use_container_width=True)
                    # buttons to save the extracted tables to database
                    if col2_3.button("Save Table", on_click=functools.partial(save_table, table_name[i], editable_table[i]), key=f"Table_{i+1}_Save", use_container_width=True):
                        st.success(f"Table {table_name[i]} saved successfully")
            
            # Save the table data to the session state
            st.session_state.table_name = table_name
            st.session_state.table_data = editable_table

# Figures Page
# Read figure data function
def read_fig_data(i, data_path, name):
    figure_name, figure_data = efigs.read_data(data_path, name)
    # rename figure image
    figure_path = st.session_state.figures[i]
    new_path = f"upload/img/figures/{figure_name}.png"
    os.rename(figure_path, new_path)
    # update the session state
    st.session_state.figure_name[i] = figure_name
    st.session_state.figure_data[i] = figure_data
    st.session_state.figures[i] = new_path
    # rerun the page
    time.sleep(0.1)  
    st.rerun()

# Delete table function
def delete_figure(i):
    # delete figure from session state
    st.session_state.figures.pop(i)
    st.session_state.figure_name.pop(i)
    st.session_state.figure_data.pop(i)

# Save table function
def save_figure(figure_name, figure_df):
    efigs.save_figure(figure_name, figure_df)

# display figures page
if st.session_state.files_processed and st.session_state.figure_page:
    st.title("🖼️ Extracted Figures")
    st.write(
        """
        This page displays the extracted figures from the uploaded file.
        """
    )
    if len(figures) == 0:
        st.write("No figures detected in the document.")
    else:
        # Display the extracted figures
        figure_name = [None] * len(figures)
        uploaded_fig_data = [False] * len(figures)
        figure_data_path = [None] * len(figures)
        figure_data = [None] * len(figures)
        st.subheader("🖼️ Extracted Figures")
        for i, figure in enumerate(figures):
            figure_name[i] = os.path.basename(figure).split(".")[0]
            figure_data[i] = st.session_state.figure_data[i]
            # container to display the extracted figure images and text_area data
            with st.container(border=True):
                col1, col2 = st.columns([3, 2])
                col1.image(figure, caption=f"Figure {i + 1}", use_container_width=True)

                # col2 to display the extracted figure data
                col2_1, col2_2 = col2.columns([3, 1], vertical_alignment="center")
                # upload csv / excel file
                uploaded_fig_data[i] = col2_1.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key=f"Figure_{i+1}_Upload")

                if uploaded_fig_data[i]:
                    # Save the uploaded file
                    file_extension = uploaded_fig_data[i].name.split(".")[-1].lower()
                    save_path = f"upload/files/{uploaded_fig_data[i].name}"
                    hfiles.save_uploaded_file(uploaded_fig_data[i], save_path)
                    figure_data_path[i] = save_path
                    # button to read the uploaded data
                    if col2_2.button("Read Data", key=f"Figure_{i+1}_Read", use_container_width=True):
                        if figure_data_path[i] and uploaded_fig_data[i]:
                            read_fig_data(i, figure_data_path[i], uploaded_fig_data[i].name)
                            st.success(f"Data read successfully")
                        else:
                            st.warning("Please upload a file first!")


                # text_area to input the extracted figure data
                figure_name[i] = col2.text_input("Figure Name", value=st.session_state.figure_name[i] if st.session_state.figure_name[i] else "", key=f"Figure_{i+1}_Name")
                # figure_data[i] = st.session_state.figure_data[i]
                figure_data[i] = col2.text_area("Edit the extracted figure data", st.session_state.figure_data[i], height=400, key=f"Figure_{i+1}_Data")
                # Buttons
                col2_1, col2_2, col2_3 = col2.columns([1, 1, 1])
                # buttons to delete figures with confirmation
                if col2_1.button("Delete Figure", on_click=functools.partial(delete_figure, i), key=f"Figure_{i+1}_Delete", use_container_width=True):
                    st.success(f"Figure deleted successfully")
                # buttons to download the images
                col2_2.download_button(label="Download Figure", data=open(figure, 'rb').read(), file_name=f"Figure_{i+1}.png", mime="image/png", key=f"Figure_{i+1}_Download", use_container_width=True)
                # buttons to save the extracted figures to database
                if col2_3.button("Save Data", on_click=functools.partial(save_figure, figure_name[i], figure_data[i]), key=f"Figure_{i+1}_Save", use_container_width=True):
                    st.success(f"Figure {figure_name[i]} saved successfully")

        # Save the figure data to the session state
        st.session_state.figure_name = figure_name
        st.session_state.figure_data = figure_data

# All Data Page
# Combine all extracted data into one string
def combine_all_data():
    text_data = st.session_state.text_data
    table_data = st.session_state.table_data
    figure_data = st.session_state.figure_data
    all_data = f"{text_data}\n\n"
    for table in table_data:
        all_data += f"{table}\n\n"
    for figure in figure_data:
        all_data += f"{figure}\n\n"
    return all_data

# Save the combined data to the database
def save_vector_data():
    all_data = combine_all_data()
    title = os.path.basename(st.session_state.file_path)
    vdb.add_data_to_vector_store(title, all_data)
    st.session_state.vector_data_saved = True
    # rerun the page
    time.sleep(0.1)  
    st.rerun()

# Similarity search
def search_data(search_text):
    file_name = os.path.basename(st.session_state.file_path)
    results = vdb.query_vector_store(query=search_text)
    return results

# Display all data
if st.session_state.files_processed and st.session_state.data_page:
    st.title("📄 All Extracted Data")
    st.write(
        """
        This page displays all the extracted data from the uploaded file.
        """
    )
    st.markdown('''
            <style>
            .fullHeight {height : 80vh;
                  width : 100%}
            </style>''', unsafe_allow_html = True)
    
    # 3 columns to display the extracted text, tables, and figures with scrollbars and full height
    col1, col2, col3 = st.columns(3)
    with col1.container(height=600):
        st.subheader("📝 Extracted Text")
        st.write(st.session_state.text_data)
    with col2.container(height=600):
        st.subheader("📊 Extracted Tables")
        for i, table in enumerate(st.session_state.table_data):
            table_name, table_data = etables.process_table_data(st.session_state.table_name[i], table)
            st.write(table_name)
            st.table(table_data)

    with col3.container(height=600):
        st.subheader("🖼️ Extracted Figures")
        for i, figure in enumerate(st.session_state.figure_data):
            try:
                figure_name, figure_data = efigs.process_figure_data(st.session_state.figure_name[i], figure)
                st.write(figure_name)
                st.write(figure_data)
            except:
                st.write("No data available")

    # Buttons to save the extracted data to vector database
    if col1.button("Save Data to Vector Database", on_click=save_vector_data, use_container_width=True):
        st.success("Data saved to vector database successfully")
    
    # Similarity search
    if st.session_state.vector_data_saved:
        st.subheader("🔍 Similarity Search")
        search_text = st.text_input("Search Text", "")
        if st.button("Search", key="search"):
            # Perform similarity search
            results = search_data(search_text)
            if len(results) == 0:
                st.write("No similar data found.")
            else:
                st.write("Similar data found:")
                for i, (res, score) in enumerate(results):
                    with st.container(border=True):
                        st.write(f"Result {i+1}, Similarity Score: {score}")
                        st.write(f"{res.page_content}")
                        st.write(f"{res.metadata}")

