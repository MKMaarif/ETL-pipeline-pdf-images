from dotenv import load_dotenv
import os
import cv2
import pytesseract
import spacy.cli
import re
import spacy
import pandas as pd
import config.db_config as db

load_dotenv()

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Download the English model for spaCy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Extract text from the images
def extract_text(pages):
    """Extracts text from a PDF by converting it to images and applying OCR."""
    
    text_data = ""

    for page in pages:
        page_img = cv2.imread(page)
        text_data += pytesseract.image_to_string(page_img)
        
    return text_data

# clean the extracted text
def clean_text(text):
    # Remove hyphenation at line breaks (join words split across lines)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Remove extra newlines that break sentences
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove reference numbers in square brackets (e.g., [1], [21])
    text = re.sub(r"\[\d+\]", "", text)

    return text

# ner the extracted text
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # make a data frame
    entities_df = pd.DataFrame()
    entities_df["Entity"] = [ent[0] for ent in entities]
    entities_df["Label"] = [ent[1] for ent in entities]

    # add a column for the count of each entity type
    entities_df["Count"] = entities_df.groupby(["Entity", "Label"])["Entity"].transform("count")
    entities_df = entities_df.drop_duplicates()

    return entities_df

# process the extracted text
def process_text(text_data):
    """Processes the extracted text."""
    
    # Clean the text
    cleaned_text = clean_text(text_data)
    
    # Extract entities from the text
    entities = extract_entities(cleaned_text)
    
    return cleaned_text, entities

def split_text(text, chunk_size=10000):
    """Splits the text into smaller chunks of specified size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# save the processed text
def save_text(file_name, text):
    """Save text and its embedding to PostgreSQL"""

    connection, engine = db.create_connection()
    cursor = connection.cursor()


    # create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS texts (
            id SERIAL PRIMARY KEY,
            filename TEXT
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_text_chunks (
            id SERIAL PRIMARY KEY,
            texts_id INTEGER REFERENCES texts(id) ON DELETE CASCADE,
            chunk_order INTEGER,
            content TEXT
        );
    """)

    # insert the text
    cursor.execute("""
        INSERT INTO texts (filename)
        VALUES (%s)
        RETURNING id;
    """, (file_name,))
    text_id = cursor.fetchone()[0]

    # split the text into chunks
    text_chunks = split_text(text)
    for i, chunk in enumerate(text_chunks):
        cursor.execute("""
            INSERT INTO pdf_text_chunks (texts_id, chunk_order, content)
            VALUES (%s, %s, %s);
        """, (text_id, i, chunk))

    connection.commit()
    cursor.close()
    connection.close()