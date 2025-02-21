### README.md (Setup Instructions & Running Streamlit)

---

#### 📌 README.md

# ETL Pipeline for Document Processing

This fully functional UI ETL pipeline extracts text, tables, and figures from PDFs and images using OCR and machine learning. The processed data can be edited, saved, and queried in a database.

## 🚀 Features
- 📄 Extract Text (OCR-based)
- 📊 Extract Tables (with CSV/Excel support)
- 🖼️ Extract Figures (Editable & Downloadable)
- 🛠 Editable Data (Modify extracted data before saving)
- 📥 Upload CSV/Excel (To replace extracted tables & figures)
- 🔍 Vector Database Integration (For querying extracted data)

---

## 🛠️ Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repository/ETL-pipeline-pdf-images.git
cd ETL-pipeline
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Install Tesseract OCR
The pipeline uses Tesseract OCR for text extraction. Install it based on your OS:

- Windows: [Download from Tesseract's official site](https://github.com/UB-Mannheim/tesseract/wiki)
- Ubuntu/Debian:
  ```bash
  sudo apt install tesseract-ocr
  ```
- MacOS (via Homebrew):
  ```bash
  brew install tesseract
  ```

> After installation, ensure Tesseract is in your system's PATH.

### 5️⃣ Set Up Environment Variables
Create a `.env` file inside the project directory and configure database settings:

```env
# PostgreSQL Database Config
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database

# YOLO Model Path
YOLO_MODEL_PATH=models/yolo11_best.pt
```

---

## 🎬 Running the Streamlit App

### 1️⃣ Initialize the Database
Before running the app, initialize the database:
```bash
python -c "from config.db_config import initialize_db; initialize_db()"
```

### 2️⃣ Start the Streamlit App
```bash
streamlit run app.py
```

### 3️⃣ Open the App in Browser
Once started, open your browser and go to:
```
http://localhost:8501
```

---

## 🛠 Folder Structure
```
/ETL-pipeline
│── app.py                   # Main Streamlit application
│── config/
│   ├── db_config.py          # Database configuration
│   ├── vector_db_config.py   # Vector database configuration
│── core/                     # Core ETL logic
│   ├── file_handler.py       # File operations
│   ├── text_extraction.py    # Text extraction and processing
│   ├── table_extraction.py   # Table extraction and processing
│   ├── figure_extraction.py  # Figure extraction and processing
│── models/                   # ML models (YOLO)
│   ├── detection_model.py    # YOLO-based detection
│── upload/                   # Folder for uploaded files
│── static/                   # Static assets if needed
│── requirements.txt          # Dependencies
│── README.md                 # Setup instructions
```

---

## 🏗 Future Improvements
- 🔗 API Integration - Expose endpoints for external data usage.
- 📊 Advanced Table Processing - Improve table structure recognition.
- 🔍 Enhanced Search - Improve vector database similarity search.

---

## 🤝 Contributing
Feel free to submit issues or pull requests for improvements!

📧 Contact: [mk.maarif29@gmail.com](mailto:mk.maarif29@gmail.com)
