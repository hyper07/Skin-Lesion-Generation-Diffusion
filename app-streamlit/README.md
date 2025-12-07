# Capstone AML Streamlit Application

This is the Streamlit web application for the Capstone AML Project (Columbia University SPS × Société Générale).

## Overview

The Streamlit app provides an interactive interface for:
- Exploratory Data Analysis (EDA) of AML transaction data
- Anomaly Detection Analytics and visualization
- Model results exploration

## Structure

```
app-streamlit/
├── pages/                   # Streamlit pages
│   ├── 1_EDA.py            # EDA page
│   └── 2_Anomaly_Detection_Analytics.py  # Anomaly detection page
├── functions/               # Utility functions
│   ├── components.py
│   ├── database.py
│   ├── downloaderCSV.py
│   ├── eda_components.py
│   ├── menu.py
│   └── visualization.py
├── images/                  # Static images
├── styles/                  # CSS styles
│   └── app.css
├── aml/                     # AML integration folder mapping
├── Home.py                  # Main Streamlit app entry
├── utils.py                 # Utility functions
├── Dockerfile               # Streamlit container
├── requirements.txt         # Python dependencies
└── requirements-dev.txt     # Development dependencies
```

## Running the Application

### Using Docker Compose (Recommended)

From the project root:

```bash
# Start all services
docker-compose up --build

# Or start only Streamlit
docker-compose up --build capstone-streamlit-app
```

Access the app at: http://localhost:58502

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run Home.py
```

## Features

- **EDA Page**: Interactive exploratory data analysis with visualizations
- **Anomaly Detection Analytics**: Model results, risk scoring, and analytics dashboard
- **Database Integration**: Connects to PostgreSQL with pgvector for data storage
- **CSV Download**: Export filtered results and analysis data

## Dependencies

See `requirements.txt` for Python dependencies and `requirements-dev.txt` for development tools.