# tap22
# Tap Bonds AI Platform

A sophisticated bond discovery and analysis platform powered by Streamlit and GROQ.

## Features

- Bond details search and visualization
- Cash flow analysis
- Company insights
- Natural language queries for bond information
- Web search capabilities

## Deployment to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select this repository
6. Set the main file path to `main.py`
7. Add your GROQ API key in the Streamlit Cloud secrets section

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## Data Files

This application requires three types of CSV files to function properly:
- Bond details CSV
- Cashflow details CSV
- Company insights CSV

Upload these files through the sidebar when running the application.
