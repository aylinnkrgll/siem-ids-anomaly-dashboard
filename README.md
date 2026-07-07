# SIEM IDS Anomaly Dashboard

This project is an academic SIEM/IDS dashboard developed with Streamlit.

It analyzes network traffic flows and provides anomaly/risk-based detection outputs using machine learning models such as Random Forest, Isolation Forest and Autoencoder.

## Features

- Network traffic analysis with CSV input
- Random Forest based classification
- Isolation Forest based anomaly detection
- Autoencoder based reconstruction error analysis
- Risk level calculation
- Model disagreement analysis
- Streamlit-based interactive dashboard
- Example CSV file for testing

## Project Files

- `appv.py`: Main Streamlit dashboard application
- `siem.ipynb`: Model development and analysis notebook
- `requirements.txt`: Python dependencies
- `sample_network_flows.csv`: Small example input file for testing
- `.gitignore`: Ignored local and unnecessary files

## Dataset

This project was developed using the CICIDS2017 dataset.

The full dataset is not included in this repository due to file size.  
Only a small sample CSV file is provided for testing the dashboard.

## Installation

```bash
pip install -r requirements.txt
