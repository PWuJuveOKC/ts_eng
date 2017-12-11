# ReadMe
Code for time series feature engineering project

## Similarity.py
Calculate similarity for 4 types of medical data (ECG/PAP/ART/CO2) using Euclidean and FastDTW distance, each type has 20 samples. Plots outputted to Output folder. Time series data length is restricted to 600.

## Sample_Extract.py
Use same sample data as in similarity.py and extracted features using tsfresh. Times series are scaled before feature extraction.

## TSHeatmap.R
Display features extracted from sample medical data, 4 groups denote 4 types in rowside 