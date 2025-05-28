# Housing Price Prediction – Machine Learning Pipeline

This project implements a complete machine learning pipeline in Python to predict housing prices. It includes feature scaling, model training, and a deployable terminal-based app for making predictions.

Developed as part of the **Data Mining and Machine Learning** course from the Bachelor of Information Sciences – Major in Data Science at Massey University.

## 📘 Course Context

- **Course:** Data Mining and Machine Learning  
- **Semester:** Semester 1, 2022  
- **University:** Massey University  
- **Program:** Bachelor of Information Sciences – Major in Data Science

## ⚙️ Project Overview

The pipeline includes:

- Data cleaning and feature engineering
- Feature and target scaling using `StandardScaler`
- Model training with regression algorithms
- Saving models and scalers with `joblib`
- Creating a Python CLI app for loading the model and performing predictions

## 🧪 Files Included

- `FinalCollection.ipynb` – Full notebook with code, plots, and model training
- `Assignment3.html` – HTML export of the notebook
- `scaler_features.joblib` – Scaler object used for feature normalization
- `scaler_target.joblib` – Scaler for target variable (e.g., price)
- `app_terminal.py` – Terminal application that loads the trained model and makes predictions

## 💡 Technologies Used

- Python
- pandas
- scikit-learn
- joblib
- Jupyter Notebook

## ▶️ How to Use

1. Open and explore `FinalCollection.ipynb` to see the full pipeline.
2. Run `app_terminal.py` to enter features and get a housing price prediction.

> Make sure you have `joblib`, `scikit-learn`, and `pandas` installed in your environment.

## 📌 Notes

- This project is for educational purposes and does not use a production-ready deployment framework.
- The model can be improved with more robust feature selection and cross-validation.

## 🏁 Sample Terminal App Usage

```bash
python app_terminal.py
