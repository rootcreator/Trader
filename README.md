# Forex Prediction System

## Overview

The Forex Prediction System is a Django-based web application designed to predict Forex rates using a combination of ensemble models. It fetches historical and current trend data, processes it through various technical, machine learning, and risk management models, and makes predictions about future Forex rates.

## Features

- Fetches and processes historical Forex data.
- Uses a variety of models including Simple Moving Average (SMA), RSI, MACD, Bollinger Bands, and several machine learning models.
- Employs ensemble techniques to improve prediction accuracy.
- Integrates risk management models for better decision-making.
- Provides a user-friendly interface for making predictions.

## Models Used

### Technical Models
- **Simple Moving Average (SMA)**
- **Relative Strength Index (RSI)**
- **Moving Average Convergence Divergence (MACD)**
- **Bollinger Bands**

### Machine Learning Models
- **Linear Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **ARIMA**

### Risk Management Models
- **Fixed Fraction Model**
- **Kelly Criterion Model**
- **Expected Value Model**

### Forex Specific Models
- **Mean Reversion Model**
- **Carry Trade Model**
- **Volatility Model**

### Ensemble Model
- Combines predictions from various models using a weighted approach to enhance overall prediction accuracy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rootcreator/trader.git
   cd forex-prediction-system
   python -m venv venv
   pip install -r requirements.txt
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   python manage.py migrate
   python manage.py runserver

