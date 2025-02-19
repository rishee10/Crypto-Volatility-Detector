# Crypto-Volatility-Detector

## Problem or Question:
The goal of this project is to detect and predict cryptocurrency volatility, which is crucial for traders and analysts to make informed decisions. By identifying potential price fluctuations, we aim to provide insights into market dynamics and help in managing investment risk.

## Approach:
In this project, we focus on detecting cryptocurrency volatility using historical market data. The main steps include:

Data Collection and Cleaning: We start by loading historical crypto data (e.g., from a CSV file) and preprocess it by converting the necessary columns to the appropriate data types. The data includes market indicators like Open, High, Low, Close, and Volume.

Volatility Indicator Calculation: We calculate volatility indicators, such as the daily percentage returns, rolling standard deviation (for volatility), Bollinger Bands (upper and lower bounds for volatility), and Average True Range (ATR). These indicators help quantify the price fluctuations over time.

Machine Learning Model: A RandomForestRegressor is trained to predict the next day's volatility (measured by standard deviation) based on the features like current price, ATR, and standard deviation. The model is evaluated using Mean Absolute Error (MAE) to assess its prediction accuracy.

Visualization: The volatility is visualized using Bollinger Bands, showing the historical price and volatility ranges over time. This helps to see how volatility interacts with the cryptocurrency price.
The solution provides a clear way to understand past volatility patterns and predict future price fluctuations. This can be used for both short-term trading strategies and long-term market analysis.

## Tools:
pandas: For data manipulation and cleaning.

numpy: For numerical calculations.

matplotlib: For visualizing price and volatility trends.

scikit-learn: For training the RandomForestRegressor model and splitting data into training and test sets.

RandomForestRegressor: To predict future volatility based on historical data.

mean_absolute_error: To evaluate the accuracy of the model.

## Install dependencies: 

### Steps to Set Up the Virtual Environment

#### Clone the Repository 

``` git clone https://github.com/rishee10/Crypto-Volatility-Detector.git  ```

``` cd Crypto-Volatility-Detector ```

#### Create a Virtual Environment

``` python -m venv venv ```

#### Activate the Virtual Environment

``` .\venv\Scripts\activate ```

#### Install the Required Dependencies

``` pip install -r requirements.txt ```

#### Run The Program

``` python main.py ```

#### Model Evaluation:

The model’s performance is assessed with Mean Absolute Error (MAE).

#### Usage: 

The trained model can be used to predict future volatility and provide insights into market risk.
 


