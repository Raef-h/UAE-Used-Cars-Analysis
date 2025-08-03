import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

try:
  
    df = pd.read_csv("UAE Used Cars Analysis.csv")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12)) 

    # ................................................................1 - Histogram (Price Distribution)
    city_car_count = df['Location'].value_counts().sort_values(ascending=False)

    axs[0, 0].barh(
         city_car_count.index,
         city_car_count.values,
         color = ['#C8102E', '#007A33', '#000000'] * (len(city_car_count) // 3 + 1)
    )

    axs[0, 0].set_title('Number of Cars in Each City', fontsize=10)
    axs[0, 0].set_xlabel('Number of Cars', fontsize=8)
    axs[0, 0].set_ylabel('City', fontsize=8)

    for i, v in enumerate(city_car_count.values):
        axs[0, 0].text(v + 5, i, str(v), va='center', fontsize=6.5, color="#000000FF")

    axs[0, 0].invert_yaxis()
    # ................................................................2 - Pie Chart (Fuel Type Distribution)
    labels_order = ['Electric', 'Diesel', 'Hybrid', 'Gasoline']
    fuel_type_ordered = df['Fuel Type'].value_counts()[labels_order]
    explode_values = [0, 0, 0, 0]
    fuel_colors = {
        'Electric': '#00BFFF',
        'Diesel': "#E6D447",
        'Hybrid': '#32CD32',
        'Gasoline': "#BE786A"
    }
    custom_colors = [fuel_colors[label] for label in labels_order]
    axs[0, 1].pie(fuel_type_ordered.values, labels=fuel_type_ordered.index, autopct='%1.1f%%', 
                 startangle=170, colors=custom_colors, explode=explode_values)
    axs[0, 1].set_title('Fuel Type Distribution', fontsize=10)

    # ................................................................3 - Line Plot (Number of Cars by Year)
    car_count_by_year = df['Year'].value_counts().sort_index()
    car_count_by_year = df['Year'].value_counts().sort_index()
    axs[1, 0].plot(car_count_by_year.index, car_count_by_year.values, marker='o', color='#2A9D8F', linestyle='-', linewidth=2)
    axs[1, 0].set_title('Number of Cars by Year', fontsize=10)
    axs[1, 0].set_xlabel('Year', fontsize=8)
    axs[1, 0].set_ylabel('Number of Cars', fontsize=8)
    axs[1, 0].grid(True)
    axs[1, 0].set_xticks(range(2005, 2026, 2))

    # ................................................................4 - Bar Chart (Top 10 Car Makes by Highest Price)
    highest_price_by_make = df.groupby('Make')['Price'].max().sort_values(ascending=False)
    top_10_highest_price = highest_price_by_make.head(10)
    formatted_prices = top_10_highest_price.apply(lambda x: f'{x/1_000_000:.2f}M' if x >= 1_000_000 else str(x))
    colors = ["#D4AF37", '#C0C0C0', '#C47E55'] + ['#708090'] * 7

    axs[1, 1].bar(
        formatted_prices.index,
        top_10_highest_price.values,
        color=colors
    )

    axs[1, 1].set_title('Top 10 Car Makes by Highest Car Price', fontsize=10)
    axs[1, 1].set_xlabel('Car Make', fontsize=8)
    axs[1, 1].set_ylabel('Highest Car Price (AED)', fontsize=8)
    axs[1, 1].tick_params(axis='x', rotation=45)

    for i, v in enumerate(top_10_highest_price.values):
        axs[1, 1].text(i, v + 5000, f'{v/1_000_000:.2f}M', ha='center', fontsize=8, color='black')

    plt.tight_layout()

    st.pyplot(fig)

    # -------------------------- Second Figure (3x1 Grid for 3 additional plots)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  

    # ................................................................5 - Bar Chart (Top 10 Car Models by Highest Price)
    highest_price_by_model = df.groupby('Model')['Price'].max().sort_values(ascending=False)
    top_10_highest_price_models = highest_price_by_model.head(10)
    formatted_prices_models = top_10_highest_price_models.apply(lambda x: f'{x/1_000_000:.2f}M' if x >= 1_000_000 else str(x))
    axs[0, 0].bar(formatted_prices_models.index, top_10_highest_price_models.values, color=sns.color_palette("pastel")[4])
    axs[0, 0].set_title('Top 10 Car Models by Highest Car Price', fontsize=10)
    axs[0, 0].set_xlabel('Car Model', fontsize=8)
    axs[0, 0].set_ylabel('Highest Car Price (AED)', fontsize=8)
    axs[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(top_10_highest_price_models.values):
        axs[0, 0].text(i, v + 5000, f'{v/1_000_000:.2f}M', ha='center', fontsize=8, color='red')

    # ................................................................6 - Pie Chart (Car Transmission Type Distribution)
    transmission_count = df['Transmission'].value_counts()
    axs[0, 1].pie(transmission_count.values, labels=transmission_count.index, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette("pastel", len(transmission_count)), explode=[0, 0.1])
    axs[0, 1].set_title('Car Transmission Type Distribution', fontsize=10)

    # ................................................................7 - Horizontal Bar Chart (Top 10 Car Makes by Number of Cars)
    car_count_by_make = df['Make'].value_counts().sort_values(ascending=False)
    top_10_car_count = car_count_by_make.head(10)
    axs[1, 0].barh(top_10_car_count.index, top_10_car_count.values, color=sns.color_palette("pastel")[3])
    axs[1, 0].set_title('Top 10 Car Makes by Number of Cars', fontsize=10)
    axs[1, 0].set_xlabel('Number of Cars', fontsize=8)
    axs[1, 0].set_ylabel('Car Make', fontsize=8)
    for i, v in enumerate(top_10_car_count.values):
        axs[1, 0].text(v + 5, i, str(v), va='center', fontsize=6, color='red')
    axs[1, 0].invert_yaxis()

    # ................................................................8 - Line Plot (Top 9 Color Distribution by Total Number of Cars)
    color_count = df['Color'].value_counts().head(9)
    axs[1, 1].plot(color_count.index, color_count.values, marker='o', color=sns.color_palette("pastel")[5], linestyle='-', linewidth=2)
    axs[1, 1].set_title('Top 9 Color Distribution by Total Number of Cars', fontsize=10)
    axs[1, 1].set_xlabel('Color', fontsize=8)
    axs[1, 1].set_ylabel('Number of Cars', fontsize=8)
    axs[1, 1].grid(True)

    
    plt.tight_layout()


    st.pyplot(fig)


    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # ................................................................9 - Linear Regression Model for Car Count Prediction
    
    car_count_by_year = df['Year'].value_counts().sort_index()

   
    years = car_count_by_year.index.values.reshape(-1, 1)  
    car_count = car_count_by_year.values  

    
    model = LinearRegression()

  
    model.fit(years, car_count)

   
    future_years = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    predicted_car_count = model.predict(future_years)

    
    axs[0].scatter(years, car_count, color='blue', label='Actual Data')

  
    axs[0].plot(future_years, predicted_car_count, color='red', label='Predicted Data')

    for i, v in enumerate(predicted_car_count):
        axs[0].text(future_years[i], v, str(int(v)), color='red', ha='center', fontsize=10)

    axs[0].set_title('Car Count Prediction for Next Years', fontsize=16)
    axs[0].set_xlabel('Year', fontsize=12)
    axs[0].set_ylabel('Number of Cars', fontsize=12)

    # ................................................................10 - Total Car Prices and Future Predictions
    price_2025 = df['Price'].sum() 

    increase_percentage = 0.05  

    years_future = np.array([2026, 2027, 2028, 2029, 2030])

    predicted_price = [price_2025 * ((1 + increase_percentage) ** (year - 2025)) for year in years_future]


    def format_price(price):
        if price >= 1_000_000_000:
            return f"{price / 1_000_000_000:.2f}B" 
        elif price >= 1_000_000:
            return f"{price / 1_000_000:.2f}M"  
        else:
            return f"{price:.2f}" 


    axs[1].bar(2025, price_2025, color='blue', width=0.3, label='Total Price in 2025')


    formatted_price_2025 = format_price(price_2025)
    axs[1].text(2025, price_2025 + (price_2025 * 0.01), formatted_price_2025, color='blue', ha='center', fontsize=12)


    axs[1].plot(years_future, predicted_price, color='red', label='Predicted Price', linestyle='--', marker='o')

    for i, v in enumerate(predicted_price):
        formatted_price = format_price(v) 
        axs[1].text(years_future[i], v + (v * 0.01), formatted_price, color='red', ha='center', fontsize=10)

    axs[1].set_title('Car Price Prediction for Next Years', fontsize=16)
    axs[1].set_xlabel('Year', fontsize=12)
    axs[1].set_ylabel('Price (AED)', fontsize=12)

    plt.tight_layout()


    st.pyplot(fig)

except Exception as e:
    st.error(f"The error occurred while reading the file.: {str(e)}")


