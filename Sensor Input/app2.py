import requests
import serial
import asyncio
import threading
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Configure serial connection
ser = serial.Serial('COM6', 9600)  # Replace 'COM6' with your port and 9600 with your baud rate

# Global variables to hold sensor data
sensor_data = {
    "N": 0.0,
    "P": 0.0,
    "K": 0.0,
    "PH": 0.0
}

# Importing model and scalers
model = pickle.load(open('random_forest_model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Your Weatherstack API Key
API_KEY = '091dc1f48fe3c588fd3b05a505f07e71'  # Replace with your actual Weatherstack API key
BASE_URL = "http://api.weatherstack.com/current"

def get_weather(location):
    """Fetches current weather data for a specified location."""
    params = {
        'access_key': API_KEY,
        'query': location
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if 'current' in data:
            current_weather = data['current']
            temperature = current_weather['temperature']
            humidity = current_weather['humidity']
            return temperature, humidity
        else:
            print(f"Error fetching data: {data.get('error', {}).get('info', 'Unknown error')}")
            return None, None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

async def read_serial():
    """Background task to read serial data from Arduino and update sensor_data."""
    while True:
        try:
            data = ser.readline().decode('utf-8').strip()  # Read and decode the serial data
            # Assume the data comes in the format: "N:12.5,P:10.0,K:8.0,PH:6.5"
            parts = data.split(',')
            for part in parts:
                key, value = part.split(':')
                if key in sensor_data:
                    sensor_data[key] = float(value)
            print(f"Updated Sensor Data: {sensor_data}")
        except Exception as e:
            print(f"Error reading serial data: {e}")

def start_background_loop(loop):
    """Starts the asyncio event loop in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_until_complete(read_serial())

# Start the asyncio loop in a background thread
loop = asyncio.new_event_loop()
t = threading.Thread(target=start_background_loop, args=(loop,))
t.start()

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Fetch the current weather data
    location = request.form.get('Location')  # Get the location from the form
    temp, humidity = get_weather(location)

    if temp is None or humidity is None:
        return "Failed to fetch weather data. Please try again later.", 500

    # Use sensor data for N, P, K, and PH values
    N = sensor_data["N"]
    P = sensor_data["P"]
    K = sensor_data["K"]
    ph = sensor_data["PH"]

    # Values entered by user
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scaling the input features
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Predicting probabilities
    probabilities = model.predict_proba(final_features)[0]

    # Sorting and selecting top 3 indices
    top_3_indices = np.argsort(probabilities)[-3:][::-1]

    # Mapping indices to crop names
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    result_image = {
        1: "static/crops/Rice.jpeg", 2: "static/crops/Maize.jpeg", 3: "static/crops/Jute.jpeg",
        4: "static/crops/Cotton.jpeg", 5: "static/crops/Coconut.jpeg", 6: "static/crops/Papaya.jpeg",
        7: "static/crops/Orange.jpeg", 8: "static/crops/Apple.jpeg", 9: "static/crops/Muskmelon.jpeg",
        10: "static/crops/Watermelon.jpeg", 11: "static/crops/Grapes.jpeg", 12: "static/crops/Mango.jpeg",
        13: "static/crops/Banana.jpeg", 14: "static/crops/Pomegranate.jpeg", 15: "static/crops/Lentil.jpeg",
        16: "static/crops/Blackgram.jpeg", 17: "static/crops/Mungbean.jpeg", 18: "static/crops/Mothbeans.jpeg",
        19: "static/crops/Pigeonpeas.jpeg", 20: "static/crops/Kidneybeans.jpeg", 21: "static/crops/Chickpea.jpeg",
        22: "static/crops/Coffee.jpeg"
    }

    crop_description = {
                        "Rice": "Rice is predominantly cultivated in Asia, especially in regions like China, India, and Southeast Asia. It thrives in clayey, loamy soils that retain water well.",
                        "Maize" : "Maize, or corn, is widely grown in the United States, Brazil, and China. It prefers well-drained, fertile loamy soils rich in organic matter." , 
                        "Jute" : "Jute is mainly cultivated in Bangladesh and India, particularly in the Ganges Delta. It grows best in alluvial soil with a high clay content." , 
                        "Cotton" : "Cotton is densely grown in India, the United States, and China. It favors deep, well-drained sandy loam soils with a slightly acidic to neutral pH." , 
                        "Coconut" : "Coconut palms are most commonly found in tropical coastal regions like Indonesia, the Philippines, and India. They grow well in sandy, loamy, or alluvial soils that are well-drained." , 
                        "Papaya" : "Papaya is extensively grown in India, Brazil, and Mexico. It thrives in well-drained sandy loam or alluvial soils rich in organic matter." , 
                        "Orange" : "Oranges are primarily cultivated in Brazil, the United States (especially Florida), and China. They prefer well-drained sandy loam soils rich in organic content." ,
                        "Apple" : " Apples are widely grown in temperate regions like the United States, China, and Europe. They thrive in well-drained loamy soils with a slightly acidic pH." , 
                        "Muskmelon" : "Muskmelon is grown in warmer climates such as in India and the United States. It prefers sandy loam soils that are well-drained and rich in organic matter." , 
                        "Watermelon" : "Watermelon is cultivated in warm regions like China, Turkey, and the United States. It grows best in sandy loam soils with good drainage and moderate organic content." , 
                        "Grapes" : "Grapes are primarily cultivated in Mediterranean climates like those found in Italy, Spain, and France. They thrive in well-drained loamy or sandy loam soils with good fertility." , 
                        "Mango" : "Mangoes are extensively grown in India, Thailand, and Mexico. They favor well-drained alluvial or loamy soils rich in organic matter." , 
                        "Banana" : "Bananas are widely cultivated in tropical regions such as India, Brazil, and Ecuador. They thrive in well-drained loamy soils with high organic content." ,
                        "Pomegranate" : "Pomegranates are primarily grown in India, Iran, and the Mediterranean. They prefer well-drained sandy or loamy soils with a neutral to slightly alkaline pH." , 
                        "Lentil" : "Lentils are commonly grown in Canada, India, and Turkey. They thrive in well-drained loamy or sandy loam soils with moderate fertility." , 
                        "Blackgram" : "Blackgram is densely cultivated in India and Myanmar. It grows well in loamy soils that are well-drained and rich in organic matter." , 
                        "Mungbean" : "Mungbeans are widely grown in India, China, and Southeast Asia. They prefer well-drained sandy loam or loamy soils with good fertility." , 
                        "Mothbeans" : "Mothbeans are commonly grown in arid regions like Rajasthan, India. They thrive in sandy loam soils that are well-drained and drought-resistant." ,
                        "Pigeonpeas" : "Pigeonpeas are extensively grown in India, Eastern Africa, and the Caribbean. They grow best in well-drained loamy soils with moderate fertility." , 
                        "Kidneybeans" : "Kidneybeans are cultivated in the United States, Brazil, and India. They thrive in well-drained loamy soils rich in organic matter." , 
                        "Chickpea" : "Chickpeas are primarily grown in India, Australia, and Turkey. They prefer well-drained loamy or sandy loam soils with moderate fertility." , 
                        "Coffee" : "Coffee is grown in tropical regions like Brazil, Vietnam, and Colombia. It thrives in well-drained loamy soils rich in organic content and slightly acidic pH."
    }

    top_3_crops = [list(crop_dict.values())[list(crop_dict.keys()).index(i + 1)] for i in top_3_indices]

    crop1 = top_3_crops[0]
    crop2 = top_3_crops[1]
    crop3 = top_3_crops[2]

    result1 = f"{crop1}"
    result2 = f"{crop2}"
    result3 = f"{crop3}"

    description1 = f"{crop_description[crop1]}.."
    description2 = f"{crop_description[crop2]}.."
    description3 = f"{crop_description[crop3]}.."

    def get_key_by_value(dictionary, value):
        result = {k: v for k, v in dictionary.items() if v == value}
        if result:
            return next(iter(result))
        return None

    key1 = get_key_by_value(crop_dict, crop1)
    key2 = get_key_by_value(crop_dict, crop2)
    key3 = get_key_by_value(crop_dict, crop3)

    return render_template(
        'index.html',
        result1=result1, result2=result2, result3=result3,
        description1=description1, description2=description2, description3=description3,
        result_image1=result_image[key1], result_image2=result_image[key2], result_image3=result_image[key3]
    )

# Python main
if __name__ == "__main__":
    app.run(debug=True)
