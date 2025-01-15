# AI-calory-Tracker
create the Calory Tracker AI app, a cutting-edge application that utilizes artificial intelligence to make calorie tracking effortless and user-friendly.
------------------
Creating a Calorie Tracker AI app involves various components, such as food recognition, calorie calculation, user interaction, and personalized recommendations. The app can be designed to be simple yet powerful, offering users real-time calorie tracking and suggestions based on their food intake. We'll use Python along with libraries like TensorFlow for AI models (if food image recognition is involved), Edamam API for food database and calorie data, and Streamlit for building an interactive web app interface.
Key Features of the Calorie Tracker AI App:

    Food Image Recognition (Optional, using TensorFlow/Keras models).
    Manual Entry for food input, allowing users to track calories directly by entering food names.
    Calorie Calculation via APIs (such as Edamam or FatSecret).
    Real-Time Suggestions to help the user stay within their calorie goals.
    Data Logging and Tracking to help users monitor their daily caloric intake.

Libraries Needed:

    requests – for making API requests.
    streamlit – for building the web interface.
    tensorflow (optional) – for image recognition (if you want the app to support food image input).
    pytesseract (optional) – for text extraction from food images using OCR.

Step 1: Install Dependencies

pip install requests tensorflow streamlit pytesseract Pillow

For food data and calorie information, you will need to create an account on Edamam or FatSecret for access to their API.
Step 2: Get an API Key for Edamam

    Go to Edamam's website and sign up for an API key.
    You will get an App ID and App Key, which will be used to query food nutritional data.

Step 3: Calorie Tracker App Code

import requests
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO
import pytesseract

# Function to get calorie info from Edamam API
def get_calories_from_edamam(food_name):
    app_id = "your_app_id"  # Replace with your Edamam App ID
    app_key = "your_app_key"  # Replace with your Edamam App Key
    url = f"https://api.edamam.com/api/food-database/v2/parser"
    params = {
        'app_id': app_id,
        'app_key': app_key,
        'ingr': food_name
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "hints" in data:
        food = data['hints'][0]['food']
        label = food['label']
        calories = food['nutrients']['ENERC_KCAL']
        return f"{label}: {calories} kcal"
    else:
        return "Food not found in the database."

# Optional: Function to predict food item from an image (using a pretrained TensorFlow model)
def predict_food_from_image(image_path):
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    predicted_class = decoded_predictions[0][0][1]
    return predicted_class

# Function to extract text from an image (OCR) to find food names (optional)
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

# Streamlit UI for Calorie Tracker
def main():
    st.title("Calorie Tracker AI")
    
    # Option for food input method (manual or image)
    choice = st.radio("How would you like to enter the food?", ("Manual Entry", "Scan Food Image"))
    
    if choice == "Manual Entry":
        food_name = st.text_input("Enter the food name:")
        if food_name:
            result = get_calories_from_edamam(food_name)
            st.write(result)

    elif choice == "Scan Food Image":
        uploaded_file = st.file_uploader("Upload food image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Food Image", use_column_width=True)

            # Predict the food item from the image using TensorFlow model (Optional)
            food_name = predict_food_from_image(uploaded_file)
            st.write(f"Predicted food item: {food_name}")

            # Get calorie info based on the predicted food name
            result = get_calories_from_edamam(food_name)
            st.write(result)

    # Option for logging daily intake and calorie goals
    st.sidebar.title("Daily Calorie Tracking")
    daily_goal = st.sidebar.number_input("Enter your daily calorie goal", min_value=100, max_value=5000, value=2000)
    st.sidebar.write(f"Your daily calorie goal is: {daily_goal} kcal")

    # Tracking logged calories
    if "calories_logged" not in st.session_state:
        st.session_state.calories_logged = 0
    
    if choice == "Manual Entry" and food_name:
        calories = int(result.split(":")[1].split(" ")[0]) if "kcal" in result else 0
        st.session_state.calories_logged += calories
        st.sidebar.write(f"Total calories logged today: {st.session_state.calories_logged} kcal")
        st.sidebar.write(f"Remaining calories: {daily_goal - st.session_state.calories_logged} kcal")

if __name__ == '__main__':
    main()

Step 4: Explanation

    Edamam API Integration:
        We use the Edamam API to fetch food calorie data by sending the food name as a query.
        Replace your_app_id and your_app_key with your actual API credentials.
        The function get_calories_from_edamam(food_name) fetches the calorie data from the API and displays it.

    Image Recognition (Optional):
        If the user prefers scanning a food image, we use TensorFlow’s pre-trained MobileNetV2 model to classify the food item.
        This method uses the predict_food_from_image() function, which processes the uploaded image, predicts the food class, and fetches its calorie information.
        Pillow is used for image loading and resizing.

    Text Recognition via OCR (Optional):
        You can also extract text from images using pytesseract (OCR), allowing the user to upload a food package image, and the app can extract the food name from the text and search for the corresponding calorie data.

    Streamlit UI:
        Streamlit is used to create a simple, user-friendly web interface.
        The app allows users to either manually enter food names or upload an image for recognition.
        It also includes a sidebar to set daily calorie goals and track logged calories.

Step 5: Running the App

To run the app, save the code in a Python file, e.g., calorie_tracker.py, and run the following command:

streamlit run calorie_tracker.py

This will launch the app in your browser, where you can test its functionality.
Step 6: Additional Features to Add

    Food Diary:
        You can extend the app to store and track food history, logging each meal with calorie information.

    Personalized Recommendations:
        Integrate AI to provide personalized meal suggestions based on the user's calorie intake, dietary preferences, or goals (e.g., weight loss, muscle gain).

    Voice Input:
        Add voice recognition to allow users to speak the food item, and the app will fetch calorie data based on voice input.

    Integrate Fitness APIs:
        Connect with fitness trackers or apps (e.g., Fitbit, MyFitnessPal) to sync activity data and adjust calorie goals.

Conclusion:

This Calorie Tracker AI app provides a simple yet effective solution for users to track their food intake, either manually or by scanning images. It uses advanced AI models for food recognition and integrates with the Edamam API for nutritional information. By adding more personalized features and enhancements, this app could become a comprehensive tool for managing nutrition and fitness goals.
