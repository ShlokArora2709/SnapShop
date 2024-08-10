from flask import Flask,render_template,jsonify,request
import google.generativeai as genai
import PIL.Image
import os
import cv2
from dotenv import load_dotenv
import speech_recognition as sr
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

df = pd.read_csv('titles.csv')
similarity = np.load('similarity.npy')

recognizer = sr.Recognizer()

# Load environment variables from .env file
load_dotenv()

def clean_response(response):
    # Remove Markdown formatting (e.g., **bold**) and newline characters
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Remove markdown
    cleaned_text = cleaned_text.replace('\n', ' ')  # Remove newline characters
    return cleaned_text.strip()  # Remove leading/trailing whitespace

def recommend(title):
    index = df[df['title'] == title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    products=[]
    for i in distances[1:6]:
        products.append(df.iloc[i[0]].title)
    return products

# Load and parse the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        items = [line.strip() for line in file.readlines()]
    return items

# Listen for the keyword
def listen_for_keyword():
    with sr.Microphone() as source:
        print("Listening for keyword...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            if 'hello' in text.lower():
                return True
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
    return False

# Capture image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('captured_image.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()

# Main function
@app.route('/search',methods=['POST'])
def get_response():
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    # Load items from the dataset
    items = load_dataset('walmart_sample_01.txt')
    items_prompt = "Please select from these items: " + ", ".join(items)
    # if listen_for_keyword():
    # capture_image()
    # data=request.get_json()
    file=request.files.get('ip-img')
    img = PIL.Image.open(file)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    # Extract the response text properly
    try:
        response = model.generate_content([f"What is in this photo? Your response should only contain the item name, nothing else, and always try to give a response from: {items_prompt}", img])
        response_text = response.candidates[0].content.parts[0].text
        clean_text = clean_response(response_text)
        products=recommend(clean_text)
        return jsonify({'name': clean_text,'products': products})
    except Exception as err:
        print(err)
        return jsonify({'message': 'Could not find the product you are looking for'})

@app.route('/',methods=['GET','POST'])
def main():
    return render_template('landing.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)