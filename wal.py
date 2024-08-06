import google.generativeai as genai
import PIL.Image
import os
import cv2
from dotenv import load_dotenv
import speech_recognition as sr
import pandas as pd
import pickle

df=pd.read_csv('titles.csv')
similarity=pickle.load('similarity.pkl')

recognizer = sr.Recognizer()

# Load environment variables from .env file
load_dotenv()

def recommend(title):
    index = df[df['title']==title].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x: x[1])
    for i in distances[1:6]:
        print(df.iloc[i[0]].title)

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
            if 'recommend' in text.lower():
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
def main():
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Load items from the dataset
    items = load_dataset('walmart_sample_01.txt')
    items_prompt = "Please select from these items: " + ", ".join(items)
    
    while True:
        if listen_for_keyword():
            capture_image()
            img = PIL.Image.open('captured_image.jpg')
            
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content([f"What is in this photo? your response should only contain item name nothing else and always try to give a response {items_prompt}", img])
            recommend(response.text)

if __name__ == "__main__":
    main()
