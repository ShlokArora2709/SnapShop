# SnapShop: Image-Based Product Recommendation System

## Overview

SnapShop is a web-based application that allows users to capture or upload images of products and receive product recommendations based on the visual content of the image. The system leverages Natural Language Processing (NLP) and computer vision techniques to identify the product in the image and recommend similar products from a pre-defined dataset.

## Features

- **Image Recognition:** Capture or upload an image of a product and have it identified using Google's Generative AI model.
- **Product Recommendations:** Receive a list of similar products based on the identified item using cosine similarity.
- **Voice Recognition:** The system listens for a keyword to trigger certain actions.
- **Web Interface:** A user-friendly web interface built using Flask to interact with the system.

## Technologies Used

- **Python:** Core language for the backend.
- **Flask:** Web framework for creating the web application.
- **Google Generative AI:** Used for analyzing images and generating text-based content.
- **Pandas:** For handling and processing datasets.
- **scikit-learn:** For NLP and cosine similarity calculations.
- **OpenCV:** For image capture via webcam.
- **SpeechRecognition:** For handling voice commands.
- **NumPy:** For numerical operations and similarity calculations.
- **dotenv:** For managing environment variables securely.
- **Jinja2:** For rendering HTML templates.

## Dataset

The system uses a pre-defined dataset of product stored in a CSV file (`dataset.csv`). The dataset contains the titles of products available in the application and is used for generating recommendations.

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)


## Usage

1. **Access the Application:**

   Open your browser and go to `http://0.0.0.0:5000`.

2. **Upload or Capture an Image:**

   - You can upload an image of a product from your device.
   - Alternatively, use the webcam to capture an image directly from the application.

3. **Receive Product Recommendations:**

   After analyzing the image, the system will identify the product and provide a list of similar products based on cosine similarity.

## Project Structure

```
.
├── app.py                     # Main application file
├── titles.csv                 # Product titles dataset
├── similarity.npy             # Precomputed similarity matrix
├── templates/
│   └── landing.html           # HTML template for the web interface
├── static/
│   ├── css/                   # CSS files for styling
│   └── assets/
│       └── images/            # Images used in the application
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables (Google API key)
```

## Key Functions

### Image Recognition and Product Identification

```python
def search():
    ...
    response = model.generate_content([f"What is in this photo? Your response should only contain the item name, nothing else, and always try to give a response from: {items_prompt}", img])
    ...
```

### Product Recommendation

```python
def recommend(title):
    index = df[df['title'] == title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    products=[]
    for i in distances[1:6]:
        products.append(df.iloc[i[0]].title)
    return products
```

