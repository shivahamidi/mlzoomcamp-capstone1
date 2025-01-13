# Breast Cancer Classification API

## Description:

Breast cancer is the most prevalent cancer among women globally, representing 25% of all cancer diagnoses. In 2015 alone, it impacted over 2.1 million individuals. The disease originates when breast cells grow uncontrollably, often forming tumors that can be detected through X-rays or felt as lumps in the breast.

The main challenge in detecting breast cancer lies in classifying tumors as either malignant (cancerous) or benign (non-cancerous). Your task is to use machine learning, specifically Support Vector Machines (SVMs), to classify these tumors using the Breast Cancer Wisconsin (Diagnostic) Dataset.

This project implements a machine learning model to classify breast cancer using the Breast Cancer Dataset. The project includes data preprocessing, model training, and deployment of the trained model as a REST API using Flask, Docker, and Kubernetes.

### Table of Contents

 #### 1. Project Overview
 #### 2. Dataset
 #### 3. Model
 #### 4. API Overview
 #### 5. Deployment
 #### 6. Usage
 #### 7. Technologies Used
 #### 8. Folder Structure

## Project Overview

The purpose of this project is to classify breast cancer samples as malignant or benign using machine learning techniques. The trained model is deployed as a REST API for easy access and scalability.

The project includes:

#### - Data cleaning and preprocessing.
#### - Training a high-accuracy Logistic Regression model.
#### - Saving the trained model using pickle.
#### - Creating an API endpoint /predict to classify new samples.
#### - Deployment using Docker and Kubernetes.

## Dataset

The dataset used for this project can be found on Kaggle: Breast Cancer Dataset.

Features:

 Includes measurements such as radius_mean, texture_mean, perimeter_mean, and more.
  ### Target variable:
  ##### 0: Benign
  ##### 1: Malignant

## Model

  Algorithm: Logistic Regression
  
  Accuracy: Achieved 99.4% accuracy on the test dataset.
  
  Preprocessing:
  
   #### Normalized features using StandardScaler.
   
   #### Handled missing and redundant features during preprocessing.

## API Overview

The Flask API exposes the following endpoint:

#### Endpoint: /predict

#### Method: POST
    
#### Input: JSON object containing the feature values.
    
#### Output: Predicted class (0 for benign, 1 for malignant) and confidence scores.

### Example Request:

    {
    "radius_mean": 12.96,
    "texture_mean": 18.29,
    "perimeter_mean": 84.18,
    "area_mean": 525.2,
    "smoothness_mean": 0.07351,
    "compactness_mean": 0.07899,
    "concavity_mean": 0.04057,
    "concave points_mean": 0.01883,
    "symmetry_mean": 0.1874,
    "fractal_dimension_mean": 0.05899
     }

### Example Response:

     {
    "prediction": 1,
    "probabilities": [0.05, 0.95]
     }

## Deployment
Local Deployment

 Clone the repository:

     git clone <repository-url>
     cd <repository-folder>

Install dependencies:

     pip install -r requirements.txt

Run the Flask app:

     python app.py

Access the API at:

    http://127.0.0.1:5000/predict

## Docker Deployment

  Build the Docker image:

     docker build -t my-ml-service .

Run the Docker container:

     docker run -p 5000:5000 my-ml-service

Test the API:

     curl -X POST -H "Content-Type: application/json" -d '{"radius_mean": 12.96, "texture_mean": 18.29}' http://127.0.0.1:5000/predict

## Kubernetes Deployment

 Push the Docker image to Docker Hub:

     docker push <your-dockerhub-username>/my-ml-service

Deploy using Kubernetes:

     kubectl apply -f deployment.yaml

  Access the API using the service's external IP or port.

## Usage

You can interact with the deployed API using:

 #### Postman
 #### cURL
 #### Any HTTP client

## Technologies Used

 #### Programming Language: Python
 #### Libraries: Flask, scikit-learn, pandas, numpy
 #### Deployment: Docker, Kubernetes
 #### Model Saving: pickle


 ## Folder Structure
      .
      ├── app.py                # Flask API definition
      ├── requirements.txt      # Python dependencies
      ├── Dockerfile            # Docker image definition
      ├── deployment.yaml       # Kubernetes deployment configuration
      ├── train.py              # Script to train the model
      ├── model/                # Directory containing the trained model
      ├── data/                 # Dataset files or instructions to download
      └── README.md             # Project documentation
