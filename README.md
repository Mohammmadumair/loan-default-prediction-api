**Loan Default Prediction API using FastAPI

Project Overview
**
This project predicts whether a customer is likely to default on a loan using machine learning and serves predictions through a FastAPI API.

The project demonstrates a production-style machine learning workflow including:

* Data preprocessing
* Handling imbalanced data
* Feature scaling
* Model training
* Pipeline creation
* API deployment using FastAPI
* Probability prediction


## Technologies Used

* Python
* pandas
* scikit-learn
* FastAPI
* Uvicorn
* joblib
* Git
* GitHub
* VS Code
* Anaconda


## Machine Learning Workflow

1. Load and clean data
2. Remove ID columns
3. Encode categorical variables
4. Handle imbalanced data
5. Scale features
6. Train Logistic Regression model
7. Create sklearn pipeline
8. Save trained pipeline
9. Build FastAPI prediction service

---

## API Features

* JSON input validation
* Prediction endpoint
* Probability output
* FastAPI Swagger documentation

---

## API Endpoint

POST /predict

Returns:

* Default prediction
* Default probability

---

## Example Response

{
"Default_Prediction": 1,
"Default_Probability": 0.88
}

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Start server:

uvicorn app:app --reload

Open API docs:

http://127.0.0.1:8000/docs

---

## What I Learned

* Building machine learning pipelines
* Handling imbalanced datasets
* Feature scaling
* FastAPI deployment
* API debugging
* Model serving
* Production ML workflow
