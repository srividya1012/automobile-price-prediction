Project Overview:
This project implements a Machine Learning–based Automobile Price Prediction System that estimates the selling price of a car based on its specifications such as brand, year, mileage, engine capacity, fuel type, transmission, and ownership details.
The trained model is deployed as a Streamlit web application, allowing users to enter car details and get real-time price predictions.


Objectives:
To predict automobile prices using historical data
To apply regression techniques for accurate price estimation
To deploy the trained model using a user-friendly web interface
To demonstrate an end-to-end ML workflow (EDA → Training → Deployment)


Machine Learning Approach:
Problem Type: Supervised Learning (Regression)
Model Used: Polynomial Regression with Ridge Regularization

Reason for Choice:
Polynomial features capture non-linear relationships
Ridge regression reduces overfitting


🛠️ Technologies Used:
Programming Language: Python
Libraries & Frameworks:
Pandas
NumPy
Scikit-learn
Streamlit
IDE: VS Code
Deployment: Streamlit Web App


Project Structure:
major project/
│── app.py
│── test_model.py
│── requirements.txt
│── README.md
│── Cardetails.csv
│── models/
│   ├── ridge_model.pkl
│   ├── poly_transformer.pkl
│── venv/


How to Run:
pip install -r requirements.txt
streamlit run app.py
Open in browser:
http://localhost:8501



Output:
Estimated Car Price: ₹1,10,869


Conclusion:
The project demonstrates an end-to-end machine learning workflow from data preprocessing and model training to deployment using a web interface.


👩‍💻 Author
Srividya Madini




