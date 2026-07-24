# Movie Rating Prediction with Python

## CodSoft Data Science Internship - Task 2

This project focuses on predicting movie ratings using Machine Learning regression models. The model estimates the rating of a movie based on features such as genre, director, actors, year, duration, and votes.

## Objective

The objective of this project is to build a regression model that predicts movie ratings accurately using historical movie data.

## Dataset

- **Dataset:** IMDb Movies India Dataset
- **File:** `IMDb Movies India.csv`

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Workflow

- Import Libraries
- Load Dataset
- Data Exploration
- Data Preprocessing
- Feature Encoding
- Train-Test Split
- Model Training
- Model Evaluation
- Sample Prediction

## Machine Learning Models

- Linear Regression
- Random Forest Regressor

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Project Structure

```text
Task_2_Movie_Rating_Prediction/
│── IMDb Movies India.csv
│── Movie_Rating_Prediction.ipynb
│── README.md
```

## Results

Both regression models were trained and evaluated on the dataset.

**Linear Regression**
- MAE: ~1.05
- RMSE: ~1.30
- R² Score: ~0.09

**Random Forest Regressor**
- MAE: ~0.85
- RMSE: ~1.13
- R² Score: ~0.31

The **Random Forest Regressor** achieved better performance than Linear Regression and was selected as the final model.

## Author

**Kumar Chiranjivee**

**CodSoft Data Science Internship**
