# 🚦 Road Accident Severity Prediction using XGBoost

This project analyzes road accident data and builds a machine learning model to predict accident severity.  
It involves exploratory data analysis (EDA), feature engineering, and model training using **XGBoost** to identify key factors influencing accident outcomes.


## 🧠 Objective

The goal of this project is to:
- Analyze patterns in road accident data  
- Identify factors that contribute to accident severity  
- Build a predictive model to classify severity levels using XGBoost

---

## ⚙️ Technologies Used

- **Python 3.10+**
- **Pandas** – data handling and preprocessing  
- **NumPy** – numerical operations  
- **Matplotlib / Seaborn** – data visualization  
- **Scikit-learn** – preprocessing, metrics, and model evaluation  
- **XGBoost** – main machine learning model  

---

## 📊 Steps in the Workflow

1. **Data Loading & Cleaning**  
   - Load dataset using pandas  
   - Handle missing values, duplicates, and incorrect datatypes  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize distribution of severity, road type, weather, and vehicle factors  
   - Identify correlations and patterns  

3. **Feature Engineering**  
   - Encode categorical variables  
   - Scale or normalize numerical features if necessary  
   - Remove highly correlated or irrelevant features  

4. **Model Training (XGBoost)**  
   - Train XGBoost classifier on processed data  
   - Tune hyperparameters for optimal performance  

5. **Evaluation**  
   - Evaluate with accuracy, precision, recall, F1-score, and confusion matrix  
   - Visualize feature importance  

6. **Results & Insights**  
   - Highlight top factors affecting accident severity  
   - Discuss potential implications for safety improvements  

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/road-accident-xgboost.git
   cd road-accident-xgboost
