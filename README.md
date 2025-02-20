# Optimized Stroke Prediction with Machine Learning 🚀

## 📌 Overview
This project aims to predict the likelihood of stroke using **machine learning models** such as **XGBoost, Random Forest, and Logistic Regression**. The dataset consists of patient health records with various risk factors, and the model is optimized using **Optuna** for hyperparameter tuning. Additionally, **SHAP (Shapley Additive Explanations)** is utilized to enhance model interpretability by identifying the top contributing features.

## 🛠️ Technologies Used
- **Python** (Core programming language)
- **Scikit-learn** (Machine Learning Library)
- **XGBoost** (Boosted Decision Trees)
- **Random Forest** (Ensemble Learning)
- **Optuna** (Hyperparameter Tuning)
- **SHAP** (Explainable AI)
- **SMOTE** (Handling Class Imbalance)
- **Pandas & NumPy** (Data Manipulation)
- **Matplotlib & Seaborn** (Visualization)

## 📊 Dataset
- The dataset used in this project contains **5,110 patient records** with features like **age, hypertension, heart disease, smoking status, glucose level, BMI, and stroke occurrence**.
- Data Source: [Healthcare Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

## 🔥 Key Features
✅ **Data Preprocessing & Feature Engineering**: Missing value imputation, outlier detection, encoding, and scaling.

✅ **Class Imbalance Handling**: Augmented minority class representation using **SMOTE** to balance stroke cases.

✅ **Model Optimization & Evaluation**: Fine-tuned **Random Forest & XGBoost**, achieving **97.4% accuracy**.

✅ **Explainable AI**: Used **SHAP** to highlight the **top 5 stroke risk factors**.

✅ **Hyperparameter Tuning**: Optimized model performance using **Optuna**, improving prediction precision and recall.

## ⚙️ Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pjaiswalusf/Stroke-Prediction
   cd stroke-prediction-ml
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv  # Create virtual environment
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate  # For Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook or Python script:**
   ```bash
   jupyter notebook
   ```
   Open `stroke_prediction.ipynb` and execute the cells.

## 📌 Usage
- **Exploratory Data Analysis (EDA):** Run the provided notebook to generate insights and visualize key features.
- **Train the Model:** Train different models using the dataset and compare performance metrics.
- **Hyperparameter Tuning:** Fine-tune the models using Optuna for the best results.
- **Explainability:** Use SHAP to analyze feature importance and understand model predictions.

## 📈 Model Performance
| Model | Accuracy | Precision | Recall | F1-score |
|--------|------------|------------|---------|----------|
| **XGBoost** | **97.4%** | 95.8% | 96.1% | 96.0% |
| **Random Forest** | 96.7% | 94.2% | 95.0% | 94.6% |
| **Logistic Regression** | 85.6% | 81.0% | 83.2% | 82.1% |

## 📊 Feature Importance (Top 5 Features by SHAP Analysis)
1️⃣ **Age** 🡆 Higher age increases stroke risk.
2️⃣ **Hypertension** 🡆 People with hypertension have a significantly higher probability of stroke.
3️⃣ **Heart Disease** 🡆 Patients with pre-existing heart disease are more likely to have a stroke.
4️⃣ **Glucose Level** 🡆 Elevated blood sugar levels strongly correlate with stroke occurrences.
5️⃣ **BMI** 🡆 Obesity is a contributing factor to stroke risk.

## 📌 Future Improvements
🚀 **Deploy the model as an API** using **Flask/FastAPI** for real-time stroke prediction.
🚀 **Improve class balancing techniques** to handle data skewness more effectively.
🚀 **Try deep learning models** (e.g., LSTMs or Neural Networks) for better accuracy.

## 🤝 Contributing
Feel free to **fork** this repository, **open issues**, or submit **pull requests**. Contributions are always welcome!

## 📜 License
This project is **open-source** and available under the **MIT License**.

## 📬 Contact
For any questions or collaborations, reach out to me at **jaiswalpratik49@gmail.com** or connect via [LinkedIn](https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/pratik-jaiswal-468315197/)). 🚀
