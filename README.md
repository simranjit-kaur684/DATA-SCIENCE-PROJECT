# DATA-SCIENCE-PROJECT
# Stress & Mood Prediction Using Machine Learning

This project applies machine learning techniques to predict **stress** and **mood** levels based on behavioural, contextual, and self‑reported data.  
It follows a full data‑science workflow including preprocessing, EDA, feature engineering, model training, evaluation, and interpretation.

---

##  Project Structure

```
project/
│
├── data/
│   ├── sample_100k.csv          # 100k-row sample dataset for GitHub
│   ├── data_dictionary.md       # Column descriptions
│
├── notebooks/
│   ├── eda.ipynb                # Exploratory Data Analysis
│   ├── model_training.ipynb     # Model building & evaluation
│
├── src/
│   ├── preprocessing.py         # Cleaning, encoding, feature engineering
│   ├── train_models.py          # ML training pipeline
│   ├── utils.py                 # Helper functions
│
├── results/
│   ├── model_performance_table.png
│   ├── feature_importance.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

##  Dataset

The original dataset is large, so only a **100k-row sample** is included in this repository for reproducibility.

- **File:** `data/sample_100k.csv`
- **Rows:** 100,000  
- **Columns:** Behavioural, contextual, and self‑reported variables  
- **Full dataset:** Available externally (https://www.kaggle.com/datasets/wafaaelhusseini/worklife-balance-synthetic-daily-wellness-dataset/data)


---

##  Methods & Workflow

This project follows the **CRISP‑DM** methodology:

1. **Business Understanding**  
   Predict stress and mood levels to understand behavioural patterns.

2. **Data Understanding**  
   EDA performed to explore distributions, correlations, missing values.

3. **Data Preparation**  
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling  
   - Outlier treatment  
   - Train-test split  

4. **Modelling**  
   Models trained:
   - Linear Regression  
   - Random Forest Regressor  
   - Gradient Boosting Regressor  

5. **Evaluation**  
   Metrics used:
   - R²  
   - MAE  
   - RMSE  

6. **Deployment (Optional)**  
   Code structured for easy extension into an API or app.

---

##  Model Performance Summary

### **Stress Prediction**

| Model              | R²     | MAE   | RMSE  | Notes |
|-------------------|--------|-------|-------|-------|
| Linear Regression  | 0.542  | 0.635 | 0.798 | Performs reasonably but limited with non-linear patterns |
| Random Forest      | 0.542  | 0.633 | 0.799 | Slight MAE improvement, no R² gain |
| Gradient Boosting  | 0.548  | 0.630 | 0.793 | **Best overall** |

### **Mood Prediction**

| Model              | R²     | MAE   | RMSE  | Notes |
|-------------------|--------|-------|-------|-------|
| Linear Regression  | 0.345  | 0.953 | 1.191 | **Best R²** |
| Random Forest      | 0.323  | 0.968 | 1.211 | Worst performance |
| Gradient Boosting  | 0.341  | 0.955 | 1.194 | Balanced, competitive |

---

##  Key Insights

- Stress is easier to model than mood due to clearer behavioural patterns.
- Gradient Boosting consistently performs best for stress.
- Mood has more noise and weaker linear relationships.
- Feature importance suggests contextual variables strongly influence stress.

---

##  How to Run the Project

### **1. Install dependencies**
```
pip install -r requirements.txt
```

### **2. Run preprocessing**
```
python src/preprocessing.py
```

### **3. Train models**
```
python src/train_models.py
```

### **4. Explore notebooks**
Open in Jupyter:
```
jupyter notebook
```

---

##  Future Work

- Hyperparameter tuning (GridSearchCV / Optuna)
- Deep learning models (LSTM for temporal patterns)
- Deployment via FastAPI or Streamlit
- Feature selection and SHAP interpretability


## 📜 License

This project is open-source for educational and research purposes.
