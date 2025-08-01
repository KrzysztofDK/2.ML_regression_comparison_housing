# 📊 Comparison of Machine Learning Regression Models on a Housing Dataset

## 🧠 About the Project
Project to predict the housing price and compare machine learning models, based on certain factors like house area, bedrooms, furnished, nearness to mainroad, etc.

## 📁 Dataset
Dataset taken from Kaggle -> https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

Contains: price, area, bedrooms, bathrooms, stories etc.

Harrison, D. and Rubinfeld, D.L. (1978) Hedonic prices and the demand for clean air. J. Environ. Economics and Management 5, 81–102.
Belsley D.A., Kuh, E. and Welsch, R.E. (1980) Regression Diagnostics. Identifying Influential Data and Sources of Collinearity. New York: Wiley.

## ⚙️ Technologies Used
- Python 3.10
- Jupyter Notebook
- VS Code
- Pandas
- Matplotlib
- Seaborn
- Chardet
- scikit-learn
- XGBoost
- joblib
- dill
- openpyxl

## 🔧 Installation:
pip install -r requirements.txt
## 🔧 Run by:
python main.py

All visualizations will be saved in 'images' folder.
All ML models and metrics will be saved in 'artifacts' folder.

## 🧪 Steps Performed
1. **Data Cleaning**
   - Removed duplicates,
   - Filled Nans,
   - Fixed columns data types (date),
   - Changed columns names,
   - Checked for zero intigers/floats,
   - Removed unnecessary columns.
2. **Feature Engineering**
   - Added a column with the logarithm of 10 price for use in ML models,
   - Applyed OneHotEncoder.
3. **Exploratory Data Analysis**
   - Basic understanding of dataset,
   - Charts were created like histograms, scatter plot, boxplot, pairplot, correlation heatmap,
4. **Models building, feature selection/extraction, hyperparameter tuning, training and metrics**
   - Selected Models:
      + Linear Regression,
      + Random Forest Regressor,
      + XGBoost,
      + SVR.
5. **Simple imput prediction of house price for selected model from file**

### Step-by-step analysis with notes and summaries is available in 'notebook' folder.

🧑‍💼 Author: Krzysztof Kopytowski
📎 LinkedIn: https://www.linkedin.com/in/krzysztof-kopytowski-74964516a/
📎 GitHub: https://github.com/KrzysztofDK
