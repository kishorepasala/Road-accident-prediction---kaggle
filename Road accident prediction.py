#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_csv('/Users/kishore/Desktop/Data sets/Road accident/road_accident_train.csv')


# In[ ]:


data.head()


# In[ ]:


print(data.shape)
print(data.isnull().sum())


# In[ ]:


data.nunique()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


categorical_col = []

for col in data.columns:
    if data[col].dtype == 'object' and data[col].nunique() < 5:
        categorical_col.append(col)


# In[ ]:


categorical_col


# In[ ]:


data['time_of_day'].value_counts()['morning']


# In[ ]:


plt.figure(figsize=(15, 15))
for i, col in enumerate(categorical_col):
    plt.subplot(3, 2, i+1)
    sns.countplot(data=data, x=col)

plt.tight_layout()
plt.show()


# # Model (Logistic regression)

# In[ ]:


df_data = data.copy()
df_data = df_data.drop(columns = ['id'])


# In[ ]:


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


# In[ ]:


df_data.head()


# In[ ]:


df_data.describe()


# In[ ]:


cat_col = []

for col in df_data.columns:
    if df_data[col].dtype == 'object':
        cat_col.append(col)

print(cat_col)


# In[ ]:


df_data['road_signs_present'].dtype


# In[ ]:


boolean_cols = []

for col in df_data.columns:
    if df_data[col].dtype == 'bool':
        boolean_cols.append(col)
        
print(boolean_cols)


# In[ ]:


# using OneHot Encoding for road_type and weather
oe = OneHotEncoder(sparse_output=False, drop='first')
nominal_cols = ['road_type','weather']
encoded_nominal = oe.fit_transform(df_data[nominal_cols])
df_data[oe.get_feature_names_out(nominal_cols)] = encoded_nominal
df_data = df_data.drop(nominal_cols, axis=1)

# using Ordinal Encoding for lighting and time of day
ord_encoding = OrdinalEncoder(categories=[['daylight','dim','night'], ['morning','afternoon','evening','night']])
df_data[['lighting', 'time_of_day']] = ord_encoding.fit_transform(df_data[['lighting','time_of_day']])


# In[ ]:


for col in df_data[boolean_cols]:
    df_data[boolean_cols] = df_data[boolean_cols].astype(int)


# In[ ]:


# df_data['speed_limit'] = np.log(df_data['speed_limit']


# In[ ]:


df_data


# In[ ]:


df_data['accident_risk'] = (df_data['accident_risk'] >= 0.5).astype(int)

y = df_data['accident_risk']
x = df_data.drop(columns='accident_risk')


# In[ ]:


# Train test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[ ]:


# Model creating

model_1 = LogisticRegression()

model_1.fit(X_train, y_train)


# In[ ]:


preds = model_1.predict(X_test)


# In[ ]:


preds = preds.reshape(-1, 1)


# In[ ]:


accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)


# # Test data prediction

# In[ ]:


test_data = pd.read_csv('/Users/kishore/Desktop/Data sets/Road accident/road_accident_test.csv')


# In[ ]:


test_data


# In[ ]:


test_id = test_data['id']


# In[ ]:


test_data = test_data.drop(columns='id')


# In[ ]:


# using OneHot Encoding for road_type and weather
oe = OneHotEncoder(sparse_output=False, drop='first')
nominal_cols = ['road_type','weather']
encoded_nominal = oe.fit_transform(test_data[nominal_cols])
test_data[oe.get_feature_names_out(nominal_cols)] = encoded_nominal
test_data = test_data.drop(nominal_cols, axis=1)

# using Ordinal Encoding for lighting and time of day
ord_encoding = OrdinalEncoder(categories=[['daylight','dim','night'], ['morning','afternoon','evening','night']])
test_data[['lighting', 'time_of_day']] = ord_encoding.fit_transform(test_data[['lighting','time_of_day']])


# In[ ]:


test_boolean_cols = []

for col in test_data.columns:
    if test_data[col].dtype == 'bool':
        test_boolean_cols.append(col)
        
print(test_boolean_cols)


# In[ ]:


for col in test_data[test_boolean_cols]:
    test_data[test_boolean_cols] = test_data[test_boolean_cols].astype(int)


# In[ ]:


test_data


# In[ ]:


test_preds = model_1.predict(test_data)
print(test_preds)


# In[ ]:


preds_df = pd.DataFrame(test_preds, columns=['predictions'])


# In[ ]:


preds_df.head(50)


# In[ ]:


import pandas as pd

submission = pd.concat(
    [test_id, pd.Series(test_preds, name="predictions")],
    axis=1
)

submission.to_csv("submission.csv", index=False)


# In[ ]:


# Define save path
save_path = "/Users/Kishore/Desktop/submission.csv"

# Save to desktop
submission.to_csv(save_path, index=False)

print(f"✅ Submission file saved to: {save_path}")


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer

model = RandomForestClassifier(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model_1, x, y, cv=kf, scoring='roc_auc')

print("CV scores:", scores)
print("Mean ROC-AUC:", scores.mean())


# # LightGBM method

# In[ ]:


data.head()


# In[ ]:


l_data = data.copy()
l_data


# In[ ]:


categorical_columns = []
for col in l_data.columns:
    if l_data[col].dtype == 'object':
        categorical_columns.append(col)

print(categorical_columns)


# In[ ]:


X = l_data.drop(columns='accident_risk')
y = l_data['accident_risk']


# In[ ]:


X = X.drop(columns='id')


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


X_train.shape


# In[ ]:


import lightgbm as lgb


# In[ ]:


for col in categorical_columns:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')


# In[ ]:


y_train = y_train.astype(int)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np


# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        categorical_feature=categorical_columns
    )

    val_preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_preds, squared=False)  # RMSE
    rmse_scores.append(rmse)

print("CV RMSE scores:", rmse_scores)
print("Mean CV RMSE:", np.mean(rmse_scores))


# In[ ]:


test_data = pd.read_csv('/Users/kishore/Desktop/Data sets/Road accident/road_accident_test.csv')


# In[ ]:


test_id = test_data['id']


# In[ ]:


test_data = test_data.drop(columns='id')


# In[ ]:


test_data.shape


# In[ ]:


categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']

for col in categorical_features:
    # Make test categories match train
    test_data[col] = test_data[col].astype('category')
    test_data[col].cat.set_categories(X_train[col].cat.categories, inplace=True)


# In[ ]:


preds = model_2.predict_proba(test_data)[:,1]


# In[ ]:


preds


# In[ ]:


import pandas as pd

# Create a DataFrame with id and predictions
submission_2 = pd.DataFrame({
    'id': test_id,       # make sure this matches your test DataFrame
    'target': preds.astype('float')
})

# Optional: save to CSV
submission_2.to_csv("/Users/Kishore/Desktop/submission_2.csv", index=False)

print(submission_2.head())


# # LGBM with hyperparameter tuning

# In[2]:


import warnings
warnings.simplefilter('ignore')


# 1. Load Data
# 
# Here we load the required libraries (pandas, numpy) and datasets.
# 
# Competition Data: train.csv and test.csv.
# 
# External Data: The original dataset from which the competition data was generated. We combine its multiple files into a single orig DataFrame to augment our training data, a common strategy for improving model performance in Playground Series competitions.
# 
# Finally, we'll do a quick sanity check with .shape and .head() to ensure everything is loaded correctly.

# In[47]:


import pandas as pd, numpy as np


# In[48]:


train = pd.read_csv('/Users/kishore/Desktop/Kaggle Data sets/Road accident/road_accident_train.csv')
test = pd.read_csv('/Users/kishore/Desktop/Kaggle Data sets/Road accident/road_accident_test.csv')

orig = pd.read_csv('/Users/kishore/Desktop/Kaggle Data sets/Kaggle_100k_roadaccident.csv')
orig_2 = pd.read_csv('/Users/kishore/Desktop/Kaggle Data sets/Kaggle_10k_roadaccident.csv')
orig_3 = pd.read_csv('/Users/kishore/Desktop/Kaggle Data sets/Kaggle_2k_roadaccident.csv')

orig = pd.concat([orig, orig_2, orig_3])


# In[49]:


print('Train shape:', train.shape)
print('Test shape:', test.shape)
print('orig:', orig.shape)

train.head(5)


# In[50]:


test_id = test['id'] 


# In[20]:


TARGET = 'accident_risk'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATEGORICAL = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']
print(f"{len(BASE)} Base Features: {BASE}")


# # 2. Baseline Model
# Before diving into complex feature engineering or parameter tuning, it's crucial to establish a baseline score. This score, calculated using only the initial features and a standard set of model parameters, will serve as a benchmark to measure all future improvements.
# 
# We will use a 5-fold cross-validation (KFold) strategy to train our XGBRegressor model. This provides a robust estimate of the model's performance.
# 
# Key points of the implementation:
# 
# We use Out-of-Fold (OOF) predictions to calculate a single, reliable CV score across the entire training set.
# We employ early stopping to automatically find the optimal number of boosting rounds and prevent overfitting, which is a very effective practice.
# We leverage XGBoost's native support for categorical features by setting enable_categorical=True.

# In[21]:


FEATURES = BASE


# In[25]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


# In[27]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"---Fold {fold+1}/5---")
    
    X_train, X_val = train.iloc[train_idx][FEATURES], train.iloc[val_idx][FEATURES]
    y_train, y_val = train.iloc[train_idx][TARGET], train.iloc[val_idx][TARGET]
    
    X_train[CATEGORICAL] = X_train[CATEGORICAL].astype('category')
    X_val[CATEGORICAL] = X_val[CATEGORICAL].astype('category')
    
    model_3 = XGBRegressor(
        n_estimators=100000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        early_stopping_rounds=200,
    )
    
    model_3.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               verbose=1000)
    
    oof_preds[val_idx] += model_3.predict(X_val)
    
    print(f"Fold {fold+1} RMSE: {mean_squared_error(y_val, oof_preds[val_idx], squared=False)}")

print(f"Overall OOF RMSE: {mean_squared_error(train[TARGET], oof_preds, squared=False): 5f}")


# In[35]:


test[CATEGORICAL] = test[CATEGORICAL].astype('category')

test = test.drop(columns='id')


# In[42]:


preds = model_3.predict(test)


# In[39]:


preds


# In[52]:


import pandas as pd

# Create a DataFrame with id and predictions
submission_3 = pd.DataFrame({
    'id': test_id ,      # make sure this matches your test DataFrame
    'target': preds.astype('float')
})

# Optional: save to CSV
submission_3.to_csv("/Users/Kishore/Desktop/submission_3.csv", index=False)

print(submission_3.head())

