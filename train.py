import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import precision_score,recall_score,roc_auc_score,accuracy_score,classification_report,confusion_matrix

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

df = pd.read_csv('breast1.csv', encoding='latin-1') 

df.rename(columns={'ï»¿id': 'id'}, inplace=True)

df['diameter_mean'] = 2 * df['radius_mean']
df['diameter_se'] = 2 * df['radius_se']
df['diameter_worst'] = 2 * df['radius_worst']

df['circumference_mean'] = 2 * np.pi * df['radius_mean']
df['circumference_se'] = 2 * np.pi * df['radius_se']
df['circumference_worst'] = 2 * np.pi * df['radius_worst']

df['circumference_mean'] = 2 * np.pi * df['radius_mean']
df['circumference_se'] = 2 * np.pi * df['radius_se']
df['circumference_worst'] = 2 * np.pi * df['radius_worst']

df.drop('id',axis=1,inplace=True)

X=df.drop('diagnosis',axis=1)
y=df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, shuffle=True)

scaler=StandardScaler()
scaler.fit(X_train)
train_scaled = scaler.transform(X_train)
val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(X_test)

def grid_search_run(param_grid,model):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(train_scaled, y_train)

    return grid_search

    param_grid = {
    'C':np.logspace(-20,0,20),
    'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],
    'penalty':['l2','l1']
}

log_model=grid_search_run(param_grid,LogisticRegression())
y_pred=log_model.predict(val_scaled)
y_pred_train=log_model.predict(train_scaled)
print('Logistic Regression')
print('Training Accuracy: ',accuracy_score(y_train,y_pred_train))
print('Testing Accuracy: ',accuracy_score(y_val,y_pred))
print('Precision: ',precision_score(y_val,y_pred))
print('Recall: ',recall_score(y_val,y_pred))
print('ROC_AUC: ',roc_auc_score(y_val,y_pred))
print(classification_report(y_val,y_pred))

results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_val, y_pred),
    'Precision': precision_score(y_val, y_pred),
    'Recall': recall_score(y_val, y_pred),
    'ROC-AUC': roc_auc_score(y_val, y_pred)
})