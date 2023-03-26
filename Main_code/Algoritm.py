import os
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.externals
import joblib


from prettytable import PrettyTable


path = os.path.join('Project_Credit_Card\creditcard.csv')
df = pd.read_csv(path)
# Оставляем в датафрейме наиболее важные фичи
# df = df[['V17','V12','V14','V10','V11','V16','V18','V9','Class']]
df = df.drop(['Amount','Time'],axis=1)
print(df)

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,:-1], df.iloc[:,-1], test_size=0.22,
    random_state=42, stratify=df.iloc[:,-1])

df_train = pd.concat([X_train, y_train], axis=1)