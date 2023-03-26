# %%
import os
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# %%
path = os.path.join('./creditcard.csv')
df = pd.read_csv(path)
# Оставляем в датафрейме наиболее важные фичи
# df = df[['V17','V12','V14','V10','V11','V16','V18','V9','Class']]назы
df = df.drop(['Amount','Time'],axis=1)

# %%
df

# %% [markdown]
# * Разделим датасет на тестовую и тренировочную выборку

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,:-1], df.iloc[:,-1], test_size=0.22,
    random_state=42, stratify=df.iloc[:,-1])

# %%
df_train = pd.concat([X_train, y_train], axis=1)


# %% [markdown]
# * Появилась идея, что можно на время убрать из датасета строки с классом '1'.
# * После этого можно будет провести обычный отбор фичей, и очистку датасета от выбросов
# * Если первый пункт не выполнить, то тогда в датасете останется очень мало объектов с классом '1', или не останется вовсе. Так дизбаланс классов станет еще более явным

# %%
# создадим отдельный датафрейм, в который запишем все объекты с классом 1
df_class_fraud = df_train[(df_train.Class == 1)]
# создадим отдельный датафрейм, в который запишем все объекты с классом 0
df_class_norm = df_train[df_train.Class == 0]
# создадим датафрейм в котором не будет параметра класс
X = df.drop(['Class'],axis=1)
X.info()

# %% [markdown]
# * Далее будем производить очистку 
# * Использую 2 способа борьбы с аномалиями, для этого создам 2 версии датасета: для одного и другого способа.

# %%
def quantile(df_cut, X):
    for i in X.columns:
        a = df_cut[i].quantile(0.25)
        b = df_cut[i].quantile(0.75)
        df_cut = df_cut[(df_cut[i] > a - 1.5 * (b - a)) & (df_cut[i] < b + 1.5 * (b - a))]
    return df_cut
        

def mean_std(df_cut, X):
    for i in X.columns:
        left_side = df_cut[i].mean() - 5 * df_cut[i].std()
        right_side = df_cut[i].mean() + 5 * df_cut[i].std()

        df_cut = df_cut[(df_cut[i] > left_side) & (df_cut[i] < (right_side))]
    return df_cut
# df_cut.info(), df_cut['Class'].value_counts()

# %%
# создадим 2 версии датафрейма, каждую из них почистим от выбросов конкретным методом
df_class_norm_quantile = df_class_norm
df_class_norm_mean_std = df_class_norm

# %%
l = 3 
while l !=0:
    first_value = df_class_norm_quantile.shape[0]
    df_class_norm_quantile = quantile(df_class_norm_quantile, X)
    second_value = df_class_norm_quantile.shape[0]
    l = first_value - second_value
df_class_norm_quantile.info()

# %%
df_class_norm_quantile.boxplot(column=[x for x in df_class_norm_quantile.columns])


# %%
df_train.Class.value_counts()

# %%
sns.scatterplot(data=df_train, x='V14', y='V17', hue='Class')

# %%
l = 3 
while l !=0:
    first_value = df_class_norm_mean_std.shape[0]
    df_class_norm_mean_std = mean_std(df_class_norm_mean_std, X)
    second_value = df_class_norm_mean_std.shape[0]
    l = first_value - second_value
df_class_norm_mean_std.info()

# %%

df_class_norm_mean_std.boxplot(column=['V17','V12','V14','V10','V11','V16','V18','V9'])


# %% [markdown]
# * Соединим наши датасеты в тренировочный датасет

# %%
# # Сначала попробуем датасет почищеный методом квартилей
# df_train = pd.concat([df_class_fraud, df_class_norm_quantile], axis=0)
# from sklearn.utils import shuffle
# df_train = shuffle(df_train)
# X_train = df_train.iloc[:,:-1]
# y_train = df_train.iloc[:,-1]
# X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%
# Теперь методом среднего и отклонения
df_train = pd.concat([df_class_fraud, df_class_norm_mean_std], axis=0)
from sklearn.utils import shuffle
df_train = shuffle(df_train)
X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1]
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%
# X_train = X_train.loc[:,['V14','V17']]
# X_test = X_test.loc[:,['V14','V17']]
# y_train.value_counts()

# %%
# clf = DecisionTreeClassifier(max_depth=3)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# %%
# clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
#     max_depth=1, random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# %%
# clf = RandomForestClassifier(max_depth=3)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# %%
import sklearn.externals
import joblib

# %%
# Обучение модели
model = KNeighborsClassifier(n_neighbors=17)
model.fit(X_train, y_train)

# Сохранение модели
filename = 'knn_model.sav'
joblib.dump(model, filename)



# %%
# Загрузка сохраненной модели
loaded_model = joblib.load(filename)

# %%
y_pred.value_counts()

# %%
# Использование загруженной модели для предсказания
y_pred = loaded_model.predict_proba(X_test)


# %%
y_test.index

# %%
y_pred = pd.DataFrame(pd.DataFrame(y_pred, index=y_test.index).iloc[:,-1])
y_pred

# %%
pd.concat([y_test,y_pred],axis=1)

# %%
# pd.concat([y_test, y_pred)

# %%
new_df = pd.DataFrame()
new_df['Class'] = pd.DataFrame(y_test)
# pd.DataFrame(y_test)
y_pred.shape, y_test.shape

# %%
new_df['Prob'] = y_pred
y_pred.value_counts(), y_test.value_counts()

# %%
new_df


# %%
clf = KNeighborsClassifier(n_neighbors=17)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# %%
confusion_matrix(y_test, y_pred)

# %%
print(classification_report(y_test, y_pred))

# %%
import graphviz
dot_data = clf.export_graphviz(clf, out_file=None,
                                feature_names=X.columns,
                                class_names=['0','1'], 
                                filled=True, rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

# %%



