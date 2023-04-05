import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pandas as pd
import json
import joblib
import time
from sklearn.utils import shuffle

cred = 0
def connect_to_my_db():
    cred = credentials.Certificate(
        r"Project_Credit_Card\Code\card-fraud-firebase-adminsdk-64z0r-bf979796e0.json"
        )
    firebase_admin.initialize_app(
        cred ,{'databaseURL': 'https://card-fraud-default-rtdb.firebaseio.com/'})#, name='My_firebase')


def connect_to_Kirill_db():
    cred = credentials.Certificate(
        r"Project_Credit_Card\Code\app-tracking-fraud-transact-firebase-adminsdk-x9r28-0fc99ebd0a.json"
        )
    firebase_admin.initialize_app(
        cred ,{'databaseURL':'https://app-tracking-fraud-transact-default-rtdb.europe-west1.firebasedatabase.app'})#, name='My_firebase')


def load_data(data, name_path):
    ref = db.reference(name_path)
    ref.set(json.loads(data.to_json()))
    return ref.get()

def load_ref_get(data, name_path):
    ref = db.reference(name_path)
    ref.set(data)
    return ref.get()

def is_none(data):
    if data is None:
        print('Succsess')
    else: print('Is not succsess')


def is_not_none(data):
    if data is None:
        print('Is not succsess')
    else: print('Succsess')


def data_delete(name_path):
    ref = db.reference(name_path)
    ref.delete()
    return ref.get()


def main():
    
    # Реализовать:
    #   Считывание с базы данных
    #   Удаление данных из бд, которые были считаны 'Data_uncertain'
    #   Преобразование данных в DF
    #   Запоминание индексов транзакций
    #   Удаление индексов из DF
    #   Передача датафрейма в алгоритм машинного обучения
    #   Принятие измененного датасета, с определенном классом
    #   Отправка данных на бд сервер "Data_certain"

    ref = db.reference('Data_uncertain') # Зашёл к uncertain
    test = ref.get()
    df = pd.DataFrame(ref.get())

    
    data_delete('Data_uncertain') # Удалил из бд
    #is_not_none(load_data(df.T, 'Data_uncertain')) # сразу востанавливаю данные, это временная мера

    data_delete('Data_certain')

    id_card = df.id # Запоминание индексов транзакций
    new_df = pd.DataFrame()
    for i in range(1,29):
        new_df[f'V{i}'] = df[f'v{i}']
    
    new_df['Amount'] = df.amount
    new_df['Class'] = df.classTran
    new_df['Time'] = df.time

    flag = '0.98'    
    if flag == 'my':
        X_test = new_df.iloc[:, :-3]
        y_test = new_df.iloc[:, -3]
        # Загрузка сохраненной модели
        loaded_model = joblib.load('Project_Credit_Card\Code\Main_code\RF_model.sav')
    elif flag == '0.97':
        data_model = new_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class']]
        X_test = data_model.iloc[:, :-1]
        y_test = data_model.iloc[:, -1]
        name = 'RF'
        Path = f'Project_Credit_Card\Code\Main_code\{name}_model.sav'

        # Загрузка сохраненной модели
        loaded_model = joblib.load(Path)
    elif flag == '0.94':
        data_model = new_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']]
       
        X_test = data_model.iloc[:, :-1]
        y_test = data_model.iloc[:, -1]
        name = '3RF'
        Path = fr'Project_Credit_Card\Code\Main_code\{name}_model.sav'

        # Загрузка сохраненной модели
        loaded_model = joblib.load(Path)
    elif flag == '0.98':
        data_model = new_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']]
       
        X_test = data_model.iloc[:, :-1]
        y_test = data_model.iloc[:, -1]
        name = 'LR'
        Path = fr'Project_Credit_Card\Code\Main_code\{name}_model.sav'

        # Загрузка сохраненной модели
        loaded_model = joblib.load(Path)
    elif flag == '0.92':
        data_model = new_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class']]
        X_test = data_model.iloc[:, :-1]
        y_test = data_model.iloc[:, -1]
        name = '2RF'
        Path = f'Project_Credit_Card\Code\Main_code\{name}_model.sav'

        # Загрузка сохраненной модели
        loaded_model = joblib.load(Path)
    else:
        data_model = new_df[['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14',
                            'V16', 'V17', 'V18', 'V21', 'V23', 'V24', 'V26', 'V27', 'V28',
                            'Amount','Class']]
        X_test = data_model.iloc[:, :-1]
        y_test = data_model.iloc[:, -1]
        name = 'MLP'
        Path = f'Project_Credit_Card\Code\Main_code\{name}_model.sav'


        # Загрузка сохраненной модели
        loaded_model = joblib.load(Path)

    y_prob = loaded_model.predict_proba(X_test)
    y_prob = pd.DataFrame(y_prob, index=y_test.index).iloc[:,-1]
    #

    result = pd.DataFrame()
    result['id'] = id_card
    result['classTran'] = round(y_prob,2)
    
    test = pd.DataFrame()
    test['prob'] = round(y_prob,2)
    test['class'] = y_test
    print(test)

    
    load_data(result.T, 'Data_certain')



if __name__ == "__main__":
    connect_to_Kirill_db()
    is_not_none(db.reference('Data_uncertain').get())
    while True:
        if db.reference('Data_uncertain').get() is not None:
            main()
        else: db.reference('Data_uncertain').get()

