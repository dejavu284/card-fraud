import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pandas as pd
#import Algoritm as alg
import copy
import json
import sklearn.externals
import joblib
from prettytable import PrettyTable, from_csv
from tabulate import tabulate

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


def fast_read(name_path):
    """Считывание данных из базы данных"""
    ref = db.reference(name_path)
    return ref.get()


def print_ref(name_path):
    """Вывод на экран данных из базы данных"""
    ref = db.reference(name_path)
    print(pd.DataFrame(ref.get()))



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


def PrettyTable_print(data):
    #x = PrettyTable()
    #x.field_names = data.columns
    # for i in range(data.shape[0]):
    #     x.add_rows(data.iloc[i,:])
    # x.add_rows([i for i in data.iloc[0,:]])
    # x = from_csv(data.to_csv())
    with open(data.to_csv(), "r") as fp: 
        x = from_csv(fp)
    print(x)

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
         
    #connect_to_my_db()

    # Код для добавление данных из локальной базы, в облачную

    # data = pd.read_csv('Project_Credit_Card\creditcard.csv').head(20)
    # print(data.shape, '\n')
    # ref_get = load_data(data, 'Data_uncertain')
    # print(pd.DataFrame(ref_get).shape, '\n')



    ref = db.reference('Data_uncertain') # Зашёл к uncertain
    test = ref.get()
    df = pd.DataFrame(ref.get())
    print(type(test))
    #PrettyTable_print(df)
    print(df.to_markdown())
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    # print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
    # print(tabulate(df, headers='keys', tablefmt='github'))

    
    is_none(data_delete('Data_uncertain')) # Удалил из бд
    is_not_none(load_data(df.T, 'Data_uncertain')) # сразу востанавливаю данные, это временная мера

    is_none(data_delete('Data_certain'))
    #is_not_none(load_data(df.T, 'Data_certain')) # Пока добавляю в certain до алгоритма машинного обучения, но вообще нужно после

    id_card = df.id # Запоминание индексов транзакций
    print(df, '\n')
    print(df.columns, '\n')
    new_df = pd.DataFrame()
    for i in range(1,29):
        new_df[f'V{i}'] = df[f'v{i}']

    new_df['Class'] = df.classTran
    print(new_df, '\n')
    print(new_df.columns, '\n')

    X_test = new_df.iloc[:, :-1]
    y_test = new_df.iloc[:, -1]

    print(X_test, '\n')
    print(y_test, '\n')

    #
    # Загрузка сохраненной модели
    loaded_model = joblib.load('Project_Credit_Card\Code\Main_code\knn_model.sav')
    y_prob = loaded_model.predict_proba(X_test)
    y_prob = pd.DataFrame(y_prob, index=y_test.index).iloc[:,-1]
    # y_prob = id_card # Kод для алгоритма машинного обучения
    #

    result = pd.DataFrame()
    result['id'] = id_card
    result['classTran'] = round(y_prob,2)
    print(result, '\n')

    
    is_not_none(load_data(result.T, 'Data_certain'))

    ref = db.reference('Data_certain')
    print(pd.DataFrame(ref.get()))



if __name__ == "__main__":
    connect_to_Kirill_db()
    if db.reference('Data_uncertain').get() is not None:
        is_not_none(db.reference('Data_uncertain').get())
        main()
    else: is_not_none(db.reference('Data_uncertain').get())



# from firebase_admin import credentials, firestore

# # Подключение к базе данных firebase
# cred = credentials.Certificate("path/to/serviceAccountKey.json")
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# # Получение последних записей из коллекции 'data'
# doc = db.collection('data').order_by(u'timestamp', direction=firestore.Query.DESCENDING).limit(10)

# # Удалить только что считанные данные из базы данных:
# batch = db.batch()
# for obj in doc:
#     ref = db.collection('data').document(obj.id)
#     batch.delete(ref)
# batch.commit()

# # Преобразование json данных в pandas.DataFrame
# data = []
# for obj in doc:
#     data.append(obj.to_dict())
# df = pd.DataFrame(data)

# # Выбор и использование алгоритма машинного обучения
# from my_algorithm import my_algorithm
# result = my_algorithm(df)

# # Отправка результата в базу данных firebase
# data = {
#     'timestamp': firestore.SERVER_TIMESTAMP,
#     'result': result
# }
# db.collection('results').add(data)


