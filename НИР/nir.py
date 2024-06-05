import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score


from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
def plot_multiclass_roc(y_score,y_test, X_test, n_classes, figsize=(6, 6)):
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'ROC  for label {i}' )
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    return fig

@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/new.csv', nrows=500)
    test_data = pd.read_csv('data/new.csv')

    return data, test_data[1000:]

st.sidebar.header('Метод ближайших соседей')
data, test_data = load_data()
cv_slider = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)
step_slider = st.sidebar.slider('Шаг для соседей:', min_value=1, max_value=50, value=10, step=1)




#Количество записей
data_len = data.shape[0]
#Вычислим количество возможных ближайших соседей
rows_in_one_fold = int(data_len / cv_slider)
allowed_knn = int(rows_in_one_fold * (cv_slider-1))
st.header("Общая информация")
if st.checkbox('Показать матрицу корреляции'):
    # st.subheader('Круговая диаграмма тональностей')
    # fig2 = plt.figure(figsize=(8, 8))
    # data['key'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#9370DB', '#DB7093', '#DA70D6', '#FFC0CB', '#D8BFD8'])
    # plt.axis('equal')  # чтобы диаграмма круглая была
    # plt.title('Тональность песен')
    # plt.legend(title='Тональность')
    # st.pyplot(fig2)
    # st.subheader('Матрица корреляции')
    fig1, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)
  
st.write('Количество строк в наборе данных - {}'.format(data_len))
st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))

st.header("Модель К ближайших соседей")
# Подбор гиперпараметра
n_range_list = list(range(1,allowed_knn,step_slider))
n_range = np.array(n_range_list)
st.write('Возможные значения соседей - {}'.format(n_range))
tuned_parameters = [{'n_neighbors': n_range}]

data_y = data['music_genre']
data_X = data.drop('music_genre', axis=1)

clf_gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=cv_slider, scoring='f1_macro')
clf_gs.fit(data_X, data_y)

st.subheader('Оценка качества модели')

st.write('Лучшее значение параметров - {}'.format(clf_gs.best_params_))

# Изменение качества на тестовой выборке в зависимости от К-соседей
fig1 = plt.figure(figsize=(7,5))
ax = plt.plot(n_range, clf_gs.cv_results_['mean_test_score'])
st.pyplot(fig1)


st.sidebar.header('Модель градиентного бустинга')
st.header('Модель градиентного бустинга')
X_test = test_data.drop('music_genre', axis = 1)
y_test = test_data['music_genre']

learning_rate = st.sidebar.slider('Learning Rate:', 0.01, 1.0, 0.1, 0.01)
n_estimators = st.sidebar.slider('Number of Estimators:', 1, 100, 10)
    
    # Train the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
gb_clf.fit(data_X, data_y)
Y_pred =  gb_clf.predict(X_test)
    
st.write('Learning Rate:', learning_rate)
st.write('Number of Estimators:', n_estimators)
st.write('F1_score: ', f1_score(y_test, Y_pred, average='macro'))
fig2 = plot_multiclass_roc(Y_pred,y_test, X_test, 8)

st.pyplot(fig2)
