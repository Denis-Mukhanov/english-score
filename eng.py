# библиотеки проекта
import streamlit as st
import csv
import pysrt
import re
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from catboost import CatBoostRegressor
# константы
HTML = r'<.*?>'
TAG = r'{.*?}'
COMMENTS = r'[\(\[][A-Z ]+[\)\]]'
LETTERS = r'[^a-zA-Z\'.,!? ]'
SPACES = r'([ ])\1+'
DOTS = r'[\.]+'
CLEAN = re.compile('[^а-яa-z\s]')

# функция для первичной обработки текста
def clean_subs(txt):
    txt = re.sub(HTML, ' ', txt) #html тэги меняем на пробел
    txt = re.sub(TAG, ' ', txt) #тэги меняем на пробел
    txt = re.sub(COMMENTS, ' ', txt) #комменты меняем на пробел
    txt = re.sub(LETTERS, ' ', txt) #все что не буквы меняем на пробел
    txt = re.sub(SPACES, r'\1', txt) #повторяющиеся пробелы меняем на один пробел
    txt = re.sub(DOTS, r'.', txt) #многоточие меняем на точку
    txt = txt.encode('ascii', 'ignore').decode() #удаляем все что не ascii символы
    txt = ".".join(txt.lower().split('.')[1:-1]) #удаляем первый и последний субтитр (обычно это реклама)
    txt = txt.replace('. .', '. ')
    txt = CLEAN.sub('', txt)
    return txt

# функция для удаления стоп слов и токинизации
def stopwords_tokenize(x):
    tokens = word_tokenize(x)
    tokenization = [word for word in tokens if not word in stopwords.words('english')]
    return tokenization

# функция для стеминга и лематизации
def stemmer_lemmatizer(x):
    stemmer = [porter_stemmer.stem(s) for s in x]
    lemmatizer = [wordnet_lemmatizer.lemmatize(w) for w in stemmer]
    return lemmatizer

# oxford by CEFR
a1_list = next(csv.reader(open('data/a1.csv', 'r')))
a2_list = next(csv.reader(open('data/a2.csv', 'r')))
b1_list = next(csv.reader(open('data/b1.csv', 'r')))
b2_list = next(csv.reader(open('data/b2.csv', 'r')))
c1_list = next(csv.reader(open('data/c1.csv', 'r')))
# model
model = CatBoostRegressor()
model.load_model('models/catboost_model')
# CSS стиль для центрирования заголовка
title_style = """
    <style>
    .title {
        text-align: center;
    }
    </style>
"""
# Вставляем CSS стиль перед заголовком
st.markdown(title_style, unsafe_allow_html=True)
# Заголовок
st.markdown("<h1 class='title'>Определение уровня сложности англоязычных фильмов</h1>", unsafe_allow_html=True)
image = "https://i.postimg.cc/GpMx0rJ0/image.png"
st.image(image, caption="", use_column_width=True)

# Отступ
st.write("")

# Описание поля загрузки файла
st.write("Загрузите файл:")
# Поле для загрузки файла .srt
srt_file = st.file_uploader("", type=".srt")

if srt_file:
    # Получаем байтовый поток для загруженного файла
    file_bytes = srt_file.read()
    file_string = file_bytes.decode('latin-1')

    # Обработка загруженного файла
    subs = pysrt.from_string(file_string)

    if len(subs) == 0:
        file_string = file_bytes.decode('utf-16')
        subs = pysrt.from_string(file_string)
    subs = ' '.join([i.text for i in subs])

    # fres/fkgl
    sentences = len(re.split(r"[.!?]", subs))
    words = len(subs.split(' '))
    syllables = sum(subs.count(g) for g in 'aeoiu') + subs.count('y') / 2

    fres = 206.835 - (words / sentences) * 1.015 - (syllables / words) * 84.6
    fkgl = (words / sentences) * 0.39 + (syllables / words) * 11.8 - 15.59

    # Очистка
    subs = clean_subs(subs)
    subs = stopwords_tokenize(subs)
    subs = stemmer_lemmatizer(subs)

    # oxford by CEFR
    a1 = sum(1 for i in subs if i in a1_list)
    a2 = sum(1 for i in subs if i in a2_list)
    b1 = sum(1 for i in subs if i in b1_list)
    b2 = sum(1 for i in subs if i in b2_list)
    c1 = sum(1 for i in subs if i in c1_list)

    count_word = a1 + a2 + b1 + b2 + c1
    a1 = a1 / count_word
    a2 = a2 / count_word
    b1 = b1 / count_word
    b2 = b2 / count_word
    c1 = c1 / count_word

    if len(subs)<100:
        st.text('Недостаточно данных')
    else:
        df = pd.DataFrame(columns=['subtitles', 'a1', 'a2', 'b1', 'b2', 'c1', 'fres', 'fkgl'])
        data = []
        for i in range(len(subs) // 100):
            data.append({'subtitles': ' '.join(subs[i*100 : (i + 1)*100]),
                         'a1': a1,
                         'a2': a2,
                         'b1': b1,
                         'b2': b2,
                         'c1': c1,
                         'fres': fres,
                         'fkgl': fkgl})
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        predictions = model.predict(df)
        predictions = np.mean(predictions)
        # Кнопка "Определить уровень сложности"
        if st.button("Определить уровень сложности"):
            if predictions < 3.5:
                st.text('CEFR: A2')
            elif  predictions < 4:
                st.text('CEFR: A2/B1')
            elif  predictions < 5:
                st.text('CEFR: B1')
            elif  predictions < 5.5:
                st.text('CEFR: B1/B2')
            elif  predictions < 6.5:
                st.text('CEFR: B2')
            elif predictions < 7:
                st.text('CEFR: B1/C1')
            else:
                st.text('CEFR: C1')
