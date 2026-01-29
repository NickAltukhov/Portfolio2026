import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz
from transliterate import translit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
import gc

class WeighedFuzzyTextProcessor:
    """
    Класс для обработки текста и вычисления сходства между строками,
    а так же сопоставления наиболее схожих значений.

    Атрибуты:
    ----------
    df (pd.DataFrame): Основной DataFrame.
    df_to_compare (pd.DataFrame): DataFrame, в котором будет происходить поиск значений.
    column_name (str): Название столбца для обработки.
    new_column_name (str): Название нового столбца после обработки.
    substrings_to_remove (str): Сюда нужно вписать части слов, входящие в слова, которые нужно удалить (по умолчанию "шин|овая").
    
    
    Методы:
    -------
    remove_words_containing(text, substrings)
        Удаляет слова, содержащие определенные подстроки.
        
    process_column(df, column_name, new_column_name, substrings_to_remove)
        Обрабатывает столбец данных и создает новый обработанный столбец.
        
    compute_token_set_ratio(query, choices)
        Вычисляет уровень сходства между текстовыми значениями методом token_set_ratio.
        Этот метод рассчитывает расстояние Левенштейна между двумя множествами токенов, игнорирует порядок слов и дубликаты.
        
    preprocess_text_for_ngrams(text)
        Преобразует текст в нижний регистр и удаляет все символы, кроме букв и цифр (даже пробелы удаляет).
        
    create_ngrams(text, n_range)
        Создает n-граммы для заданного текста.
        
    process_ngrams(df, df_to_compare, n_range)
        Обрабатывает n-граммы и вычисляет косинусное сходство.
        
    calculate_similarity
        Выполняет полную обработку текста и вычисление сходства, добавляет к df наилучшие по схожести значения и уровень сходства.
    
    """

    def __init__(self, df, df_to_compare, column_name='Наименование', new_column_name='Наименование обработанное', substrings_to_remove="шин|овая"):
        """
        Инициализация класса WeighedFuzzyTextProcessor.

        Аргументы:
        ----------
            df (pd.DataFrame): Основной DataFrame для обработки.
            df_to_compare (pd.DataFrame): DataFrame для сравнения.
            column_name (str): Название столбца для обработки.
            new_column_name (str): Название нового столбца после обработки.
            substrings_to_remove (str): Подстроки для удаления из текста.
        """
        self.df = df
        self.df_to_compare = df_to_compare
        self.column_name = column_name
        self.new_column_name = new_column_name
        self.substrings_to_remove = substrings_to_remove

    def remove_words_containing(self, text, substrings):
        """
        Удаляет слова, содержащие заданные подстроки, из текста.

        Аргументы:
        ----------
            text (str): Исходный текст.
            substrings (str): Подстроки для удаления слов.

        Возвращает:
        ----------
            str: Обработанный текст.
        """
        pattern = r'\b\w*(' + '|'.join([substrings]) + r')\w*\b'
        return re.sub(pattern, '', text)

    def process_column(self, df, column_name, new_column_name, substrings_to_remove):
        """
        Обрабатывает текст в указанном столбце DataFrame.

        Аргументы:
        ----------
            df (pd.DataFrame): DataFrame для обработки.
            column_name (str): Название столбца для обработки.
            new_column_name (str): Название нового столбца после обработки.
            substrings_to_remove (str): Подстроки для удаления из текста.

        Возвращает:
            pd.DataFrame: Обработанный DataFrame.
        """
        df[new_column_name] = df[column_name].str.lower()
        df[new_column_name] = df[new_column_name].str.replace('ё', 'е')
        df[new_column_name] = df[new_column_name].str.replace(r'ⅱ|Ⅱ', 'ii', regex=True)
        df[new_column_name] = df[new_column_name].str.replace(r"\(|\)|\'", '', regex=True)
        df[new_column_name] = df[new_column_name].str.replace(r'\s+|\-|\_',' ', regex=True)
        df[new_column_name] = df[new_column_name].str.replace(r"(?<!\w),|,(?!\w)|(?<!\w)\.|\.(?!\w)", '', regex=True)
        df[new_column_name] = df[new_column_name].str.replace(r",", '.', regex=True)
        df[new_column_name] = df[new_column_name].apply(lambda x: self.remove_words_containing(x, substrings_to_remove))
        df[new_column_name] = df[new_column_name].apply(lambda x: translit(x, 'ru', reversed=True))
        df[new_column_name] = df[new_column_name].str.replace(r'\s+',' ', regex=True).str.strip()
        return df

    def compute_token_set_ratio(self, query, choices):
        """
        Вычисляет уровень сходства между текстовыми значениями методом token_set_ratio.
        Этот метод рассчитывает расстояние Левенштейна между двумя множествами токенов, игнорирует порядок слов и дубликаты.

        Аргументы:
        ----------
            query (str): Запрашиваемая строка.
            choices (list): Список строк для сравнения.

        Возвращает:
        ----------
            tuple: Лучшее совпадение и уровень сходства.
        """
        match = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
        if match:
            return match[0], match[1]
        else:
            return None, None

    def preprocess_text_for_ngrams(self, text):
        """
        Преобразует текст в нижний регистр и удаляет все символы, кроме букв и цифр (даже пробелы удаляет).

        Аргументы:
        ----------
            text (str): Исходный текст.

        Возвращает:
        ----------
            str: Предобработанный текст.
        """
        return ''.join(filter(str.isalnum, text.lower()))

    def create_ngrams(self, text, n_range):
        """
        Создает n-граммы для заданного текста.

        Аргументы:
        ----------
            text (str): Исходный текст.
            n_range (tuple): Диапазон значений n для n-грамм. Формат: (1, 3).

        Возвращает:
        ----------
            list: Список n-грамм.
        """
        ngrams = []
        for n in range(n_range[0], n_range[1] + 1):
            ngrams.extend([text[i:i+n] for i in range(len(text)-n+1)])
        return ngrams

    def process_ngrams(self, df, df_to_compare, n_range):
        """
        Обрабатывает n-граммы и вычисляет косинусное сходство.
        
        Алгоритм следующий:
        ----------
            1) Удаляем из текста все, кроме букв и цифр (даже пробелы удаляются).
            2) Создаем n-граммы (последовательности из n символов).
            3) Одна функция будет создавать последовательности из 1, 2, 3 символов. Вторая - из 1, 2, 3, 4 символов.
            4) Затем подсчитываем количество вхождений каждой n-граммы для каждой строки.
            5) Применяем косинусное сходство.
            6) Выделяем по 2 наилучших сходства.

        Аргументы:
        ----------
            df (pd.DataFrame): Основной DataFrame для обработки.
            df_to_compare (pd.DataFrame): DataFrame для сравнения.
            n_range (tuple): Диапазон значений n для n-грамм.

        Возвращает:
        ----------
            tuple: Индексы и значения топ-2 косинусных сходств.
        """
        df[f'{n_range[0]}-{n_range[1]}n-граммы'] = df[self.new_column_name].apply(lambda x: ' '.join(self.create_ngrams(self.preprocess_text_for_ngrams(x), n_range)))
        df_to_compare[f'{n_range[0]}-{n_range[1]}n-граммы'] = df_to_compare[self.new_column_name].apply(lambda x: ' '.join(self.create_ngrams(self.preprocess_text_for_ngrams(x), n_range)))
        combined_ngrams = pd.concat([df_to_compare[f'{n_range[0]}-{n_range[1]}n-граммы'], df[f'{n_range[0]}-{n_range[1]}n-граммы']])
        vectorizer = CountVectorizer()
        combined_count_vec = vectorizer.fit_transform(combined_ngrams)
        normalizer = Normalizer()
        combined_count_vec_norm = normalizer.fit_transform(combined_count_vec)
        count_vec_df_to_compare_norm = combined_count_vec_norm[:len(df_to_compare)]
        count_vec_df_norm = combined_count_vec_norm[len(df_to_compare):]
        cos_matrix = count_vec_df_norm.dot(count_vec_df_to_compare_norm.T).toarray()
        top_2_indices = np.argsort(cos_matrix, axis=1)[:, -2:]
        top_2_values = np.take_along_axis(cos_matrix, top_2_indices, axis=1)
        return top_2_indices, top_2_values

    def calculate_similarity(self):
        """
        Выполняет полную обработку текста и вычисление сходства.
        Алгоритм следующий:
        ----------
            1) Основной этап обработки текста.
            2) Вычисление token_set_ratio, который рассчитывает расстояние Левенштейна между двумя множествами токенов, игнорирует порядок слов и дубликаты.
            3) Дополнительная обработка текста для cos сходства n-грамм, удаляем из текста все, кроме букв и цифр (даже пробелы удаляются).
            4) Создаем n-граммы (последовательности из n символов).
            5) Одна функция будет создавать последовательности из 1, 2, 3 символов. Вторая - из 1, 2, 3, 4 символов.
            6) Затем подсчитываем количество вхождений каждой n-граммы для каждой строки.
            7) Применяем косинусное сходство.
            8) Выделяем по 2 наилучших сходства.
            9) Применяем взвешенное решение для трех полученных методов.

        Возвращает:
        ----------
            pd.DataFrame: Обработанный DataFrame с результатами сходства.
        """
        self.df = self.process_column(self.df, self.column_name, self.new_column_name, self.substrings_to_remove)
        self.df_to_compare = self.process_column(self.df_to_compare, self.column_name, self.new_column_name, self.substrings_to_remove)

        best_matches = self.df[self.new_column_name].apply(lambda x: self.compute_token_set_ratio(x, self.df_to_compare[self.new_column_name]))
        self.df['token_set_ratio уровень сходства'] = best_matches.apply(lambda x: x[1])
        self.df['token_set_ratio лучшее совпадение'] = best_matches.apply(lambda x: x[0])
        
        top_2_indices_1_3, top_2_values_1_3 = self.process_ngrams(self.df, self.df_to_compare, (1, 3))
        top_2_indices_1_4, top_2_values_1_4 = self.process_ngrams(self.df, self.df_to_compare, (1, 4))
        
        gc.collect()

        self.df['Топ1 индекс cos сходства 1_3'] = top_2_indices_1_3[:, -1]
        self.df['Топ2 индекс cos сходства 1_3'] = top_2_indices_1_3[:, -2]
        self.df['Топ1 уровень cos сходства 1_3'] = top_2_values_1_3[:, -1]
        self.df['Топ2 уровень cos сходства 1_3'] = top_2_values_1_3[:, -2]
        self.df['Cos сходство топ1 совпадение 1-3'] = self.df['Топ1 индекс cos сходства 1_3'].apply(lambda x: self.df_to_compare[self.new_column_name].iloc[x])
        self.df['Cos сходство топ2 совпадение 1-3'] = self.df['Топ2 индекс cos сходства 1_3'].apply(lambda x: self.df_to_compare[self.new_column_name].iloc[x])

        self.df['Топ1 индекс cos сходства 1_4'] = top_2_indices_1_4[:, -1]
        self.df['Топ2 индекс cos сходства 1_4'] = top_2_indices_1_4[:, -2]
        self.df['Топ1 уровень cos сходства 1_4'] = top_2_values_1_4[:, -1]
        self.df['Топ2 уровень cos сходства 1_4'] = top_2_values_1_4[:, -2]
        self.df['Cos сходство топ1 совпадение 1-4'] = self.df['Топ1 индекс cos сходства 1_4'].apply(lambda x: self.df_to_compare[self.new_column_name].iloc[x])
        self.df['Cos сходство топ2 совпадение 1-4'] = self.df['Топ2 индекс cos сходства 1_4'].apply(lambda x: self.df_to_compare[self.new_column_name].iloc[x])

        self.df['Топ1 уровень cos сходства 1_3'] = self.df['Топ1 уровень cos сходства 1_3'] * 100
        self.df['Топ2 уровень cos сходства 1_3'] = self.df['Топ2 уровень cos сходства 1_3'] * 100
        self.df['Топ1 уровень cos сходства 1_4'] = self.df['Топ1 уровень cos сходства 1_4'] * 100
        self.df['Топ2 уровень cos сходства 1_4'] = self.df['Топ2 уровень cos сходства 1_4'] * 100

        self.df.drop(['Топ1 индекс cos сходства 1_4', 'Топ2 индекс cos сходства 1_4', 'Топ1 индекс cos сходства 1_3', 'Топ2 индекс cos сходства 1_3', '1-3n-граммы', '1-4n-граммы'], axis=1, inplace=True)

        self.df = self.df.round(3)

        self.df['Взвешенное сходство'] = None
        self.df['Уровень взвеш. сходства'] = None

        for index, row in self.df.iterrows():
            if row['Топ1 уровень cos сходства 1_3'] >= row['token_set_ratio уровень сходства'] and row['Топ1 уровень cos сходства 1_3'] >= 97:
                self.df.at[index, 'Взвешенное сходство'] = row['Cos сходство топ1 совпадение 1-3']
                self.df.at[index, 'Уровень взвеш. сходства'] = row['Топ1 уровень cos сходства 1_3']
            elif row['token_set_ratio уровень сходства'] >= 97:
                self.df.at[index, 'Взвешенное сходство'] = row['token_set_ratio лучшее совпадение']
                self.df.at[index, 'Уровень взвеш. сходства'] = row['token_set_ratio уровень сходства']
            elif row['Cos сходство топ1 совпадение 1-3'] == row['Cos сходство топ1 совпадение 1-4']:
                self.df.at[index, 'Взвешенное сходство'] = row['Cos сходство топ1 совпадение 1-3']
                self.df.at[index, 'Уровень взвеш. сходства'] = row['Топ1 уровень cos сходства 1_3']
            elif row['Cos сходство топ1 совпадение 1-3'] == row['Cos сходство топ2 совпадение 1-4'] and row['Cos сходство топ2 совпадение 1-3'] == row['Cos сходство топ1 совпадение 1-4']:
                mean_cos_sim_top2ngr3_top1ngr4 = (row['Топ2 уровень cos сходства 1_3'] + row['Топ1 уровень cos сходства 1_4']) / 2
                mean_cos_sim_top1ngr3_top2ngr4 = (row['Топ1 уровень cos сходства 1_3'] + row['Топ2 уровень cos сходства 1_4']) / 2
                if mean_cos_sim_top2ngr3_top1ngr4 >= mean_cos_sim_top1ngr3_top2ngr4:
                    self.df.at[index, 'Взвешенное сходство'] = row['Cos сходство топ1 совпадение 1-4']
                    self.df.at[index, 'Уровень взвеш. сходства'] = row['Топ1 уровень cos сходства 1_4']
                else:
                    self.df.at[index, 'Взвешенное сходство'] = row['Cos сходство топ1 совпадение 1-3']
                    self.df.at[index, 'Уровень взвеш. сходства'] = row['Топ1 уровень cos сходства 1_3']
            else:
                diff_cos_sim_top1ngr4_top2ngr4 = abs(row['Топ1 уровень cos сходства 1_4'] - row['Топ2 уровень cos сходства 1_4'])
                diff_cos_sim_top1ngr3_top1ngr3 = abs(row['Топ1 уровень cos сходства 1_3'] - row['Топ2 уровень cos сходства 1_3'])
                if diff_cos_sim_top1ngr4_top2ngr4 > diff_cos_sim_top1ngr3_top1ngr3:
                    self.df.at[index, 'Взвешенное сходство'] = row['Cos сходство топ1 совпадение 1-4']
                    self.df.at[index, 'Уровень взвеш. сходства'] = row['Топ1 уровень cos сходства 1_4']
                elif diff_cos_sim_top1ngr4_top2ngr4 == 0 and diff_cos_sim_top1ngr3_top1ngr3 == 0:
                    if row['token_set_ratio лучшее совпадение'] in [row['Cos сходство топ1 совпадение 1-4'], row['Cos сходство топ2 совпадение 1-4'], row['Cos сходство топ1 совпадение 1-3'], row['Cos сходство топ2 совпадение 1-3']]:
                        self.df.at[index, 'Взвешенное сходство'] = row['token_set_ratio лучшее совпадение']
                        self.df.at[index, 'Уровень взвеш. сходства'] = row['Топ1 уровень cos сходства 1_3'] if (row['Топ1 уровень cos сходства 1_3'] >= row['token_set_ratio уровень сходства']) else row['token_set_ratio уровень сходства']
                    else:
                        self.df.at[index, 'Взвешенное сходство'] = row['Cos сходство топ1 совпадение 1-4']
                        self.df.at[index, 'Уровень взвеш. сходства'] = row['Топ1 уровень cos сходства 1_3'] if (row['Топ1 уровень cos сходства 1_3'] >= row['Топ1 уровень cos сходства 1_4']) else row['Топ1 уровень cos сходства 1_4']
                else:
                    self.df.at[index, 'Взвешенное сходство'] = row['Cos сходство топ1 совпадение 1-3']
                    self.df.at[index, 'Уровень взвеш. сходства'] = row['Топ1 уровень cos сходства 1_3']

        return self.df

