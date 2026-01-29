import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, time
import uuid

class TestDataGenerator:
    """
    Класс для генерации различных тестовых случайных данных.

    Атрибуты:
    ----------
    data_frame (pd.DataFrame): Основной DataFrame для добавления данных.

    Методы:
    -------
    add_categorical_column(column_name, categories, probabilities=None)
        Добавляет столбец с категориальными данными с заданными вероятностями.

    add_discrete_column(column_name, distribution, *params)
        Добавляет столбец с дискретными случайными значениями из заданного распределения.

    add_continuous_column(column_name, distribution, *params)
        Добавляет столбец с непрерывными случайными значениями из заданного распределения.

    add_date_column(column_name, start_date, end_date)
        Добавляет столбец с случайными датами в заданном диапазоне.

    add_string_column(column_name, length)
        Добавляет столбец с случайными строками заданной длины.

    add_boolean_column(column_name, true_probability=0.5)
        Добавляет столбец с булевыми значениями с заданной вероятностью True.

    add_time_column(column_name)
        Добавляет столбец с случайными значениями времени суток.

    add_uuid_column(column_name)
        Добавляет столбец с уникальными идентификаторами (UUID).
    """

    def __init__(self, data_frame):
        """
        Инициализация класса TestDataGenerator.

        Аргументы:
        ----------
            data_frame (pd.DataFrame): Основной DataFrame для добавления данных.
        """
        self.data_frame = data_frame

    def add_categorical_column(self, column_name, categories, probabilities=None):
        """
        Добавляет столбец с категориальными данными с заданными вероятностями.

        Аргументы:
        ----------
            column_name (str): Название столбца.
            categories (list): Список категорий.
            probabilities (list, optional): Список вероятностей для каждой категории.
                Если не задано, вероятности будут равномерно распределены.

        Пример:
        ----------
            test_data_generator.add_categorical_column('Категории', ['A', 'B', 'C'], [0.1,0.3,0.6])
            
        """
        self.data_frame[column_name] = np.random.choice(categories, size=len(self.data_frame), p=probabilities)

    def add_discrete_column(self, column_name, distribution, *params):
        """
        Добавляет столбец с дискретными случайными значениями из заданного распределения.

        Аргументы:
        ----------
            column_name (str): Название столбца.
            distribution (str): Тип распределения ('uniform', 'poisson', 'binomial', 'negative_binomial', 'geometric', 'hypergeometric').
            *params: Параметры распределения.

        Примеры:
        ----------
        Равномерное.
            test_data_generator.add_discrete_column('Равномерное распределение', 'uniform', 1, 10)
            Параметры: low, high
            
        Пуассона. Моделирование числа событий, происходящих в фиксированном интервале времени или пространства.
            test_data_generator.add_discrete_column('Распределение Пуассона', 'poisson', 5)
            Параметр: lambda (среднее число событий в фиксированном интервале)
            
        Биномиальное. Количество успехов в n испытаниях с определенной вероятностью.
            test_data_generator.add_discrete_column('Биномиальное распределение', 'binomial', 10, 0.5)
            Параметры: n (кол-во испытаний), p (вероятность успеха)
            
        Негативное биномиальное. Количество неудач до того как произойдет n успехов.
            test_data_generator.add_discrete_column('Негативное биномиальное распределение', 'negative_binomial', 10, 0.5)
            Параметры: n (количество успехов, которые нужно достичь), p (вероятность успеха)
             
        Геометрическое. Количество испытаний до первого успеха.
            test_data_generator.add_discrete_column('Геометрическое', 'geometric', 0.5)
            Параметр: p (вероятность успеха)
            
        Гипергеометрическое. Количество выбранных элементов нужного типа из общего количества.
            test_data_generator.add_discrete_column('Гипергеометрическое', 'hypergeometric', 10, 5, 3)
            Параметры: ngood (общее количество нужного типа), nbad (ненужного), nsample (размер выборки)
            
            
        """
        if distribution == 'uniform':
            low, high = params
            self.data_frame[column_name] = np.random.randint(low, high, size=len(self.data_frame))
        elif distribution == 'poisson':
            lam = params[0]
            self.data_frame[column_name] = np.random.poisson(lam, size=len(self.data_frame))
        elif distribution == 'binomial':
            n, p = params
            self.data_frame[column_name] = np.random.binomial(n, p, size=len(self.data_frame))
        elif distribution == 'negative_binomial':
            n, p = params
            self.data_frame[column_name] = np.random.negative_binomial(n, p, size=len(self.data_frame))
        elif distribution == 'geometric':
            p = params[0]
            self.data_frame[column_name] = np.random.geometric(p, size=len(self.data_frame))
        elif distribution == 'hypergeometric':
            ngood, nbad, nsample = params
            self.data_frame[column_name] = np.random.hypergeometric(ngood, nbad, nsample, size=len(self.data_frame))
        else:
            raise ValueError(f"Неизвестное распределение: {distribution}")

    def add_continuous_column(self, column_name, distribution, *params):
        """
        Добавляет столбец с непрерывными случайными значениями из заданного распределения.

        Аргументы:
        ----------
            column_name (str): Название столбца.
            distribution (str): Тип распределения ('normal', 'uniform', 'exponential', 'lognormal').
            *params: Параметры распределения.

        Примеры:
        ----------
        Нормальное.
            test_data_generator.add_continuous_column('Нормальное', 'normal', 0, 1)
            Параметры: mean, std
            
        Непрерывное равномерное.
            test_data_generator.add_continuous_column('Непрерывное равномерное', 'uniform', 0, 1)
            Параметры: low, high
            
        Экспоненциальное. Время ожидания до следующего события.
            test_data_generator.add_continuous_column('Экспоненциальное', 'exponential', 1.0)
            Параметры: mean (среднее количество испытаний в единицу времени)
            
        Логнормальное.
            test_data_generator.add_continuous_column('Логнормальное', 'lognormal', 0, 1)
            Параметры: mean, std
            
            
        """
        if distribution == 'normal':
            mean, std = params
            self.data_frame[column_name] = np.random.normal(mean, std, size=len(self.data_frame))
        elif distribution == 'uniform':
            low, high = params
            self.data_frame[column_name] = np.random.uniform(low, high, size=len(self.data_frame))
        elif distribution == 'exponential':
            scale = 1/params[0]
            self.data_frame[column_name] = np.random.exponential(scale, size=len(self.data_frame))
        elif distribution == 'lognormal':
            mean, sigma = params
            self.data_frame[column_name] = np.random.lognormal(mean, sigma, size=len(self.data_frame))
        else:
            raise ValueError(f"Неизвестное распределение: {distribution}")

    def add_date_time_column(self, column_name, mode, start_date=None, end_date=None):
        """
        Добавляет столбец с случайными датами, временем или датой и временем в заданном диапазоне.

        Аргументы:
        ----------
            column_name (str): Название столбца.
            mode (str): метод ('datetime', 'date', 'time').
            start_date (str, optional): Начальная дата в формате 'YYYY-MM-DD'. Обязательно для методов 'date' и 'datetime'.
            end_date (str, optional): Конечная дата в формате 'YYYY-MM-DD'. Обязательно для методов 'date' и 'datetime'.

        Примеры:
        ----------
            test_data_generator.add_date_time_column('Дата и время', 'datetime', '2020-01-01', '2022-12-31')
            
            test_data_generator.add_date_time_column('Дата', 'date', '2020-01-01', '2022-12-31')
            
            test_data_generator.add_date_time_column('Время', 'time')
            
        """
        if mode == 'datetime':
            if start_date is None or end_date is None:
                raise ValueError("start_date и end_date должны быть указаны с mode='datetime'")
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            delta = end_date - start_date
            self.data_frame[column_name] = [start_date + timedelta(days=random.randint(0, delta.days), hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59)) for _ in range(len(self.data_frame))]
        elif mode == 'date':
            if start_date is None or end_date is None:
                raise ValueError("start_date и end_date должны быть указаны с mode='date'")
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            delta = end_date - start_date
            self.data_frame[column_name] = [start_date + timedelta(days=random.randint(0, delta.days)) for _ in range(len(self.data_frame))]
        elif mode == 'time':
            self.data_frame[column_name] = [time(hour=random.randint(0, 23), minute=random.randint(0, 59), second=random.randint(0, 59)) for _ in range(len(self.data_frame))]
        else:
            raise ValueError(f"Неизвестное значение атрибута mode: {mode}")

    def add_string_column(self, column_name, length):
        """
        Добавляет столбец с случайными строками заданной длины и префикса.

        Аргументы:
        ----------
            column_name (str): Название столбца.
            length (int): Длина строк.

        Пример:
        ----------
            test_data_generator.add_string_column('Текст', 5)
            
        """
        self.data_frame[column_name] = [''.join(random.choices('абвгдеёжзийклмнопрстуфхцчшщъыьэюя ', k=length)) for _ in range(len(self.data_frame))]

    def add_boolean_column(self, column_name, true_probability=0.5):
        """
        Добавляет столбец с булевыми значениями с заданной вероятностью True.

        Аргументы:
        ----------
            column_name (str): Название столбца.
            true_probability (float, optional): Вероятность значения True. По умолчанию 0.5.

        Пример:
        ----------
            test_data_generator.add_boolean_column('True и False', true_probability=0.7)
            
        """
        self.data_frame[column_name] = np.random.choice([True, False], size=len(self.data_frame), p=[true_probability, 1 - true_probability])


    def add_uuid_column(self, column_name):
        """
        Добавляет столбец с уникальными идентификаторами (UUID).

        Аргументы:
        ----------
            column_name (str): Название столбца.

        Пример:
        ----------
            test_data_generator.add_uuid_column('Идентификатор uuid')
        
        """
        self.data_frame[column_name] = [uuid.uuid4() for _ in range(len(self.data_frame))]


