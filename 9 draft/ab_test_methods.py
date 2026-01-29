import pandas as pd
import numpy as np
from typing import Optional, Callable
from scipy.stats import mannwhitneyu
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

class NonParametricMethods_2samp:
    """
    Класс для проведения непараметрических статистических тестов сравнения двух выборок A и B.

    Параметры
    ----------
    a : pd.Series
        Первая выборка данных.
    b : pd.Series
        Вторая выборка данных.
        
    Методы:
    -------
    m_whitneyu(q)
        Проводит тест Манна–Уитни для сравнения двух независимых выборок.
        
    bootstrap_ci
        Оценивает 95% доверительный интервал разницы статистики (по умолчанию среднего) 
        двух выборок A и B с помощью bootstrap-метода.
        
    perm_test
        Проводит перестановочный тест для сравнения двух выборок A и B.

    Примечание
    ----------
    Для параметра alternative:
      - 'big_b'   : проверка альтернативной гипотезы о том, что выборка B больше выборки A;
      - 'small_b' : проверка альтернативной гипотезы о том, что выборка B меньше выборки A;
      - 'two_sided' : двусторонняя проверка.
    """

    def __init__(self, a: pd.Series, b: pd.Series):
        """
        Инициализация и сохранение переданных выборок, а так же вычисление их размеров.

        Параметры
        ----------
        a : pd.Series
            Первая выборка данных.
        b : pd.Series
            Вторая выборка данных.
        """
        self.a = a
        self.b = b
        self.n_a = len(a)
        self.n_b = len(b)

    def m_whitneyu(self, alternative: str = 'big_b'):
        """
        Проводит тест Манна–Уитни для сравнения двух независимых выборок.
        Использует библиотечную функцию `mannwhitneyu` из scipy.

        Параметры
        ----------
        alternative : str, по умолчанию 'big_b'
            Определяет тип альтернативы:
              - 'big_b'    : альтернативная гипотеза, что B > A;
              - 'small_b'  : альтернативная гипотеза, что B < A;
              - 'two_sided': двусторонняя проверка.

        Возвращает
        ----------
        MannwhitneyuResult
            Результат теста Манна–Уитни (значение статистики и p-значение).
        """
        if alternative == 'big_b':
            alternative = 'less'
        elif alternative == 'small_b':
            alternative = 'greater'
        else:
            alternative = 'two-sided'
        return mannwhitneyu(self.a, self.b, alternative=alternative)

    def bootstrap_ci(self, a:Optional[pd.Series] = None, b:Optional[pd.Series] = None, confidence: float = 0.95, n_iterations: int = 30_000, func: Callable = np.mean, alternative: str = 'big_b', print_:bool=True):
        """
        Оценивает 95% доверительный интервал разницы статистики (по умолчанию среднего) двух выборок A и B
        с помощью bootstrap-метода.

        Параметры
        ----------
        a : Опционально(pd.Series), если None - указывается атрибут экземпляра self.a
            Первая выборка данных.
        b : Опционально(pd.Series), если None - указывается атрибут экземпляра self.b
            Вторая выборка данных.
        confidence : float, по умолчанию 0.95
            Уровень доверия для доверительного интервала.
        n_iterations : int, по умолчанию 30_000
            Количество бутстрап-итераций.
        func : Callable, по умолчанию np.mean
            Функция, применяемая к выборкам (например, np.mean, np.median).
        alternative : str, по умолчанию 'big_b'
            Определяет тип альтернативы:
              - 'big_b'   : альтернативная гипотеза, что B > A;
              - 'small_b' : альтернативная гипотеза, что B < A;
              - 'two_sided' : двусторонняя проверка.

        Возвращает
        ----------
            Функция выводит на экран доверительный интервал (или его границы).
        """
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        func_name = func.__name__
        bootstrap_diffs = []

        samples_a = np.random.choice(a, size=(n_iterations, self.n_a), replace=True)
        samples_b = np.random.choice(b, size=(n_iterations, self.n_b), replace=True)
        bootstrap_diffs = func(samples_a, axis=1) - func(samples_b, axis=1)

        if alternative == 'big_b':
            alpha = 100 * (1 - confidence)
            ci_upper = np.percentile(bootstrap_diffs, 100 - alpha)
            if print_==True:
                print(
                    f"Верхняя граница {int(confidence*100)}% доверительного интервала разницы {func_name} значений: {ci_upper}\n"
                    f"Для подтверждения альтернативной гипотезы, что {func_name}(B) > {func_name}(A), "
                    f"верхняя граница должна быть ниже нуля"
                )
            ci = ci_upper
        elif alternative == 'small_b':
            alpha = 100 * (1 - confidence)
            ci_lower = np.percentile(bootstrap_diffs, alpha)
            if print_==True:
                print(
                    f"Нижняя граница {int(confidence*100)}% доверительного интервала разницы {func_name} значений: {ci_lower}\n"
                    f"Для подтверждения альтернативной гипотезы, что {func_name}(B) < {func_name}(A), "
                    f"нижняя граница должна быть выше нуля"
                )
            ci = ci_lower
        else:
            alpha = 100 * ((1 - confidence) / 2)
            ci_lower = np.percentile(bootstrap_diffs, alpha)
            ci_upper = np.percentile(bootstrap_diffs, 100 - alpha)
            if print_==True:
                print(
                    f"{int(confidence*100)}% Доверительный интервал разницы {func_name} значений: ({ci_lower}, {ci_upper})"
                )
            ci = (ci_lower, ci_upper)

        return ci
        
    def bootstrap_power_analysis(self, delta:float, n_sim:int = 1000, n_bootstrap_iterations:int = 10_000, confidence:float = 0.95, func:Callable = np.mean, alternative:str= 'big_b',
                                mode:str = 'real')->float:
        """
        Оценивает мощность (power) теста на основе bootstrap-метода для сравнения двух выборок A и B.
    
        Шаги:
        1) В цикле n_sim раз бутстрапим обе выборки:
           synthetic_a = выборка из A
           synthetic_b = выборка из B
        2) Смещаем synthetic_b на delta (synthetic_b += delta).
        3) Строим доверительный интервал через метод bootstrap_ci
        4) Проверяем, отвергается ли H0 в пользу выбранной альтернативы (big_b / small_b / two_sided).
        5) Мощность = доля симуляций, в которых H0 отвергается.
    
        Параметры
        ----------
        delta: float
            Смещение, которое моделируем для выборки B.
        n_sim: int, по умолчанию 1_000
            Количество повторов (симуляций) для оценки мощности.
        n_bootstrap_iterations: int, по умолчанию 10_000
            Число итераций внутри bootstrap для оценки интервала.
        confidence: float, по умолчанию 0.95
            Уровень доверия для bootstrap_ci.
        func: Callable, по умолчанию np.mean
            Статистика (среднее, медиана и т. п.), которая сравнивается между A и B.
        alternative: {'big_b', 'small_b', 'two_sided'}, по умолчанию 'big_b'
            Формулировка альтернативной гипотезы.
        mode: {'real', 'simulated'}, по умолчанию 'real'
            'real' - использование обоих выборок для проведения сравнения.
            'simulated' - использование выборки A и симулированной копии этой выборки + delta (смещение).
            
    
        Возвращает
        ----------
        float
            Мощность теста (доля случаев, когда доверительный интервал указывает на отличие в нужную сторону).
        """


        successes = 0

        for _ in range(n_sim):
            if mode=='real':
                synthetic_a = pd.Series(np.random.choice(self.a, size=self.n_a, replace=True))
                synthetic_b = pd.Series(np.random.choice(self.b, size=self.n_b, replace=True) + delta)
            else:
                synthetic_a = pd.Series(np.random.choice(self.a, size=self.n_a, replace=True))
                synthetic_b = pd.Series(np.random.choice(self.a, size=self.n_b, replace=True)+delta)
    
            ci = self.bootstrap_ci(a=synthetic_a, b=synthetic_b, confidence=confidence, n_iterations=n_bootstrap_iterations, func=func, alternative=alternative, print_=False)
    
            if alternative == 'big_b':
                if ci < 0:
                    successes += 1
            elif alternative == 'small_b':
                if ci > 0:
                    successes += 1
            else:
                lower, upper = ci
                if (upper < 0) or (lower > 0):
                    successes += 1
    
        return successes / n_sim


    def perm_test(self, n_iterations: int = 30_000, func: Callable = np.mean, alternative: str ='big_b'):
        """
        Проводит перестановочный тест для сравнения двух выборок A и B.
        Оценка вероятности (p-значения) основана на перестановках элементов объединенной выборки.

        Параметры
        ----------
        n_iterations : int, по умолчанию 30_000
            Количество перестановок.
        func : Callable, по умолчанию np.mean
            Функция, применяемая к выборкам для оценки (например, np.mean, np.median).
        alternative : str, по умолчанию 'big_b'
            Определяет тип альтернативы:
              - 'big_b'   : альтернативная гипотеза, что B > A;
              - 'small_b' : альтернативная гипотеза, что B < A;
              - 'two_sided': двусторонняя проверка.

        Возвращает
        ----------
            Функция выводит p-значение на экран и возвращает None.
        """
        func_name = func.__name__
        real_diff = func(self.a) - func(self.b)
        combined = np.concatenate([self.a, self.b])

        perms = np.array([np.random.permutation(combined) for _ in range(n_iterations)])
        new_a = perms[:, :len(self.a)]
        new_b = perms[:, len(self.a):]

        diffs = func(new_a, axis=1) - func(new_b, axis=1)

        if alternative == "big_b":
            p_value = np.mean(diffs <= real_diff)
            p = print(
                f"p-значение для разниц {func_name} значений выборок: {p_value}\n"
                f"Если p-значение меньше заданного ур. значимости, мы принимаем альтернативную "
                f"гипотезу о том, что {func_name}(B) > {func_name}(A)"
            )
        elif alternative == "small_b":
            p_value = np.mean(diffs >= real_diff)
            p = print(
                f"p-значение для разниц {func_name} значений выборок: {p_value}\n"
                f"Если p-значение меньше заданного ур. значимости, мы принимаем альтернативную "
                f"гипотезу о том, что {func_name}(B) < {func_name}(A)"
            )
        else:
            p_value = np.mean(np.abs(diffs) >= np.abs(real_diff))
            p = print(
                f"p-значение для разниц {func_name} значений выборок: {p_value}\n"
                f"Если p-значение меньше заданного ур. значимости, мы принимаем альтернативную "
                f"гипотезу о том, что B и A не происходят из одного распределения (двусторонняя проверка)"
            )
        return p






class DeltaAndSampleSize:
    """
    Класс для вычисления размера эффекта и оценки размера выборки для A/B тестирования.

    Параметры
    ----------
    a : pd.Series
        Первая выборка данных.
    b : pd.Series
        Вторая выборка данных.

    Методы:
    -------
    cohens_delta()
        Вычисляет размер эффекта по Коэну для двух выборок.
        Показывает, насколько сильно различаются две группы в стандартных отклонениях.
        

    glass_delta()
        Вычисляет размер эффекта по Глассу для двух выборок.
        Показывает, насколько сильно различаются две группы в стандартном отклонении выборки A.
    
    proportion_delta(p1, p2)
        Вычисляет размер эффекта для пропорций.
        
    cliffs_delta()
        Вычисляет размер эффекта по Клиффу для двух выборок.
        Размер эффекта по Клиффу измеряет степень различия между двумя распределениями, 
        сравнивая каждую пару значений из A и B. Результат определяется 
        как разность количества пар, где значение из B больше значения из A,
        и количества пар, где значение из B меньше значения из A,
        деленная на общее число пар (n1 * n2).

    sample_size(effect_size, alternative='big_b')
        Оценивает необходимый размер выборки.

    Примечание
    ----------
    Для параметра alternative:
      - 'big_b'   : проверка альтернативной гипотезы о том, что выборка B больше выборки A;
      - 'small_b' : проверка альтернативной гипотезы о том, что выборка B меньше выборки A;
      - 'two_sided' : двусторонняя проверка.
    """
    def __init__(self, a: pd.Series, b: pd.Series):
        """
        Инициализация и сохранение переданных выборок, а так же вычисление их размеров.

        Параметры
        ----------
        a : pd.Series
            Первая выборка данных.
        b : pd.Series
            Вторая выборка данных.
        """
        self.a = a
        self.b = b
        self.n_a = len(a)
        self.n_b = len(b)
        
    def cohens_delta(self)->float:
        """
        Вычисляет размер эффекта по Коэну для двух выборок.

        Показывает, насколько сильно различаются две группы в единицах стандартных отклонений.

        Возвращает
        ----------
        float
            Значение размера эффекта Коэна.
            Большие абсолютные значения указывают на больший размер эффекта.
        """
        std_a = np.std(self.a, ddof=1)
        std_b = np.std(self.b, ddof=1)
        pooled_std = np.sqrt(((self.n_a - 1) * std_a**2 + (self.n_b - 1) * std_b**2) / (self.n_a + self.n_b - 2))
        delta = (np.mean(self.b) - np.mean(self.a)) / pooled_std
        return delta
        
    def glass_delta(self)->float:
        """
        Вычисляет размер эффекта по Глассу для двух выборок.

        Похожа на дельту Коэна, но использует стандартное отклонение только первой
        выборки. Это полезно, когда стандартные отклонения выборок значительно различаются.

        Возвращает
        ----------
        float
            Значение размера эффекта по Глассу.
            Большие абсолютные значения указывают на больший размер эффекта.
        """
        delta = (np.mean(self.b) - np.mean(self.a)) / np.std(self.a, ddof=1)
        return delta
    def proportion_delta(self, p1:float, p2:float)->float:
        """
        Вычисляет размер эффекта для пропорций.

        Использует функцию `proportion_effectsize` из `statsmodels.stats.proportion`.

        Параметры
        ----------
        p1 : float
            Пропорция в первой группе.
        p2 : float
            Пропорция во второй группе.

        Возвращает
        ----------
        float
            Размер эффекта для пропорций.
        """
        return proportion_effectsize(p1, p2)
        
    def cliffs_delta(self)->float:
        """
        Вычисляет размер эффекта по Клиффу для двух выборок A и B.
        
        Размер эффекта по Клиффу измеряет степень различия между двумя распределениями, 
        сравнивая каждую пару значений из A и B. Результат определяется 
        как разность количества пар, где значение из B больше значения из A,
        и количества пар, где значение из B меньше значения из A,
        деленная на общее число пар (n1 * n2).
        
        Параметры:
        ----------
            a: pd.Series
            b: pd.Series
        
        Возвращаемое значение:
        ----------
            delta: значение эффекта по методу Cliff's Delta, принимающее значения 
                   от -1 до 1. Значение близкое к 1 означает B>A, -1 означает A>B.
                   Значение, близкое к 0, означает небольшой размер эффекта.
                   
        Пример:
        ----------
            Допустим, получился размер эффекта равный 0.07, это означает:
            вероятность того, что случайно выбранное значение из группы B окажется больше
            случайно выбранного значения из группы A, примерно на 7 процентных пунктов выше,
            чем вероятность противоположного исхода.
        """
        a = np.array(self.a)
        b = np.array(self.b)
    
        diff = b[:, None] - a[None, :]
        
        more = np.sum(diff > 0)
        less = np.sum(diff < 0)
        
        delta = (more - less) / (self.n_a * self.n_b)
        return delta
        
    def sample_size(self, effect_size:float, alternative:str = 'big_b', alpha:float = 0.05, power:float = 0.8)->float:
        """
        Оценивает необходимый размер выборки для независимого t-теста Стьюдента
        с заданной мощностью, уровнем значимости и размером эффекта.

        Использует функцию `solve_power` из `statsmodels.stats.power.TTestIndPower`.

        Параметры
        ----------
        effect_size : float
            Ожидаемый размер эффекта (дельта Коэна).
        alternative : str, по умолчанию 'big_b'
            Определяет тип альтернативы:
              - 'big_b'   : альтернативная гипотеза, что B > A;
              - 'small_b' : альтернативная гипотеза, что B < A;
              - 'two_sided': двусторонняя проверка.

        Возвращает
        ----------
        float
            Оцененный необходимый размер выборки (для каждой группы).
        """
        if alternative=='big_b':
            alt = 'larger'
        elif alternative=='small_b':
            alt = 'smaller'
        else:
            alt = 'two-sided'

        analysis = TTestIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size,
                              alpha=alpha,
                              power=power,
                              ratio=1,
                              alternative=alt)
        return sample_size
        
    
