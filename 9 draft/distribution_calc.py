import numpy as np
from scipy import stats

class DistributionProbabilityCalculator:
    """
    Класс для вычисления различных статистических характеристик распределений.

    Атрибуты:
    ----------
    dist_name (str): Название распределения.
    distribution: Объект распределения.

    Методы:
    -------
    ppf(q)
        Вычисляет значение переменной, соответствующее заданному процентилю.

    ppf_range(q_central_interval=None, q_top=None, q_bottom=None)
        Вычисляет интервал значений, соответствующий заданным процентилям или проценту центрального интервала.
        
    pmf_cdf_rangepmf(value, value_bottom=None)
        Вычисляет вероятность конкретного значения или вероятность диапазона значений.
        Включает нижнюю границу в рассчеты вероятности. То есть при указании value=5 и value_bottom=3, найдет вероятность появления значений 3, 4, 5.

    mean()
        Возвращает среднее значение распределения.

    std()
        Возвращает стандартное отклонение распределения.

    Примеры использования:
    -------
    Биномиальное распределение.
        DistributionProbabilityCalculator('binom', n=10, p=0.5)
        Параметры: n (количество испытаний Бернулли), p (вероятность успеха)

    Распределение Пуассона.
        DistributionProbabilityCalculator('poisson', mu=3)
        Параметры: mu (среднее число событий за фиксированный интервал времени)

    Геометрическое распределение.
        DistributionProbabilityCalculator('geom', p=0.5)
        Параметры: p (вероятность успеха)

    Гипергеометрическое распределение.
        DistributionProbabilityCalculator('hypergeom', M=20, n=7, N=12)
        Параметры: M (размер генеральной совокупности), n (число успехов в генеральной совокупности), N (размер выборки)

    Негативное биномиальное распределение.
        DistributionProbabilityCalculator('nbinom', n=5, p=0.5)
        Параметры: n (число успехов), p (вероятность успеха)

    Равномерное дискретное распределение.
        DistributionProbabilityCalculator('randint', low=1, high=10)
        Параметры: low (нижняя граница, включительно), high (верхняя граница, исключительно)

    Нормальное распределение.
        DistributionProbabilityCalculator('norm', loc=0, scale=1)
        Параметры: loc (среднее), scale (стандартное отклонение)

    Экспоненциальное распределение.
        DistributionProbabilityCalculator('expon', scale=1)
        Параметры: scale (среднее время между событиями). 1/значение - кол-во событий в единицу времени.
    
    Равномерное непрерывное распределение.
        DistributionProbabilityCalculator('uniform', loc=0, scale=1)
        Параметры: loc (нижняя граница), scale (размах распределения)

    t-распределение Стьюдента.
        DistributionProbabilityCalculator('t', df=10)
        Параметры: df (степени свободы)
        
    Логнормальное распределение.
        DistributionProbabilityCalculator('lognorm', s=0.954, scale=1)
        Параметры: s (стандартное отклонение логарифма), scale (медиана распределения)

    Хи-квадрат распределение.
        DistributionProbabilityCalculator('chi2', df=2)
        Параметры: df (степени свободы)

    F-распределение.
        DistributionProbabilityCalculator('f', dfn=2, dfd=3)
        Параметры: dfn (числитель степени свободы), dfd (знаменатель степени свободы)

    Распределение Вейбулла.
        DistributionProbabilityCalculator('weibull_min', c=1.5, scale=1)
        Параметры: c (форма), scale (растяжение/сжатие)
        
    """
    def __init__(self, dist_name, **kwargs):
        """
        Инициализация класса DistributionProbabilityCalculator.

        Аргументы:
        ----------
            dist_name (str): Название распределения.
            **kwargs: Параметры распределения.
        """
        self.dist_name = dist_name
        self.distribution = getattr(stats, dist_name)(**kwargs)
        self.is_discrete = self._check_if_discrete(dist_name)

    def _check_if_discrete(self, dist_name):
        """
        Проверка, является ли распределение дискретным.
        
        Аргументы:
        ----------
        dist_name (str): Название распределения.
        
        Возвращает:
        ----------
        True, если распределение дискретное, иначе False.
        
        """
        return dist_name in ['binom', 'poisson', 'geom', 'hypergeom', 'nbinom', 'bernoulli', 'randint', 'zipf','multinomial','dlaplace','logser','zipfian']

    def ppf(self, q):
        """
        Вычисляет значение переменной, соответствующее заданному процентилю.
        Например, для процентиля 0.1 выдаст значение, ниже которого лежит 10% данных распределения.

        Аргументы:
        ----------
            q (float): Процентиль (от 0 до 1).

        Возвращает:
        ----------
            float: Значение переменной, соответствующее заданному процентилю.

        Пример:
        ----------
            calc = DistributionProbabilityCalculator('norm', loc=0, scale=1)
            value = calc.ppf(0.95)
            print(f"Значение, соответствующее 95-му процентилю: {value}")
        """
        return self.distribution.ppf(q)

    def ppf_range(self, q_central_interval=None, q_top=None, q_bottom=None):
        """
        Вычисляет интервал значений, соответствующий заданным процентилям или проценту центрального интервала.

        Аргументы:
        ----------
            q_central_interval (float, optional): Центральный интервал (например, 0.9 для 90%-го центрального интервала).
            q_top (float, optional): Верхний процентиль.
            q_bottom (float, optional): Нижний процентиль.

        Возвращает:
        ----------
            tuple: Кортеж, содержащий верхнюю и нижнюю границы интервала.

        Пример:
        ----------
            calc = DistributionProbabilityCalculator('norm', loc=0, scale=1)
            central_interval = calc.ppf_range(q_central_interval=0.9)
            print(f"90% центральный интервал: {central_interval}")

            top_bottom_interval = calc.ppf_range(q_top=0.95, q_bottom=0.05)
            print(f"Интервал между 5-ым и 95-ым процентилями: {top_bottom_interval}")
            
        """
        if q_top==None:
            q_top = 1 - (1 - q_central_interval)/2
            q_bottom = (1 - q_central_interval)/2
            return (self.distribution.ppf(q_bottom), self.distribution.ppf(q_top))
        else:
            return (self.distribution.ppf(q_bottom), self.distribution.ppf(q_top))
        

    def pmf_cdf_rangepmf(self, value, value_bottom=None):
        """
        Вычисляет вероятность конкретного значения или вероятность диапазона значений.
        *Включает нижнюю границу в рассчеты вероятности. То есть при указании value=5 и value_bottom=3, найдет вероятность появления значений 3, 4, 5.

        При не указанном value_bottom для дискретных рассчитывает вероятность конкретного значения value, а для непрерывных рассчитывает cdf до этого значения.
        При указанном - рассчитывает диапазон значений (все границы включены в диапазон).
        

        Аргументы:
        ----------
            value (int, float): Значение/Верхняя граница диапазона значений.
            value_bottom (int, float, optional): Нижняя граница диапазона значений.

        Возвращает:
        ----------
            float: Вероятность.

        Пример:
        ----------
            calc = DistributionProbabilityCalculator('binom', n=10, p=0.5)
            calc.pmf_cdf_rangepmf(value=5) # равно 0.2460

            calc.pmf_cdf_rangepmf(value=5, value_bottom=3) # равно 0.5683
            

        """

        if self.is_discrete:
            if value_bottom is None:
                return self.distribution.cdf(value) - self.distribution.cdf(value - 1)
            else:
                return self.distribution.cdf(value) - self.distribution.cdf(value_bottom - 1)
                
        else:
            if value_bottom is None:
                return self.distribution.cdf(value)
            else:
                return self.distribution.cdf(value) - self.distribution.cdf(value_bottom)
    
                
    def mean(self):
        """
        Рассчитывает среднее значение распределения.

        Возвращает:
        ----------
            float: Среднее значение распределения.

        Пример:
        ----------
            calc = DistributionProbabilityCalculator('binom', n=10, p=0.5)
            mean_value = calc.mean() # равно 5
        """
        return self.distribution.mean()

    def std(self):
        """
        Рассчитывает стандартное отклонение распределения.

        Возвращает:
        ----------
            float: Стандартное отклонение распределения.

        Пример:
        ----------
            calc = DistributionProbabilityCalculator('binom', n=10, p=0.5)
            calc.std() # равно 1.58
        """
        return self.distribution.std()