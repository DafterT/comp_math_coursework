import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
from prettytable import PrettyTable


def get_x():
    """
    Возвращает результат решения уравнения x=1.4^x
    """
    # Начальное приближение
    x = 2
    eps = 1e-7

    # Функция, у которой ищем ноль
    def F(x):
        return x - (1.4 ** x)

    return scipy.optimize.newton(F, x, tol=eps)


def get_E():
    """
    Возвращает переменную E, взяв интеграл и умножив на константу
    """
    start = 0
    end = 1

    # Функция для интегрирования
    def F(x):
        return np.sin(x) / (x ** 2 + 1)

    return 1.553791 * scipy.integrate.quad(F, start, end, )[0]


def get_Mathieu_function(delt, E):
    """
    Возвращает функцию Метьё
    """

    def F(t, X):
        dX = np.zeros(X.shape)
        dX[1] = X[0]
        dX[0] = -(delt + E * np.cos(2 * t)) * X[1]
        return dX

    return F


def rkf853(f, T, X0):
    """
    Решает `x' = f(t, x)` для каждого `t` в `T`
    С начальным значением `X0`, используя явный метод Рунге-Кутта 8 5 3
    """
    runge = scipy.integrate.ode(f).set_integrator('dop853').set_initial_value(X0, T[0])
    X = [X0, *[runge.integrate(T[i]) for i in range(1, len(T))]]
    return np.array([i[0] for i in X]), np.array([i[1] for i in X])


def calculate(points, error):
    """
    Вычисляет функцию Матьё с заданной погрешностью
    """
    # Вычисляем А на основании x*
    A = 0.5300355 * get_x() + error * random.choice([-1, 1])
    # B задано
    B = 0 + error * random.choice([-1, 1])
    # Дельта задано
    delt = 1 + error * random.choice([-1, 1])
    # Получим E
    E = get_E() + error * random.choice([-1, 1])
    # Получим функцию Матьё
    Mathieu_function = get_Mathieu_function(delt, E)
    # Зададим начальные значения
    X0 = np.array([A, B])
    # Получим результат вычисления
    Mathieu = rkf853(Mathieu_function, points, X0)[0]
    # Вывели результат в виде графика
    plt.title(f'Error = {error}')
    plt.plot(points, Mathieu, '-o')
    plt.savefig(f'graphs/error_{error}.png')
    plt.show()
    return Mathieu


def calculate_without_error(points):
    """
    Вычисляет функцию Матьё
    """
    # Вычисляем А на основании x*
    A = 0.5300355 * get_x()
    # B задано
    B = 0
    # Дельта задано
    delt = 1
    # Получим E
    E = get_E()
    # Получим функцию Матьё
    Mathieu_function = get_Mathieu_function(delt, E)
    # Зададим начальные значения
    X0 = np.array([A, B])
    # Получим результат вычисления
    Mathieu = rkf853(Mathieu_function, points, X0)[0]
    return Mathieu


def main():
    # Получение решения
    points = np.arange(0, 10.5, 0.5)
    without_error = calculate_without_error(points)
    # Получение решения с двойной точностью
    doubled_points = np.arange(0, 10.25, 0.25)
    without_error_doubled_points = calculate_without_error(doubled_points)
    # Вывод на график
    plt.plot(points, without_error, '-o', color='g', label='Step 0.5')
    plt.plot(doubled_points, without_error_doubled_points, '-o', color='b', label='Step 0.25', alpha=0.3)
    plt.legend()
    plt.savefig('graphs/graphs_without_error.png')
    plt.show()
    # Оценка точности
    Runge_rule = abs(without_error - without_error_doubled_points[::2]) / (2 ** 8 - 1)
    plt.plot(points, Runge_rule, '-o')
    plt.savefig('graphs/error_of_method.png')
    plt.show()
    pt = PrettyTable()
    pt.add_column('X', points)
    pt.add_column('Without Error', without_error)
    pt.add_column('Without Error Doubled', without_error_doubled_points[::2])
    pt.add_column('Runge Rule', Runge_rule)
    print(pt)
    print(f'Max error = {max(Runge_rule)}')
    # Оценка влияния погрешности исходных данных
    error = []
    error_add = np.array([10 ** (-i) for i in range(6, 0, -1)])
    for i in error_add:
        error.append(calculate(points, i))
    # Вывод значений
    pt = PrettyTable()
    pt.add_column('X', points)
    pt.add_column('0', [f'{i:.07}' for i in without_error])
    for error_val, result in zip(error_add, error):
        pt.add_column(f'{error_val:.0e}', [f'{i:.07}' for i in result])
    print(pt)
    # Вывод погрешностей
    max_delta = []
    pt = PrettyTable()
    pt.add_column('X', points)
    for error_val, result in zip(error_add, error):
        max_delta.append(max(without_error - result))
        pt.add_column(f'{error_val:.0e}', [f'{i:.05e}' for i in without_error - result])
    print(pt)
    # Вывод максимальных погрешностей
    pt = PrettyTable()
    pt.add_column('Error val', [f'{i:.0e}' for i in error_add])
    pt.add_column('Max Error', [f'{i:.5e}' for i in max_delta])
    print(pt)


if __name__ == "__main__":
    main()
