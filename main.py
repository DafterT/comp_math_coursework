import matplotlib.pyplot as plt
import numpy as np
import scipy


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


def main():
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
    # Создадим точки для подсчета
    points = np.arange(0, 10.5, 0.5)
    # Получим результат вычисления
    Mathieu = rkf853(Mathieu_function, points, X0)[0]
    plt.plot(points, Mathieu)
    plt.show()
    print(Mathieu)


if __name__ == "__main__":
    main()
