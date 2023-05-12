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

    return 1.553791 * scipy.integrate.quad(F, start, end)[0]


def main():
    # Вычисляем А на основании x*
    A = 0.5300355 * get_x()
    # B задано
    B = 0
    # Дельта задано
    delt = 1
    # Получим E
    E = get_E()


if __name__ == "__main__":
    main()
