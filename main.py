import numpy as np
from scipy.optimize import newton


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

    return newton(F, x, tol=eps)


def main():
    print(f"{get_x()}")


if __name__ == "__main__":
    main()
