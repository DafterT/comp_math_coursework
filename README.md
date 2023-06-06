# Курсовая работа по Вычислительной Математике
## Задача
Вариант К-3-21. Уравнение Матьё  
Дано уравнение Матьё вида:  
<img src="https://latex.codecogs.com/svg.image?&space;\begin{cases}&space;&&space;\frac{\mathrm{d}^2U&space;}{\mathrm{d}&space;t^2}&space;&plus;&space;(\delta&space;&plus;&space;E\cdot&space;cos(2t))U=0\\&space;&&space;U(0)=A&space;\\&space;&&space;U'(0)=B\end{cases}" />  
Где коэффициенты имеют следующие значения:  
* <img src="https://latex.codecogs.com/svg.image?A=0.5300355\cdot&space;x^{\circ}" />, где  <img src="https://latex.codecogs.com/svg.image?x^{\circ}" /> наименьший корень уравнения <img src="https://latex.codecogs.com/svg.image?x=1.4^x" />
* <img src="https://latex.codecogs.com/svg.image?B=0" />
* <img src="https://latex.codecogs.com/svg.image?\delta&space;=1" />
* <img src="https://latex.codecogs.com/svg.image?E=1.553791\int_{0}^{1}\tfrac{sin(x))}{x^2&plus;1}dx" />
Необходимо построить график U(t), оценить погрешность результата и влияние на точность погрешности исходных данных.  
___
## Инструменты для решения
В качестве инструмента для решения поставленной задачи был выбран язык Python т.к. в нем есть все необходимые инструменты для работы, а именно:  
1. _**NumPy**_ – для большей скорости расчетов и простоты обработки
2. **_SciPy_** – для функций расчета значения интеграла, поиска минимума функции, решения
системы дифференциальных уравнений.
3. **_PrettyTable_** – для красивого вывода таблицы в консоль
4. **_MatPlotLib_** – для вывода графиков
___
## Аналитические выкладки
Первым делом необходимо привести дифференциальное уравнение 2 порядка к системе первого порядка.  
Все математические выкладки можно найти в отчете.  
Создадим функцию для получения этой системы:  
```Python
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
```
___
## Поиск коэффициента А
Найдем начальное приближение. используя сторонние программы и используя метод Ньютона получим искомую точку с хорошим приближением:
```Python
def get_x():
    """
    Возвращает результат решения уравнения x=1.4^x
    """
    # Начальное приближение
    x = 2
    eps = 1e-9

    # Функция, у которой ищем ноль
    def F(x):
        return x - (1.4 ** x)

    return scipy.optimize.newton(F, x, tol=eps)
```
___
## Поиск коэффициента Е
Для поиска коэффициента Е необходимо решить интеграл, однако у него нет решения в стандартных функциях. Будем искать его приближенно:
```Python
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
```
___
## Решение системы дифференциальных уравнений
Для решения системы можно было воспользоваться аналогом RKF45 в Python, однако существует более точный метод, использующий метод Рунге-Кутты 8 степени:
```Python
def rkf853(f, T, X0):
    """
    Решает `x' = f(t, x)` для каждого `t` в `T`
    С начальным значением `X0`, используя явный метод Рунге-Кутта 8
    """
    runge = scipy.integrate.ode(f).set_integrator('dop853').set_initial_value(X0, T[0])
    X = [X0, *[runge.integrate(T[i]) for i in range(1, len(T))]]
    return np.array([i[0] for i in X]), np.array([i[1] for i in X])
```
## Подсчет точности
Для подсчета точности будем использовать Метод Рунге. Для него необходимо найти 2 решения с разным шагом, создадим для этого функцию, которая будет принимать на вход точки, в которых необходимо искать решение:
```Python
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
```
Выведем полученные результаты в виде графиков и таблиц:
```Python
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
```
Полученные результаты можно найти в отчете.
## Оценка устойчивости
Для оценки устойчивости будем случайным образом в N-й знак входных данных добавлять единицу и смотреть что получится.  
Для этого изменим функцию подсчета, а так же сразу же в неё перенесем вывод графиков:
```Python
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
```
После чего соберем все данные и выведем в виде таблиц для удобства анализа:
```Python
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
```
Полученные результаты так же можно найти в отчете.
# Содержимое файлов
* **_main.py_** - весть код проекта
* _**graphs**_ - все графики проекта
* **_report_** - отчет в формате docx и pdf
* **_report presentation_** - презентация к отчету в формате pptx и pdf