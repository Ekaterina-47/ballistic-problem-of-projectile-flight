import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline


# Уравнение поверхности
def surface_equation(x):
    return -0.1 * x


# Исходные параметры
alpha = 1      # коэффициент сопротивления воздуха
theta = 13     # угол броска
V0 = 40        # начальная скорость
m = 10         # масса снаряда
g = 9.8        # ускорение свободного падения


# Начальные условия
x1_0 = 0                           # начальная координата поверхности земли по OX
x2_0 = surface_equation(x1_0)      # начальная координата поверхности земли по OY
x3_0 = V0 * np.cos(np.radians(theta))    # начальные компоненты скорости
x4_0 = V0 * np.sin(np.radians(theta))

x0 = [x1_0, x2_0, x3_0, x4_0]    # вектор начальных условий

h = 0.1

# Генерация временных точек
t = np.arange(0, 6, h)


# Система уравнений движения
def projective_equations(x, t, theta, m, g):
    x1, x2, x3, x4 = x
    dxdt = np.array([x3, x4, -(alpha / m) * x3, -g - (alpha / m) * x4])
    return dxdt


# Метод Рунге-Кутты
def runge_kutta(f, x0, t, h, surface_equation, args=()):
    n = len(t)
    x_positions = np.zeros((n, len(x0)))
    x_positions[0] = x0
    x_interval = []
    time_interval = []

    for i in range(n - 1):
        k1 = f(x_positions[i], t[i], *args)
        k2 = f(x_positions[i] + k1 * h / 2, t[i] + h / 2, *args)
        k3 = f(x_positions[i] + k2 * h / 2, t[i] + h / 2, *args)
        k4 = f(x_positions[i] + k3 * h, t[i] + h, *args)
        x_positions[i + 1] = x_positions[i] + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6

        # Проверка столкновения
        if (x_positions[i][1] - surface_equation(x_positions[i][0])) * (
                x_positions[i + 1][1] - surface_equation(x_positions[i + 1][0])) < 0:
            x_interval.extend([x_positions[i][1], x_positions[i + 1][1]])
            time_interval.extend([t[i], t[i + 1]])
            break

    return t[:i + 2], x_positions[:i + 2], time_interval


# Выполнение метода Рунге-Кутты
result_t, result_x, time_interval = runge_kutta(projective_equations, x0, t, h, surface_equation,
                                                args=(theta, m, g))

# Вывод промежуточных результатов:
print("Результат метода Рунге-Кутты")
print("Интервал столкновения по координате x1: {}".format([result_x[-2][0], result_x[-1][0]]))
print("Интервал столкновения по координате x2: {}".format([result_x[-2][1], result_x[-1][1]]))
print("Временной интервал столкновения: {}".format(time_interval))

# Интерполяция Эрмита для найденного интервала

# Начальные и конечные точки
p0 = np.array([result_x[-2][0], result_x[-2][1]])  # x1_0 и x2_0
p1 = np.array([result_x[-1][0], result_x[-1][1]])  # x1_1 и x2_2

# производные
d0 = np.array([result_x[-2][2], result_x[-2][3]])
d1 = np.array([result_x[-1][2], result_x[-1][3]])

# создание объекта
hermite_spline = CubicHermiteSpline([0, 1], np.array([p0, p1]), np.array([d0, d1]))

print("\nРезультат после применения интерполяции Эрмита и метода секущих: ")


# Применение метода секущих
def secant_method(f, x0, x1, epsilon=1e-6, max_iterations=100):
    for iteration in range(max_iterations):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

        # Проверка на сходимость
        if abs(f(x2)) < epsilon:
            return x2

        # Обновление переменных для следующей итерации
        x0, x1 = x1, x2

    # В случае, если не достигнута сходимость
    print("Метод секущих не сошелся после максимального числа итераций.")
    return None


# Определение уравнения для поиска времени столкновения по вертикальной координате (x2)
surface_equation_y = lambda t: hermite_spline(t)[1] - surface_equation(hermite_spline(t)[0])

# Применение метода секущих
collision_time = secant_method(surface_equation_y, 0, 1)

# Проверка на успешность сходимости
if collision_time is not None:

    t_collision = time_interval[0] + (time_interval[1] - time_interval[0]) * collision_time



    # Вывод точки столкновения
    collision_point = np.array([hermite_spline(collision_time)[0], surface_equation(hermite_spline(collision_time)[0])])
    print("Точка столкновения:", collision_point)
    collision_time = time_interval[0] + collision_time
    print("Время столкновения:", t_collision)
else:
    print("Метод секущих не сошелся.")

# Построение графика
plt.figure(figsize=(10, 8))

# Траектория движения снаряда
plt.plot(result_x[:, 0], result_x[:, 1], label='Траектория движения снаряда', marker='o')

# Линия уровня поверхности земли
x_surface = np.linspace(0, max(result_x[:, 0]) + 10, 1000)
plt.plot(x_surface, surface_equation(x_surface), label='Линия уровня поверхности земли', linestyle='-', color='green')

# Интервал пересечения
plt.scatter([result_x[-2][0], result_x[-1][0]], [result_x[-2][1], result_x[-1][1]], color='red',
            label='Интервал пересечения', s=100)

# Точка столкновения
plt.scatter([collision_point[0]], [collision_point[1]], color='blue', label='Точка столкновения', marker='x', s=200)

# Добавление информации о времени на график
for i, txt in enumerate(result_t):
    plt.annotate(f'{txt:.2f}', (result_x[i, 0], result_x[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center',
                 fontsize=8)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel('X1', fontsize=14)
plt.ylabel('X2', fontsize=14)
plt.title('Точка столкновения: {}, время: {}'.format(collision_point, t_collision), fontsize=13)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()



angels = [angel for angel in range(0, 90)]
x1_mass = []
for angel in angels:
    x1_0 = 0
    x2_0 = surface_equation(x1_0)
    x3_0 = V0 * np.cos(np.radians(angel))
    x4_0 = V0 * np.sin(np.radians(angel))
    x0 = [x1_0, x2_0, x3_0, x4_0]
    dxdt = projective_equations(x0, t, angel, m, g)
    res_t, res_x, t_interval = runge_kutta(projective_equations, x0, t, h, surface_equation,
                                                    args=(angel, m, g))
    x1_mass.append(res_x[-1][0])

print(x1_mass)


# Построение графика зависимости
plt.figure(figsize=(10, 8))
plt.plot(angels, x1_mass, marker='*')
plt.title('Зависимость расстояния (X1) от угла', fontsize=13)
plt.xlabel('Угол', fontsize=14)
plt.ylabel('X1', fontsize=14)
plt.show()
