import numpy as np

# Задача 1:

# Приращение по умолчанию
dx = 0.001

# Производная равна: - sin(x) + 0.15 * x ** 2 + 2 / (x * np.log(2))
def task_1_func(x):
    return np.cos(x) + 0.05 * x ** 3 + np.log2(x ** 2)

# А тут поиск значения по формуле произоводной в точке x: (f(x + dx) / f(x)) / dx
def derivation(x, func):
    return round((func(x + dx) - func(x)) / dx, 2)

# Задача 2:

# x_1 = x[0]; x_2 = x[1]

def task_2_func(x):
    return (x[0] * np.cos(x[1])) + (0.05 * x[1] ** 3) + (3 * x[0] ** 3) * (np.log2(x[1] ** 2))

def gradient(x, func):
    return (round((func([x[0] + dx, x[1]]) - func(x)) / dx, 2), # dx / dx_1
            round((func([x[0], x[1] + dx]) - func(x)) / dx, 2)) # dx / dx_2

# Задача 3:

epsilon = 0.001
iterations_count = 50
initial_value = 10

def gradient_optimization_one_dim(func):
    result = initial_value
    for _ in range(iterations_count):
        result -= epsilon * derivation(result, func)
    return round(result, 2)

# Задача 4:

initial_weights = [4, 10]

def gradient_optimisation_multi_dim(func):
    result = initial_weights
    for _ in range(iterations_count):
        result = [result[i] - grad_i * epsilon for i, grad_i in enumerate(gradient(result, func))]
    return [round(number, 2) for number in result]
