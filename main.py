"""
Точка входа лабораторной работы. Запускает задания 2 и 3, а также делегирует
выполнение заданий 4 и 5 модулям task4 и task5.

Задание 2: тестирование на овражной функции f(x) = x1^2 + a*x2^2
           при a ∈ {1, 250, 1000}, ε ∈ {1e-3, 1e-5}.
Задание 3: минимизация тестовой функции №3 с вычислением ошибки
           относительно точного аналитического решения.
"""

import sys
import numpy as np
from utils import EvalTracker, Tee
from methods import (conjugate_gradients, cyclic_coordinate_descent,
                     hooke_jeeves, newton_method, random_search,
                     regular_simplex, steepest_descent)
from task4 import run_task4
from task5 import run_task5
from task6 import run_task6

# Единая точность одномерного поиска для заданий 2, 3, 4
LINE_SEARCH_TOL = 1e-6

# Методы, использующие одномерный поиск поразрядным методом
LINE_SEARCH_METHODS = {
    "Наискорейший спуск",
    "Сопряженные градиенты",
    "Покоординатный спуск"
}

# Методы, принимающие параметр use_analytical (градиентные первого и второго порядка)
ANALYTICAL_METHODS = {
    "Наискорейший спуск",
    "Сопряженные градиенты",
    "Ньютон"
}

# Прямые методы — не используют аналитические производные
DIRECT_METHODS = {
    "Правильный симплекс",
    "Покоординатный спуск",
    "Хук-Дживс",
    "Случайный поиск"
}

METHODS = {
    "Наискорейший спуск": steepest_descent,
    "Сопряженные градиенты": conjugate_gradients,
    "Ньютон": newton_method,
    "Правильный симплекс": regular_simplex,
    "Покоординатный спуск": cyclic_coordinate_descent,
    "Хук-Дживс": hooke_jeeves,
    "Случайный поиск": random_search
}


def build_kwargs(name, tol):
    """Формирует kwargs для вызова метода с учётом его сигнатуры."""
    kwargs = {"tol": tol}
    if name in LINE_SEARCH_METHODS:
        kwargs["line_search_tol"] = LINE_SEARCH_TOL
    if name in ANALYTICAL_METHODS:
        kwargs["use_analytical"] = True
    return kwargs


# --- Задание 2: Овражная функция ---
def get_ravine_funcs(a):
    func = lambda x: x[0]**2 + a * x[1]**2
    grad = lambda x: np.array([2 * x[0], 2 * a * x[1]])
    hess = lambda x: np.array([[2, 0], [0, 2 * a]])
    return func, grad, hess


def run_task2():
    print()
    print("\n" + "=" * 50)
    print("ЗАДАНИЕ 2: Овражная функция f(x) = x1^2 + a*x2^2")
    print("=" * 50)
    a_values = [1, 250, 1000]
    epsilons = [1e-3, 1e-5]
    x0 = np.array([1.0, 1.0])

    for a in a_values:
        for eps in epsilons:
            print(f"\n[ a={a}, eps={eps} ]")
            func, grad, hess = get_ravine_funcs(a)

            for name, method in METHODS.items():
                tracker = EvalTracker(func, grad, hess)
                result = method(tracker, x0, **build_kwargs(name, eps))

                total_calls = tracker.f_evals + tracker.g_evals + tracker.h_evals
                print(f"{name:25} | Итераций: {result.iters:6} | "
                      f"Вызовов (f+g+h): {total_calls:6} | "
                      f"f(x*): {tracker.peek(result.x):.2e}")


# --- Задание 3: Тестовая функция №3 ---
def get_test_func_3():
    # f(x) = 254*x1^2 + 506*x1*x2 + 254*x2^2 + 50*x1 - 130*x2 + 111
    func = lambda x: 254 * x[0]**2 + 506 * x[0] * x[1] + 254 * x[1]**2 + 50 * x[0] - 130 * x[1] + 111
    grad = lambda x: np.array([508 * x[0] + 506 * x[1] + 50, 506 * x[0] + 508 * x[1] - 130])
    hess = lambda x: np.array([[508, 506], [506, 508]])
    return func, grad, hess


def run_task3():
    print("\n" + "=" * 50)
    print("ЗАДАНИЕ 3: Тестовая функция №3")
    print("=" * 50)
    func, grad, hess = get_test_func_3()
    tracker = EvalTracker(func, grad, hess)
    x0 = np.array([0.0, 0.0])

    # Точное решение (аналитически: grad = 0 => x1 ≈ 0.88, x2 ≈ -1.14)
    x_exact = np.linalg.solve(np.array([[508, 506], [506, 508]]), np.array([-50, 130]))

    for name, method in METHODS.items():
        tracker = EvalTracker(func, grad, hess)
        result = method(tracker, x0, **build_kwargs(name, 1e-3))

        error = np.linalg.norm(result.x - x_exact)
        total_calls = tracker.f_evals + tracker.g_evals + tracker.h_evals
        print(f"{name:25} | x*: {result.x} | Ошибка: {error:.2e} | "
              f"Итераций: {result.iters:6} | Вызовов: {total_calls:6}")


"""Параметр **«Ошибка»** в выводе задания №3 — это **евклидово расстояние** (L2-норма разности) между численно найденной точкой минимума (`result.x`) и точным аналитическим решением (`x_exact`).

### Как это вычисляется
$$ \text{Ошибка} = \| x_{\text{num}} - x_{\text{exact}} \|_2 = \sqrt{(x_{1,\text{num}} - x_{1,\text{exact}})^2 + (x_{2,\text{num}} - x_{2,\text{exact}})^2} $$

### Зачем это нужно
Тестовая функция №3 является квадратичной. Для квадратичных функций точный минимум можно найти аналитически, решив систему линейных уравнений $\nabla f(x) = 0$. 

Этот параметр показывает **реальную точность** алгоритма, независимо от заданного критерия остановки $\varepsilon$. 

### Как интерпретировать значения
- `1.00e-04` — алгоритм остановился на расстоянии $0.0001$ от истинного минимума (хорошая точность).
- `1.00e-15` — машинная точность (алгоритм попал точно в минимум, характерно для метода Ньютона на квадратичных функциях за 1 итерацию).
- `> 1.00e-02` — алгоритм остановился преждевременно или сошелся плохо (возможно, $\varepsilon$ слишком велико или одномерный поиск был слишком грубым).

Это стандартная метрика валадации численных методов оптимизации, когда известно точное решение."""


if __name__ == "__main__":
    # Открываем файл для записи результатов
    with open("lab2_results.txt", "w", encoding="utf-8") as f:
        # Создаём Tee-объект: пишет и в stdout, и в файл
        tee = Tee(sys.stdout, f)

        # Сохраняем оригинальный stdout
        original_stdout = sys.stdout

        try:
            # Перенаправляем stdout на Tee
            sys.stdout = tee

            # Запускаем все задания
            run_task2()
            run_task3()
            run_task4()
            run_task5()
            run_task6()

        finally:
            # Восстанавливаем оригинальный stdout
            sys.stdout = original_stdout

    print("\n" + "=" * 70)
    print("Результаты сохранены в файл: lab2_results.txt")
    print("=" * 70)
