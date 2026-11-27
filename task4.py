import time

import matplotlib.pyplot as plt
import numpy as np

from methods import (conjugate_gradients, cyclic_coordinate_descent,
                     hooke_jeeves, newton_method, random_search,
                     regular_simplex, steepest_descent)
from utils import EvalTracker

# Та же точность, что и в main.py — для согласованности сравнения
LINE_SEARCH_TOL = 1e-6

LINE_SEARCH_METHODS = {
    "Наискорейший спуск",
    "Сопряженные градиенты",
    "Покоординатный спуск"
}

ANALYTICAL_METHODS = {
    "Наискорейший спуск",
    "Сопряженные градиенты",
    "Ньютон"
}

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


def get_ravine_func(a):
    """Овражная функция f(x) = x1^2 + a*x2^2"""
    func = lambda x: x[0]**2 + a * x[1]**2
    grad = lambda x: np.array([2 * x[0], 2 * a * x[1]])
    hess = lambda x: np.array([[2, 0], [0, 2 * a]])
    return func, grad, hess


def get_test_func_3():
    """Тестовая функция №3"""
    func = lambda x: 254 * x[0]**2 + 506 * x[0] * x[1] + 254 * x[1]**2 + 50 * x[0] - 130 * x[1] + 111
    grad = lambda x: np.array([508 * x[0] + 506 * x[1] + 50, 506 * x[0] + 508 * x[1] - 130])
    hess = lambda x: np.array([[508, 506], [506, 508]])
    return func, grad, hess


def run_task4():
    """Задание 4: сравнительный анализ алгоритмов оптимизации"""
    print("\n" + "=" * 50)
    print("ЗАДАНИЕ 4: Сравнение эффективности методов")
    print("=" * 50)

    # Настройки тестов
    tests = [
        ("Овражная функция (a=250)", get_ravine_func(250), np.array([1.0, 1.0])),
        ("Тестовая функция №3", get_test_func_3(), np.array([0.0, 0.0]))
    ]

    results = {}  # Для хранения траекторий (для графика)

    for test_name, (func, grad, hess), x0 in tests:
        print(f"\n{'=' * 70}")
        print(f"Тест: {test_name}")
        print(f"{'=' * 70}")
        print(f"{'Метод':<25} | {'Итераций':>8} | {'Вызовов f':>9} | {'Время (мс)':>10} | {'f(x*)':>12}")
        print("-" * 70)

        results[test_name] = {}

        for name, method in METHODS.items():
            tracker = EvalTracker(func, grad, hess)

            start_time = time.perf_counter()
            result = method(tracker, x0, **build_kwargs(name, 1e-3))
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            total_calls = tracker.f_evals + tracker.g_evals + tracker.h_evals
            f_opt = tracker.peek(result.x)

            print(f"{name:<25} | {result.iters:>8} | {total_calls:>9} | "
                  f"{elapsed_ms:>10.2f} | {f_opt:>12.2e}")

            results[test_name][name] = result.trajectory

    # График сходимости
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, test_name in enumerate(["Овражная функция (a=250)", "Тестовая функция №3"]):
        ax = axes[idx]
        for name in METHODS.keys():
            if name in results[test_name]:
                ax.semilogy(results[test_name][name], label=name, linewidth=1.5)

        ax.set_xlabel("Итерация")
        ax.set_ylabel("f(x)")
        ax.set_title(test_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("task4_convergence.png", dpi=150)
    print(f"\n{'=' * 70}")
    print("График сходимости сохранён в task4_convergence.png")
    print("=" * 70)

    # Объяснение результатов
    print("\nАНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("-" * 70)
    print("1. Овражная функция (a=250):")
    print("   - Метод Ньютона: минимальное число итераций (квадратичная сходимость)")
    print("   - Градиентные методы: много итераций из-за плохой обусловленности")
    print("   - Сопряженные градиенты: эффективнее наискорейшего спуска")
    print("   - Прямые методы: требуют много вычислений функции")
    print()
    print("2. Тестовая функция №3:")
    print("   - Метод Ньютона: сходится за 1 итерацию (квадратичная функция)")
    print("   - Градиентные методы: быстрая сходимость (хорошая обусловленность)")
    print("   - Прямые методы: работают, но требуют больше вычислений")
    print()
    print("ВЫВОД: Метод Ньютона наиболее эффективен для квадратичных функций,")
    print("но требует вычисления гессиана. Для овражных функций метод")
    print("сопряженных градиентов предпочтительнее наискорейшего спуска.")
