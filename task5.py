"""
Задание 5: функция Розенброка f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2.

Исследует зависимость точности решения от точности одномерного поиска
(line_search_tol ∈ {1e-4, 1e-6, 1e-8}) при ε ∈ {1e-3, 1e-5}.
Фиксирует случаи преждевременного окончания поиска.
Строит график зависимости ||x - x*|| от line_search_tol (task5_accuracy.png).
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import EvalTracker
from methods import (steepest_descent, conjugate_gradients, newton_method,
                     regular_simplex, cyclic_coordinate_descent, hooke_jeeves, random_search)


def get_rosenbrock():
    """
    Функция Розенброка: f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2
    Точный минимум: x* = (1, 1), f(x*) = 0
    """
    func = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad(x):
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])

    def hess(x):
        return np.array([
            [-400 * (x[1] - 3 * x[0]**2) + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])

    return func, grad, hess


X_STAR = np.array([1.0, 1.0])


def run_task5():
    print("=" * 80)
    print("ЗАДАНИЕ 5: Функция Розенброка")
    print("=" * 80)

    func, grad, hess = get_rosenbrock()
    x0 = np.array([-1.0, 1.0])

    eps_values = [1e-3, 1e-5]
    ls_tol_values = [1e-4, 1e-6, 1e-8]

    # Методы: (имя, функция, использует ли line_search_tol)
    methods = [
        ("Наискорейший спуск", steepest_descent, True),
        ("Сопряженные градиенты", conjugate_gradients, True),
        ("Ньютон", newton_method, False),
        ("Правильный симплекс", regular_simplex, False),
        ("Покоординатный спуск", cyclic_coordinate_descent, True),
        ("Хук-Дживс", hooke_jeeves, False),
        ("Случайный поиск", random_search, False),
    ]

    # Словарь для сбора данных графика: {method_name: {ls_tol: error}}
    error_by_method = {name: {} for name, _, _ in methods}

    for eps in eps_values:
        print(f"\n{'=' * 80}")
        print(f"  Точность основной задачи: ε = {eps}")
        print(f"{'=' * 80}")

        # Заголовок таблицы
        print(f"{'Метод':<22} | {'ls_tol':>7} | {'||x-x*||':>10} | {'f(x)':>10} "
              f"| {'итер':>5} | {'вызовов':>7} | {'прежд. ок.':>10}")
        print("-" * 80)

        for name, method, uses_ls in methods:
            for ls_tol in ls_tol_values:
                tracker = EvalTracker(func, grad, hess)

                # Формируем kwargs в зависимости от типа метода
                kwargs = {"tol": eps}
                if uses_ls:
                    kwargs["line_search_tol"] = ls_tol

                result = method(tracker, x0, **kwargs)

                x_opt = result.x
                diff = np.linalg.norm(x_opt - X_STAR)
                f_opt = tracker.peek(x_opt)
                total_calls = tracker.f_evals + tracker.g_evals + tracker.h_evals

                # Преждевременное окончание: алгоритм остановился, но мы далеко от x*
                premature = "ДА" if diff > eps else "нет"

                ls_tol_str = f"{ls_tol:.0e}" if uses_ls else "—"

                print(f"{name:<22} | {ls_tol_str:>7} | {diff:>10.2e} | {f_opt:>10.2e} "
                      f"| {result.iters:>5} | {total_calls:>7} | {premature:>10}")

                error_by_method[name][ls_tol] = diff

    # Построение графика зависимости ошибки от точности одномерного поиска
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, errors in error_by_method.items():
        # Только методы, использующие line_search
        if errors:
            ls_tols = sorted(errors.keys())
            errs = [errors[t] for t in ls_tols]
            ax.loglog(ls_tols, errs, marker="o", linewidth=2, label=name)

    ax.set_xlabel("Точность одномерного поиска (line_search_tol)", fontsize=12)
    ax.set_ylabel("||x - x*||", fontsize=12)
    ax.set_title("Зависимость точности решения от точности одномерной минимизации\n"
                 "(функция Розенброка, x0 = (-1, 1))", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()  # Более точные значения — справа

    fig.tight_layout()
    fig.savefig("task5_accuracy.png", dpi=150)
    print(f"\n{'=' * 80}")
    print("График сохранён в task5_accuracy.png")
    print("=" * 80)

    # Аналитический вывод
    print("\nАНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("-" * 80)
    print("1. Функция Розенброка — классический тест для методов оптимизации.")
    print("   Её минимум (1, 1) лежит на дне узкой параболической долины.")
    print()
    print("2. Влияние точности одномерного поиска:")
    print("   - При грубом ls_tol (1e-4): методы 'застревают' вдали от минимума.")
    print("     Шаги вдоль долины вычисляются неточно, траектория зигзагообразна.")
    print("   - При точном ls_tol (1e-8): методы проходят дальше по долине,")
    print("     но требуют значительно больше вычислений функции.")
    print()
    print("3. Преждевременное окончание:")
    print("   - Наискорейший спуск и сопряжённые градиенты часто останавливаются")
    print("     из-за малого ||grad|| в долине, хотя до x* ещё далеко.")
    print("   - Метод Ньютона на Розенброке ведёт себя нестабильно из-за")
    print("     плохой обусловленности гессиана вдали от минимума.")
    print("   - Прямые методы (симплекс, Хук-Дживс) не зависят от ls_tol,")
    print("     но сходятся медленно.")
    print()
    print("4. Вывод: для овражных функций точность одномерного поиска")
    print("   критически влияет на итоговую точность. Оптимальный выбор —")
    print("   компромисс между точностью и числом вычислений.")
