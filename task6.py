
"""
Задание 6: функция Химмельблау (многоэкстремальная).

Исследует зависимость работы методов от начального приближения на функции
    f(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
с 4 локальными минимумами (все с f = 0):
    (3.0, 2.0), (-2.805118, 3.131312),
    (-3.779310, -3.283186), (3.584428, -1.848126).

Для двух начальных точек x0 = (0, 0) и x0 = (-5, 0):
- прогоняет 7 методов минимизации,
- определяет, в какой минимум попал каждый метод (через identify_minimum),
- обрабатывает вырождение гессиана в методе Ньютона (LinAlgError).

Генерирует файл task6_himmelblau.png — две контурные карты рельефа
log10(f(x)) с наложением:
- красных звёздочек (известные минимумы),
- зелёного круга (x0),
- цветных маркеров P (точки сходимости методов).

Экспортирует: run_task6().
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from methods import (conjugate_gradients, cyclic_coordinate_descent,
                     hooke_jeeves, newton_method, random_search,
                     regular_simplex, steepest_descent)
from utils import KNOWN_MINIMA_HIMMELBLAU, EvalTracker, identify_minimum

# Те же настройки, что и в main.py / task4.py — для согласованности
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

METHODS = {
    "Наискорейший спуск": steepest_descent,
    "Сопряженные градиенты": conjugate_gradients,
    "Ньютон": newton_method,
    "Правильный симплекс": regular_simplex,
    "Покоординатный спуск": cyclic_coordinate_descent,
    "Хук-Дживс": hooke_jeeves,
    "Случайный поиск": random_search
}

# Цвета и стили для траекторий методов
METHOD_STYLES = {
    "Наискорейший спуск": ("#1f77b4", "-o"),
    "Сопряженные градиенты": ("#ff7f0e", "-s"),
    "Ньютон": ("#2ca02c", "-^"),
    "Правильный симплекс": ("#d62728", "-D"),
    "Покоординатный спуск": ("#9467bd", "-v"),
    "Хук-Дживс": ("#8c564b", "-<"),
    "Случайный поиск": ("#e377c2", "->"),
}


def build_kwargs(name, tol):
    """Формирует kwargs для вызова метода с учётом его сигнатуры."""
    kwargs = {"tol": tol}
    if name in LINE_SEARCH_METHODS:
        kwargs["line_search_tol"] = LINE_SEARCH_TOL
    if name in ANALYTICAL_METHODS:
        kwargs["use_analytical"] = True
    return kwargs


def get_himmelblau():
    """
    Функция Химмельблау:
        f(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
    Имеет 4 локальных минимума с f = 0:
        (3.0, 2.0), (-2.805118, 3.131312),
        (-3.779310, -3.283186), (3.584428, -1.848126)
    """
    def func(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def grad(x):
        u = x[0]**2 + x[1] - 11
        v = x[0] + x[1]**2 - 7
        return np.array([
            4 * x[0] * u + 2 * v,
            2 * u + 4 * x[1] * v
        ])

    def hess(x):
        return np.array([
            [12 * x[0]**2 + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]],
            [4 * x[0] + 4 * x[1], 4 * x[0] + 12 * x[1]**2 - 26]
        ])

    return func, grad, hess


def run_task6():
    print("=" * 80)
    print("ЗАДАНИЕ 6: Функция Химмельблау (многоэкстремальная)")
    print("=" * 80)

    func, grad, hess = get_himmelblau()
    start_points = [
        np.array([0.0, 0.0]),
        np.array([-5.0, 0.0])
    ]

    # {x0_str: {method_name: (min_idx, x_opt, iters, trajectory)}}
    all_results = {}

    for x0 in start_points:
        x0_key = f"({x0[0]:.0f}, {x0[1]:.0f})"
        print(f"\n{'=' * 80}")
        print(f"  Начальная точка: x0 = {x0_key}")
        print(f"{'=' * 80}")
        print(f"{'Метод':<22} | {'Минимум':>7} | {'x*':>22} | "
              f"{'f(x*)':>8} | {'итер':>5} | {'вызовов':>7}")
        print("-" * 80)

        all_results[x0_key] = {}

        for name, method in METHODS.items():
            tracker = EvalTracker(func, grad, hess)

            try:
                result = method(tracker, x0, **build_kwargs(name, 1e-5))
            except np.linalg.LinAlgError:
                # Гессиан мог быть вырожден (актуально для Ньютона)
                print(f"{name:<22} | {'ВЫРОЖД.':>7} | {'—':>22} | "
                      f"{'—':>8} | {'—':>5} | {'—':>7}")
                continue

            min_idx = identify_minimum(result.x, KNOWN_MINIMA_HIMMELBLAU, tol=1e-2)
            min_label = f"№{min_idx}" if min_idx >= 0 else "—"
            total_calls = tracker.f_evals + tracker.g_evals + tracker.h_evals
            f_opt = tracker.peek(result.x)

            print(f"{name:<22} | {min_label:>7} | "
                  f"[{result.x[0]:>7.3f}, {result.x[1]:>7.3f}] | "
                  f"{f_opt:>8.2e} | {result.iters:>5} | {total_calls:>7}")

            all_results[x0_key][name] = {
                "min_idx": min_idx,
                "x_opt": result.x,
                "trajectory": result.trajectory,
                # "x_trajectory": _extract_x_trajectory(method, tracker, x0, name)
            }

    # График: контуры + траектории методов для двух начальных точек
    _plot_trajectories(all_results)

    # Аналитический вывод
    _print_analysis(all_results)


# def _extract_x_trajectory(method, tracker, x0, name):
#     """
#     Повторный прогон метода для сбора траектории по x (не по f).
#     Нужно только для визуализации — не влияет на счётчики основного прогона.
#     """
#     # Чтобы не усложнять методы, пересобираем траекторию по x через monkey-patch.
#     # Проще: запустить метод второй раз с тем же seed (для random_search).
#     np.random.seed(42) if name == "Случайный поиск" else None
#     result = method(tracker, x0, **build_kwargs(name, 1e-5))
#     # У методов нет траектории по x — восстанавливать её слишком затратно.
#     # Вместо этого используем только факт попадания в минимум.
#     return []


def _plot_trajectories(all_results):
    """Строит контурный plot функции с наложением точек-результатов."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Сетка для контурного plot
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

    # Ограничим диапазон значений для наглядности (лог-шкала по Z)
    Z_plot = np.log10(Z + 1e-6)

    start_points = list(all_results.keys())

    for idx, x0_key in enumerate(start_points):
        ax = axes[idx]
        cf = ax.contourf(X, Y, Z_plot, levels=50, cmap="viridis", alpha=0.7)
        ax.contour(X, Y, Z_plot, levels=15, colors="white", linewidths=0.5, alpha=0.5)
        fig.colorbar(cf, ax=ax, label="log10(f(x)+ε)")

        # Отметить известные минимумы
        for i, m in enumerate(KNOWN_MINIMA_HIMMELBLAU):
            ax.plot(m[0], m[1], "r*", markersize=12, label=f"минимум №{i}" if i == 0 else "")

        # Отметить начальную точку
        x0 = np.array([float(c) for c in x0_key.strip("()").split(", ")])
        ax.plot(x0[0], x0[1], "go", markersize=10, label="x0")

        # Отметить точки, в которые попали методы
        for name, data in all_results[x0_key].items():
            if data["min_idx"] >= 0:
                color, _ = METHOD_STYLES[name]
                ax.plot(data["x_opt"][0], data["x_opt"][1],
                        marker="P", color=color, markersize=9,
                        markeredgecolor="black", markeredgewidth=0.8)

        ax.set_title(f"Начальная точка x0 = {x0_key}", fontsize=13)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.legend(loc="upper right", fontsize=8)

    # Общая легенда методов
    handles = []
    for name, (color, _) in METHOD_STYLES.items():
        handles.append(Line2D([0], [0], color="none", marker="P",
                              markerfacecolor=color, markeredgecolor="black",
                              markersize=9, label=name))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, title="Методы (маркер — точка сходимости)",
               title_fontsize=11)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("task6_himmelblau.png", dpi=150)
    print(f"\n{'=' * 80}")
    print("График сохранён в task6_himmelblau.png")
    print("=" * 80)


def _print_analysis(all_results):
    """Аналитический вывод по результатам."""
    print("\nАНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("-" * 80)
    print("1. Функция Химмельблау имеет 4 локальных минимума с одинаковым")
    print("   значением f = 0. Все они — глобальные, но лежат в разных")
    print("   «чашах» рельефа функции.")
    print()
    print("2. Зависимость от начального приближения:")

    for x0_key, methods_data in all_results.items():
        print(f"\n   x0 = {x0_key}:")
        for name, data in methods_data.items():
            min_label = f"минимум №{data['min_idx']}" if data["min_idx"] >= 0 else "не попал в известный"
            print(f"     - {name:<22}: {min_label}")

    print()
    print("3. Особенности градиентных методов на многоэкстремальных функциях:")
    print("   - Все градиентные методы (наискорейший спуск, сопряжённые")
    print("     градиенты, Ньютон) детерминированно сходятся к одному и тому")
    print("     же минимуму для фиксированной начальной точки.")
    print("   - Смена x0 может привести к попаданию в другой минимум —")
    print("     это фундаментальное свойство локальных методов.")
    print("   - Метод Ньютона может давать сбой, если гессиан вырожден")
    print("     в седловой точке (для Химмельблау таких точек несколько).")
    print()
    print("4. Прямые методы (симплекс, Хук-Дживс) и случайный поиск:")
    print("   - Также детерминированы (кроме случайного поиска), но из-за")
    print("     другой геометрии шагов могут попадать в другой минимум")
    print("     при том же x0, что и градиентные методы.")
    print("   - Случайный поиск с фиксированным seed даёт одинаковый")
    print("     результат между запусками, но при смене seed результат")
    print("     может меняться.")
    print()
    print("ВЫВОД: для многоэкстремальных функций локальные методы не")
    print("гарантируют нахождение глобального минимума. Результат сильно")
    print("зависит от начального приближения. Для надёжного поиска")
    print("необходимо либо запускать метод из нескольких начальных точек,")
    print("либо использовать глобальные методы (эволюционные, роевые и т.п.).")
