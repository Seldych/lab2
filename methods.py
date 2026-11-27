"""
Реализация семи методов безусловной минимизации функций многих переменных.

Все методы принимают EvalTracker и начальную точку x0, возвращают OptimizeResult.
Градиентные методы поддерживают аналитические и конечно-разностные производные.

Методы:
- steepest_descent: наискорейший спуск
- conjugate_gradients: сопряжённые градиенты (Флетчера-Ривза)
- newton_method: классический метод Ньютона
- regular_simplex: метод правильного симплекса
- cyclic_coordinate_descent: циклический покоординатный спуск
- hooke_jeeves: метод Хука-Дживса
- random_search: алгоритм случайного поиска
"""

import numpy as np
from utils import numerical_grad, numerical_hess, digit_search_1d, OptimizeResult


def steepest_descent(tracker, x0, tol=1e-3, line_search_tol=1e-5, use_analytical=True):
    """Метод наискорейшего спуска."""
    x = x0.copy()
    iters = 0
    trajectory = [tracker.peek(x)]  # добавление траектории сходимости

    while True:
        grad = tracker.g(x) if use_analytical else numerical_grad(tracker.f, x)
        if np.linalg.norm(grad) < tol:
            break

        # Одномерная минимизация поразрядным поиском
        phi = lambda a: tracker.f(x - a * grad)
        alpha = digit_search_1d(phi, h=1.0, tol=line_search_tol)

        x = x - alpha * grad
        trajectory.append(tracker.peek(x))  # добавляем значение функции в траеткорию
        iters += 1
        if iters > 10000:
            break
    return OptimizeResult(x=x, iters=iters, trajectory=trajectory)


def conjugate_gradients(tracker, x0, tol=1e-3, line_search_tol=1e-5, use_analytical=True):
    """Метод сопряженных градиентов (Флетчера-Ривза)."""
    x = x0.copy()
    grad = tracker.g(x) if use_analytical else numerical_grad(tracker.f, x)
    d = -grad
    iters = 0
    trajectory = [tracker.peek(x)]  # добавление траектории сходимости

    while np.linalg.norm(grad) > tol and iters < 10000:
        phi = lambda a: tracker.f(x + a * d)
        alpha = digit_search_1d(phi, h=1.0, tol=line_search_tol)

        x_new = x + alpha * d
        grad_new = tracker.g(x_new) if use_analytical else numerical_grad(tracker.f, x_new)

        beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
        d = -grad_new + beta * d

        x, grad = x_new, grad_new
        trajectory.append(tracker.peek(x))  # добавить новое значние фунции в третекторию
        iters += 1
    return OptimizeResult(x=x, iters=iters, trajectory=trajectory)


def newton_method(tracker, x0, tol=1e-3, use_analytical=True):
    """Классический метод Ньютона."""
    x = x0.copy()
    iters = 0
    trajectory = [tracker.peek(x)]  # добавление траектории сходимости

    while np.linalg.norm(tracker.g(x) if use_analytical else numerical_grad(tracker.f, x)) > tol and iters < 10000:
        grad = tracker.g(x) if use_analytical else numerical_grad(tracker.f, x)
        hess = tracker.h(x) if use_analytical else numerical_hess(tracker.f, x)

        # Решение системы H * step = -grad
        step = np.linalg.solve(hess, -grad)
        x = x + step
        trajectory.append(tracker.peek(x))  # добавить в траекторию
        iters += 1
    return OptimizeResult(x=x, iters=iters, trajectory=trajectory)


def regular_simplex(tracker, x0, tol=1e-3):
    """Метод правильного симплекса (прямой поиск)."""
    n = len(x0)
    # Инициализация правильного симплекса
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    step_size = 1.0
    for i in range(1, n + 1):
        simplex[i] = x0
        simplex[i, i - 1] += step_size

    iters = 0
    trajectory = []  # добавление траектории сходимости

    while True:
        f_vals = np.array([tracker.f(pt) for pt in simplex])
        worst_idx = np.argmax(f_vals)
        best_idx = np.argmin(f_vals)  # носитель информации о прогрессе это лучшая вершина симплекса.
        trajectory.append(f_vals[best_idx])   # ← лучшее значение текущего симплекса
        # Нюанс: значение берётся до отражения/редукции,
        # т.е. траектория отстаёт на одну итерацию относительно остальных методов.
        # Для графика сходимости это несущественно, но при сравнении длин траекторий об этом надо помнить.

        # Центроид без худшей точки
        centroid = np.mean(np.delete(simplex, worst_idx, axis=0), axis=0)

        # Отражение
        x_r = centroid + (centroid - simplex[worst_idx])
        if tracker.f(x_r) < f_vals[best_idx]:
            simplex[worst_idx] = x_r
        else:
            # Редукция (сжатие к лучшей точке)
            for i in range(n + 1):
                if i != best_idx:
                    simplex[i] = simplex[best_idx] + 0.5 * (simplex[i] - simplex[best_idx])

        # Критерий остановки: размер симплекса
        max_dist = np.max([np.linalg.norm(simplex[i] - simplex[best_idx]) for i in range(n + 1)])

        if max_dist < tol:
            break
        iters += 1
        if iters > 10000:
            break
    return OptimizeResult(x=simplex[best_idx], iters=iters, trajectory=trajectory)


def cyclic_coordinate_descent(tracker, x0, tol=1e-3, line_search_tol=1e-5):
    """Метод циклического покоординатного спуска."""
    x = x0.copy()
    n = len(x0)
    iters = 0
    trajectory = [tracker.peek(x)]  # добавление траектории сходимости

    while True:
        x_old = x.copy()
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            phi = lambda a, x=x, e_i=e_i: tracker.f(x + a * e_i)
            alpha = digit_search_1d(phi, h=1.0, tol=line_search_tol)
            x = x + alpha * e_i
        trajectory.append(tracker.peek(x))  # добавить значение функции в траекторию
        if np.linalg.norm(x - x_old) < tol:
            break
        iters += 1
        if iters > 10000:
            break
    return OptimizeResult(x=x, iters=iters, trajectory=trajectory)


def hooke_jeeves(tracker, x0, tol=1e-3, initial_step=1.0, reduction=0.5):
    """Метод Хука-Дживса."""
    x = x0.copy()
    step = initial_step
    iters = 0
    trajectory = [tracker.peek(x)]  # добавление траектории сходимости

    while step > tol:
        x_old = x.copy()
        # Исследующий поиск
        for i in range(len(x)):
            e_i = np.zeros(len(x))
            e_i[i] = 1.0
            if tracker.f(x + step * e_i) < tracker.f(x):
                x = x + step * e_i
            elif tracker.f(x - step * e_i) < tracker.f(x):
                x = x - step * e_i

        # Проверка улучшения
        if tracker.f(x) < tracker.f(x_old):
            # Перемещение по образцу (pattern move)
            x_new = x + (x - x_old)
            x_old = x
            x = x_new
        else:
            step *= reduction
        trajectory.append(tracker.peek(x))  # добавить новое значние фунции в траекторию
        iters += 1
        if iters > 10000:
            break
    return OptimizeResult(x=x, iters=iters, trajectory=trajectory)


def random_search(tracker, x0, tol=1e-3, initial_step=1.0):
    """Алгоритм случайного поиска."""
    x = x0.copy()
    step = initial_step
    iters = 0
    trajectory = [tracker.peek(x)]  # добавление траектории сходимости
    np.random.seed(42)

    while step > tol:
        direction = np.random.randn(len(x0))
        direction /= np.linalg.norm(direction)

        x_new = x + step * direction
        if tracker.f(x_new) < tracker.f(x):
            x = x_new
        else:
            step *= 0.9  # Уменьшение шага при неудаче

        trajectory.append(tracker.peek(x))  # добавить новое значение функции в траеторию
        iters += 1
        if iters > 10000:
            break
    return OptimizeResult(x=x, iters=iters, trajectory=trajectory)
