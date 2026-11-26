import numpy as np
from utils import EvalTracker
from methods import (steepest_descent, conjugate_gradients, newton_method,
                     regular_simplex, cyclic_coordinate_descent, hooke_jeeves, random_search)

# --- Задание 2: Овражная функция ---


def get_ravine_funcs(a):
    func = lambda x: x[0]**2 + a * x[1]**2
    grad = lambda x: np.array([2 * x[0], 2 * a * x[1]])
    hess = lambda x: np.array([[2, 0], [0, 2 * a]])
    return func, grad, hess


def run_task2():
    print("=" * 50)
    print("ЗАДАНИЕ 2: Овражная функция f(x) = x1^2 + a*x2^2")
    print("=" * 50)
    a_values = [1, 250, 1000]
    epsilons = [1e-3, 1e-5]
    x0 = np.array([1.0, 1.0])

    methods = {
        "Наискорейший спуск": steepest_descent,
        "Сопряженные градиенты": conjugate_gradients,
        "Ньютон": newton_method,
        "Правильный симплекс": regular_simplex,
        "Покоординатный спуск": cyclic_coordinate_descent,
        "Хук-Дживс": hooke_jeeves,
        "Случайный поиск": random_search
    }

    for a in a_values:
        for eps in epsilons:
            print(f"\n[ a={a}, eps={eps} ]")
            func, grad, hess = get_ravine_funcs(a)

            for name, method in methods.items():
                tracker = EvalTracker(func, grad, hess)
                # Прямые методы не используют аналитические градиенты
                use_anal = name not in ["Правильный симплекс", "Покоординатный спуск", "Хук-Дживс", "Случайный поиск"]

                x_opt, iters = method(tracker, x0, tol=eps, use_analytical=use_anal) if use_anal else method(
                    tracker, x0, tol=eps)

                total_calls = tracker.f_evals + tracker.g_evals + tracker.h_evals
                print(f"{name:25} | Итераций: {iters:4} | Вызовов (f+g+h): {total_calls:6} | f(x*): {tracker.f(x_opt):.2e}")

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

    methods = {
        "Наискорейший спуск": steepest_descent,
        "Сопряженные градиенты": conjugate_gradients,
        "Ньютон": newton_method
    }

    for name, method in methods.items():
        tracker.reset()
        x_opt, iters = method(tracker, x0, tol=1e-3)
        error = np.linalg.norm(x_opt - x_exact)
        print(f"{name:25} | x*: {x_opt} | Ошибка: {error:.2e} | Итераций: {iters} | Вызовов: {tracker.f_evals + tracker.g_evals + tracker.h_evals}")


if __name__ == "__main__":
    run_task2()
    run_task3()
