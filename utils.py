from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class OptimizeResult:
    """Результат работы метода оптимизации."""
    x: np.ndarray              # Найденная точка минимума
    iters: int                 # Число итераций
    trajectory: List[float]    # Траектория значений f(x_k)
    converged: bool = True     # Флаг сходимости (для задания 5)

    def __str__(self):
        return f"OptimizeResult(x={self.x}, iters={self.iters}, converged={self.converged})"


class EvalTracker:
    """Класс для подсчета вычислений функции, градиента и гессиана (требование п.1)."""

    def __init__(self, func, grad=None, hess=None):
        self.func = func
        self.grad = grad
        self.hess = hess
        self.f_evals = 0
        self.g_evals = 0
        self.h_evals = 0

    def f(self, x):
        self.f_evals += 1
        return self.func(x)

    def g(self, x):
        self.g_evals += 1
        if self.grad is None:
            raise ValueError("Аналитический градиент не задан")
        return self.grad(x)

    def h(self, x):
        self.h_evals += 1
        if self.hess is None:
            raise ValueError("Аналитический гессиан не задан")
        return self.hess(x)

    def reset(self):
        self.f_evals = self.g_evals = self.h_evals = 0

    def peek(self, x):
        """Значение f без инкремента счётчика — только для траекторий и вывода."""
        return self.func(x)


def numerical_grad(f, x, h=1e-5):
    """Конечно-разностная аппроксимация градиента (центральные разности)."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


def numerical_hess(f, x, h=1e-4):
    """Конечно-разностная аппроксимация гессиана."""
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pp[i] += h
            x_pp[j] += h
            x_pm = x.copy()
            x_pm[i] += h
            x_pm[j] -= h
            x_mp = x.copy()
            x_mp[i] -= h
            x_mp[j] += h
            x_mm = x.copy()
            x_mm[i] -= h
            x_mm[j] -= h
            val = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
            hess[i, j] = hess[j, i] = val
    return hess


def digit_search_1d(phi, alpha0=0.0, h=1.0, tol=1e-5):
    """
    Метод поразрядного поиска для одномерной минимизации (требование п.1).
    phi: функция одной переменной (шаг вдоль направления).
    """
    alpha = alpha0
    while h > tol:
        # Движение в положительном направлении
        while phi(alpha + h) < phi(alpha):
            alpha += h
        # Движение в отрицательном направлении
        while phi(alpha - h) < phi(alpha):
            alpha -= h
        h /= 2.0  # Уменьшение шага (разрядность)
    return alpha
