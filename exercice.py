#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (math.sqrt(x**2 + y**2), math.atan2(y, x))
            for x, y in cartesian_coordinates
        ]
    )


def find_closest_index(values: np.ndarray, number: float) -> int:
    diff = np.abs(values - number)
    return int(diff.argmin())


def math_func(x):
    return (x**2) * np.sin(1 / (x**2)) + x


def plot_func():
    x = np.linspace(-1, 1, 250)
    y = math_func(x)

    plt.plot(x, y)

    plt.show()


def monte_carlo_pi(sample_count):
    # inspire du corrige (on ne cree pas directement le ndarray)
    points = np.random.random_sample((sample_count, 2))

    inner_points = []
    outer_points = []

    for point in points:
        if (point[0] ** 2 + point[1] ** 2) < 1:
            # inner point
            inner_points.append([point[0], point[1]])
        else:
            outer_points.append([point[0], point[1]])

    inner_array = np.array(inner_points).transpose()
    outer_array = np.array(outer_points).transpose()

    fig = plt.figure()
    ax = fig.add_subplot()

    plt.scatter(
        inner_array[0],
        inner_array[1],
    )

    plt.scatter(
        outer_array[0],
        outer_array[1],
        c="r",
    )

    ax.set_aspect("equal", adjustable="box")

    print(
        f"Ratio: {len(inner_points) / sample_count} ~= pi/3; pi ~= {4 * len(inner_points) / sample_count}"
    )

    plt.show()


def integral_function(x):
    return math.exp(-(x**2))


def compute_integral(start, end):
    return integrate.quad(integral_function, start, end)


if __name__ == "__main__":
    # plot_func()
    # monte_carlo_pi(10000)
    value, err = compute_integral(-np.inf, np.inf)
    print(f"la valeur de l'integrale est: {value:.4f} +/- {err:.4g}")

    x_vals = np.linspace(-4, 4, 100)
    y_vals = np.zeros(100)
    for i in range(100):
        y_vals[i] = compute_integral(0, x_vals[i])[0]

    plt.plot(
        x_vals,
        y_vals,
    )

    plt.show()
