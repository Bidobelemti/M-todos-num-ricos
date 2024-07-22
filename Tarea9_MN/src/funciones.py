import numpy as np
import matplotlib.pyplot as plt
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())


def plot_system(a1, b1, c1, a2, b2, c2):
    # Crear una figura y un eje
    fig, ax = plt.subplots()

    # Crear puntos x para la primera ecuación
    x = np.linspace(-10, 10, 400)
    y1 = (c1 - a1*x) / b1
    y2 = (c2 - a2*x) / b2

    # Trazar las líneas
    ax.plot(x, y1, label=f'{a1}x + {b1}y = {c1}')
    ax.plot(x, y2, label=f'{a2}x + {b2}y = {c2}')

    # Calcular el punto de intersección
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([c1, c2])
    try:
        solution = np.linalg.solve(A, B)
        ax.plot(solution[0], solution[1], 'ro')  # marcar la solución en el gráfico
        plt.title(f'Solución: x = {solution[0]:.2f}, y = {solution[1]:.2f}')
    except np.linalg.LinAlgError:
        plt.title('No hay solución única (las rectas son paralelas o coincidentes)')

    # Configurar los ejes
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    # Agregar leyenda
    ax.legend()

    # Mostrar el gráfico
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def eliminacion_gaussiana(A: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de eliminación gaussiana.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1), donde n es el número de incógnitas.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A)
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    for i in range(0, n - 1):  # loop por columna

        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if A[pi, i] == 0:
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(A[pi, i]) < abs(A[p, i]):
                p = pi

        if p is None:
            # no pivot found.
            raise ValueError("No existe solución única.")

        if p != i:
            # swap rows
            logging.debug(f"Intercambiando filas {i} y {p}")
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

        print(f"\n{A}")
    # --- Sustitución hacia atrás
    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
        solucion[i] = (A[i, n] - suma) / A[i, i]

    return solucion
