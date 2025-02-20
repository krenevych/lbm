import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Параметри симуляції
Nx, Ny = 100, 100  # Розмір сітки
rho0 = 1.0  # Початкова густина
tau = 0.6  # Час релаксації (в'язкість)
c_s = 1 / np.sqrt(3)  # Швидкість звуку в решітці

# Визначення дискретних напрямків для D2Q9
v = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
weights = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)  # Вагові коефіцієнти

# Ініціалізація функції розподілу
f = np.ones((9, Nx, Ny)) * rho0 / 9

# Додамо початкову швидкість для руху потоку
ux = np.zeros((Nx, Ny))
uy = np.zeros((Nx, Ny))
ux[:, Nx // 2] = 0.5  # Початковий потік у центрі


# Граничні умови (періодичні)
def streaming(f):
    f_new = np.copy(f)
    for i, vi in enumerate(v):
        f_new[i] = np.roll(np.roll(f[i], vi[0], axis=0), vi[1], axis=1)
    return f_new


# Ініціалізація фігури для анімації
fig, ax = plt.subplots()
cmap = ax.imshow(np.zeros((Nx, Ny)), cmap='jet', animated=True)
plt.colorbar(cmap, label='Швидкість потоку')


# Функція оновлення для анімації
def update(frame):
    global f, ux, uy
    # Обчислення макроскопічних параметрів
    rho = np.sum(f, axis=0)
    ux_new = np.sum(f * v[:, 0].reshape(-1, 1, 1), axis=0) / rho
    uy_new = np.sum(f * v[:, 1].reshape(-1, 1, 1), axis=0) / rho

    # Збереження попередніх швидкостей
    ux = 0.9 * ux + 0.1 * ux_new
    uy = 0.9 * uy + 0.1 * uy_new

    # Рівноважна функція розподілу
    feq = np.zeros_like(f)
    for i, vi in enumerate(v):
        eu = vi[0] * ux + vi[1] * uy
        feq[i] = weights[i] * rho * (1 + 3 * eu + 9 / 2 * eu ** 2 - 3 / 2 * (ux ** 2 + uy ** 2))

    # Зіткнення (релаксація BGK)
    f = f + (feq - f) / tau

    # Стримування
    f = streaming(f)

    # Оновлення графіка
    cmap.set_array(np.sqrt(ux ** 2 + uy ** 2).T)
    return cmap,


# Запуск анімації
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.title('Анімація потоку у квадратному домені')
plt.show()