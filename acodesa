import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AntColonyOptimizer:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        best_cost = float('inf')
        all_paths = []

        for i in range(self.n_iterations):
            paths = self.generate_paths()
            all_paths.extend(paths)
            self.spread_pheromone(paths, self.best_paths(paths))

            best_path_cost, best_path = min((self.path_cost(path), path) for path in paths)
            if best_path_cost < best_cost:
                best_cost = best_path_cost
                shortest_path = best_path

            self.pheromone *= self.decay

        return shortest_path, best_cost, all_paths

    def generate_paths(self):
        paths = []
        for _ in range(self.n_ants):
            path = [np.random.randint(len(self.distances))]
            while len(path) < len(self.distances):
                move = self.probabilistic_next_move(path)
                path.append(move)
            paths.append(path)
        return paths

    def probabilistic_next_move(self, current_path):
        last_node = current_path[-1]
        weights = np.copy(self.pheromone[last_node])
        weights[current_path] = 0  # Hindari kunjungan ulang

        mask = self.distances[last_node] > 0
        weights[mask] /= self.distances[last_node][mask] ** self.beta

        if np.sum(weights) == 0:
            return np.random.choice(self.all_inds)

        normalized_weights = weights / np.sum(weights)
        next_node = np.random.choice(self.all_inds, p=normalized_weights)
        return next_node

    def spread_pheromone(self, paths, best_paths):
        for path_cost, path in best_paths:
            for move in range(len(path) - 1):
                self.pheromone[path[move], path[move + 1]] += 1.0 / path_cost

    def best_paths(self, paths):
        return sorted([(self.path_cost(path), path) for path in paths])[:self.n_best]

    def path_cost(self, path):
        return sum([self.distances[path[i], path[i + 1]] for i in range(len(path) - 1)])

# **Membaca Dataset CSV**
df = pd.read_csv("desa_baru.csv", index_col=0)

# Konversi dataset ke numpy array
distances = df.to_numpy()
desa_list = df.index.tolist()

print("Daftar Desa:", desa_list)
print("Matriks Jarak:\n", distances)

# **Menjalankan Algoritma ACO**
aco = AntColonyOptimizer(distances, n_ants=10, n_best=3, n_iterations=100, decay=0.95, alpha=1, beta=2)
shortest_path, cost, all_paths = aco.run()

print("Jalur Terpendek:", [desa_list[i] for i in shortest_path])
print("Total Kilometer:", cost)

# **Visualisasi Jalur**
fig, ax = plt.subplots(figsize=(8, 8))
x = np.random.rand(len(desa_list))
y = np.random.rand(len(desa_list))

# Semua jalur yang diuji
for path in all_paths:
    for i in range(len(path) - 1):
        ax.plot([x[path[i]], x[path[i+1]]], [y[path[i]], y[path[i+1]]], 'grey', alpha=0.5, linewidth=1)

for i, (xi, yi) in enumerate(zip(x, y)):
    ax.scatter(xi, yi, color='blue')
    ax.text(xi, yi, desa_list[i], color='black')

ax.set_title("Semua Jalur yang Dicoba")
plt.show()

# **Visualisasi Jalur Terbaik**
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(len(shortest_path) - 1):
    ax.plot([x[shortest_path[i]], x[shortest_path[i+1]]], [y[shortest_path[i]], y[shortest_path[i+1]]], 'b-', linewidth=2)

for i, (xi, yi) in enumerate(zip(x, y)):
    ax.scatter(xi, yi, color='red')
    ax.text(xi, yi, desa_list[i], color='black')

ax.set_title(f"Jalur Terpendek (Total Kilometer: {cost})")
plt.show()
