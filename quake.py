import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "quake.csv"  
data = pd.read_csv(file_path)

X = data.values  

# Normalizar os dados 
def normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

X_normalized = normalize(X)

# Função para calcular a distância euclidiana 
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# Implementação do K-Médias 
def k_means(X, k, max_iters=100):
    np.random.seed(42)
    n_samples, n_features = X.shape

    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        distances = np.zeros((n_samples, k))
        for i, centroid in enumerate(centroids):
            distances[:, i] = euclidean_distance(X, centroid)
        clusters = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    reconstruction_error = np.sum([np.sum((X[clusters == i] - centroids[i]) ** 2) for i in range(k)])
    return centroids, clusters, reconstruction_error

# Função para calcular o índice Davies-Bouldin
def davies_bouldin(X, clusters, centroids):
    k = len(centroids)
    cluster_distances = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            avg_distance = np.mean(np.sqrt(np.sum((cluster_points - centroids[i]) ** 2, axis=1)))
            cluster_distances.append(avg_distance)

    db_index = 0
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                inter_cluster_distance = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                ratio = (cluster_distances[i] + cluster_distances[j]) / inter_cluster_distance
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio

    return db_index / k

# Avaliação para diferentes valores de k
k_values = range(4, 21)
best_db = float('inf')
best_k = None
best_centroids = None
best_clusters = None

for k in k_values:
    for _ in range(20):  
        centroids, clusters, error = k_means(X_normalized, k)
        db_index = davies_bouldin(X_normalized, clusters, centroids)
        if db_index < best_db:
            best_db = db_index
            best_k = k
            best_centroids = centroids
            best_clusters = clusters

# Visualizar o melhor resultado
plt.figure(figsize=(8, 6))
for i in range(best_k):
    cluster_points = X_normalized[best_clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.scatter(best_centroids[:, 0], best_centroids[:, 1], color='red', marker='x', s=100, label='Centroids')
plt.title(f'K-Médias Clustering (Melhor k={best_k}, DB={best_db:.2f})')
plt.xlabel('1ª Coluna (normalizada)')
plt.ylabel('2ª Coluna (normalizada)')
plt.legend()

plt.savefig("imagens/kmeans_clusters.png")
plt.show()

# Exibir os melhores resultados
print("Resultados do Melhor Modelo de K-Médias:")
print(f"Melhor valor de k: {best_k}")
print(f"Menor Índice Davies-Bouldin (DB): {best_db:.4f}")

# Erro de reconstrução do melhor modelo
best_error = np.sum([np.sum((X_normalized[best_clusters == i] - best_centroids[i]) ** 2) for i in range(best_k)])
print(f"Erro de reconstrução para o melhor k: {best_error:.4f}")

# Mostrar os centróides finais
print("\nCentróides finais (coordenadas normalizadas):")
for i, centroid in enumerate(best_centroids):
    print(f"  Centróide {i + 1}: {centroid}")

# Mostrar o número de elementos em cada cluster
print("\nDistribuição dos elementos por cluster:")
for i in range(best_k):
    cluster_size = np.sum(best_clusters == i)
    print(f"  Cluster {i + 1}: {cluster_size} elementos")

plt.figure(figsize=(8, 6))
for i in range(best_k):
    cluster_points = X_normalized[best_clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.scatter(best_centroids[:, 0], best_centroids[:, 1], color='red', marker='x', s=100, label='Centroids')
plt.title(f'K-Médias Clustering (Melhor k={best_k}, DB={best_db:.2f})')
plt.xlabel('1ª Coluna (normalizada)')
plt.ylabel('2ª Coluna (normalizada)')
plt.legend()

output_path = "imagens/kmeans_clusters.png"
plt.savefig(output_path)
plt.show()
print(f"\nVisualização salva em: {output_path}")
