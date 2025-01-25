import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists('imagens'):
    os.makedirs('imagens')

file_path = "penguins.csv"  
data = pd.read_csv(file_path)

numeric_columns = data.select_dtypes(include=[np.number]).columns  
data = data.dropna(subset=numeric_columns)  
X = data[numeric_columns].values  


def normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

X_normalized = normalize(X)

# Função para realizar o PCA
def pca(X, num_components):
    # Centralizar os dados
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Calcular a matriz de covariância
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Obter autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Ordenar por autovalores em ordem decrescente
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Selecionar os principais componentes
    selected_vectors = eigenvectors[:, :num_components]

    # Projetar os dados nos componentes principais
    X_reduced = np.dot(X_centered, selected_vectors)

    return X_reduced, eigenvalues

# Projeção em 2D
X_pca_2d, eigenvalues = pca(X_normalized, num_components=2)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], label="Dados")
plt.title("Projeção em 2D com PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.savefig("imagens/projecao_2d_pca.png")
plt.show()

# Calcular a variação explicada
total_variance = np.sum(eigenvalues)
explained_variance = [np.sum(eigenvalues[:i]) / total_variance for i in range(1, len(eigenvalues) + 1)]

print("Variação explicada por número de componentes:")
for i, var in enumerate(explained_variance, start=1):
    print(f"Dimensão {i}: {var * 100:.2f}% da variação explicada")

# Plotar a variação explicada
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues) + 1), explained_variance, marker='o')
plt.title("Variação Explicada pelo PCA")
plt.xlabel("Número de Componentes Principais")
plt.ylabel("Proporção da Variação Explicada")
plt.grid()
plt.savefig("imagens/variacao_explicada_pca.png")
plt.show()
