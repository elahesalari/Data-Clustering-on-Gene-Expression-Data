import numpy as np
import pandas as pd

import keras

from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras import layers, Model
from sklearn.mixture import GaussianMixture


def read_dataset():
    dataset = pd.read_csv('Spellman.csv', header=None)
    data = np.array(dataset.iloc[1:, 1:])

    for n in [3, 5]:
        Clustring(dataset, data, n)


def Clustring(dataset, data, n):
    input_size = data.shape[1]
    latent_size = n
    output_size = data.shape[1]

    # Construct AutoEncoder
    input_layer = layers.Input(shape=(input_size,))

    # Encoder
    hidden_layer = layers.Dense(latent_size, activation='sigmoid')(input_layer)

    # Decode
    output_layer = layers.Dense(output_size, activation='tanh')(hidden_layer)

    # Train AutoEncoder
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(data, data, epochs=20)

    # Feature Extraction
    encoder = Model(input_layer, hidden_layer)
    encoder_output = encoder(data)

    x = encoder_output

    k_mean_results = {}
    gmm_results = {}
    for k in [3, 4]:
        # K Means
        k_means = KMeans(n_clusters=k)
        k_means.fit(x)
        y_pred = k_means.predict(x)

        # Davies–Bouldin index (DBI)
        DBI_kmean = davies_bouldin_score(x, y_pred)
        k_mean_results.update({k: (DBI_kmean, y_pred)})

        # GMM
        gmm = GaussianMixture(n_components=k)
        gmm.fit(x)
        label = gmm.predict(x)

        DBI_GMM = davies_bouldin_score(x, label)
        gmm_results.update({k: (DBI_GMM, label)})

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        fig.suptitle(f'Data Scattering for {n} Neuron')
        ax1.set_title(f'K Maen - k={k}')
        ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c=y_pred, s=10, cmap='viridis')
        centers = k_means.cluster_centers_
        ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=20, alpha=1)

        ax2.set_title(f'GMM - k={k}')
        ax2.scatter(x[:, 0], x[:, 1], x[:, 2], c=label, s=20, cmap='viridis')

        plt.show()

    print('--------------------######--------------------')
    print(f'K Mean DBI for {n} neuron in latent space:')
    print(f'\t k=3 :', k_mean_results[3][0])
    print(f'\t k=4 :', k_mean_results[4][0])
    print(f'GMM DBI for {n} neuron in latent space:')
    print(f'\t k=3 :', gmm_results[3][0])
    print(f'\t k=4 :', gmm_results[4][0])

    print('--------------------*****--------------------')

    best_cluster = np.min([k_mean_results[3][0], k_mean_results[4][0], gmm_results[3][0], gmm_results[4][0]])

    print('Best cluster value:', best_cluster)

    if best_cluster == k_mean_results[3][0]:
        predict = k_mean_results[3][1]
    elif best_cluster == k_mean_results[4][0]:
        predict = k_mean_results[4][1]
    elif best_cluster == gmm_results[3][0]:
        predict = gmm_results[3][1]
    elif best_cluster == gmm_results[4][0]:
        predict = gmm_results[4][1]

    gene_ontology(dataset, predict, n)


def gene_ontology(dataset, label, n):
    dataset = np.array(dataset.iloc[1:, 0]).reshape(-1, 1)
    unique = np.unique(label, return_counts=True)
    n_clst = len(unique[0])
    count_cluster = unique[1]
    label = label.reshape(-1, 1)
    print('Number of k cluster:', n_clst)
    print('Count of genes in each cluster:', count_cluster)

    new_data = np.hstack((dataset, label))

    gene_names = {}
    for i in range(n_clst):
        g_cls = {i: []}
        for j in range(dataset.shape[0]):
            if i == new_data[j, 1]:
                g_cls[i].append(new_data[j, 0])
        gene_names.update(g_cls)

    print(gene_names)

    with open(f'Gene_names {n}.txt', 'w') as f:
        for key, value in gene_names.items():
            seperate = ', '.join(value)
            name = str(seperate).strip('[]').replace(',', '')
            f.write('%s:%s\n' % (key, name))


if __name__ == '__main__':
    read_dataset()
