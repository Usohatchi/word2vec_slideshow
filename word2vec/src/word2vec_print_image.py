import tensorflow as tf
import argparse
from tensorflow.python.lib.io import file_io
from io import BytesIO

import numpy as np

from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser("word2vec")
    parser.add_argument(
        "--image-matrix",
        type=str,
        default="out/a_example_image_matrix.npy")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/a_example.txt")
    args = parser.parse_args()

    # Load embedding matrix for images
    ### WORKS WITH GCP
    embedding_matrix = BytesIO(file_io.read_file_to_string(args.image_matrix, binary_mode=True))
    embedding_matrix = np.load(embedding_matrix)

    with file_io.FileIO(args.input_file, 'r') as f:
        lines = f.readlines()[1:]
        lines = [line.strip().split() for line in lines]


    print(embedding_matrix)

    # Draw graph of embeddings in 2 dimensions
    # I dunno how this works 
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    embedding_matrix = model.fit_transform(embedding_matrix)

    normalizer = preprocessing.Normalizer()
    #embedding_matrix =  normalizer.fit_transform(embedding_matrix, 'l2')

    fig, ax = plt.subplots()
    for index, line in enumerate(lines):
        ax.annotate("image #{}".format(index), (embedding_matrix[index][0], embedding_matrix[index][1]))
    plt.show()




