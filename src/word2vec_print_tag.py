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
        "--input-dir",
        type=str,
        default="out")
    parser.add_argument(
        "--file-name",
        type=str,
        default="a_example")
    args = parser.parse_args()

    # Load embedding matrix and tags dict from file
    ### WORKS WITH GCP
    matrix_name = "{}/{}_matrix.npy".format(args.input_dir, args.file_name)
    embedding_matrix = BytesIO(file_io.read_file_to_string(matrix_name, binary_mode=True))
    embedding_matrix = np.load(embedding_matrix)

    dict_name = "{}/{}_dict.npy".format(args.input_dir, args.file_name)
    embedding_dict = BytesIO(file_io.read_file_to_string(dict_name, binary_mode=True))
    embedding_dict = np.load(embedding_dict).item()

    print(embedding_matrix)
    print(embedding_dict)

    # Draw graph of embeddings in 2 dimensions
    # I dunno how this works 
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    embedding_matrix = model.fit_transform(embedding_matrix)

    normalizer = preprocessing.Normalizer()
    #embedding_matrix =  normalizer.fit_transform(embedding_matrix, 'l2')

    fig, ax = plt.subplots()
    for word in embedding_dict:
        print(word, embedding_matrix[embedding_dict[word]][1])
        ax.annotate(word, (embedding_matrix[embedding_dict[word]][0],embedding_matrix[embedding_dict[word]][1] ))
    plt.show()




