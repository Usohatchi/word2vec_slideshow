import tensorflow as tf
import numpy as np

from tensorflow.python.lib.io import file_io
import json
import argparse

import gensim
from gensim.models import word2vec
import logging

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

class Sentence_Generator(object):
    def __init__(self, filename):
        # Read file
        with file_io.FileIO(filename, 'r') as f:
            self.lines = f.readlines()[1:]
            self.lines = [line.strip().split() for line in self.lines]

        self.horizontal = []
        self.vertical = []
        self.tags = []

        self.max_sample_size = 0.

        # Save frames and build list of all tags
        for i, line in enumerate(self.lines):
            self.tags.extend(line[2:])
            if line[0] == 'H':
                self.horizontal.append(tuple(line[2:]))
                self.max_sample_size += 1.
            else:
                self.vertical.append(tuple(line[2:]))
                self.max_sample_size += 0.5

        # Reduce list to unique values
        self.tags = list(set(self.tags))
        self.vocab_size = len(self.tags)

        # Generate tag -> int dictionary
        self.tag2int = {}
        for i,tag in enumerate(self.tags):
            self.tag2int[tag] = i

    def __iter__(self):
        total = len(self.horizontal)+len(self.vertical)
        proportions = [len(self.horizontal) / total, len(self.vertical) / total]
        while True:
            is_horizontal = np.random.choice([True,False],p=proportions)
            is_horizontal = True
            if is_horizontal:
                slide = self.horizontal[np.random.randint(len(self.horizontal))]
                yield slide
            # If vertical, pick two random vertical photos and add their tags together
            ### SHOULDNT MATTER FOR NOW BECUASE WE AREN'T TESTING ON DATASETS WITH VERTICALS FOR RL ###
            else:
                slide_left = self.vertical[np.random.randint(len(self.vertical))]
                slide_right = self.vertical[np.random.randint(len(self.vertical))]
                slide = list(slide_left) + list(slide_right)
                yield slide

def main(args):
    print("Starting sentence generator generation...")
    # Gen data
    sentence_generator = Sentence_Generator(args.input_file)
    print("Sentence generator built and file parsed")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentence_generator, iter=100, min_count=2, size=5, workers=4)
    print(model.wv['cat'])

    '''
    # Save the tag -> int dictionary and the embeddings layer in their own files
    ### WORKS WITH GCP ML ENGINE
    tag_matrix_out = "{}/{}_tag_embedding.npy".format(args.output_dir, args.output_name)
    np.save(file_io.FileIO(tag_matrix_out, 'w'), tag_embedding)
    dict_out = "{}/{}_tag_dict.npy".format(args.output_dir, args.output_name)
    np.save(file_io.FileIO(dict_out, 'w'), tag2int)

    # Build image2vec embedding matrix
    # first line is embedding[0], second is embedding[1], ... embedding[x]
    image_embedding = []
    for index, line in enumerate(lines):
        # Add up all tag vectors to make an image vector
        image_vector = np.zeros(args.embedding_dim)
        for tag in line[2:]:
            image_vector += tag_embedding[tag2int[tag]]
        image_embedding.append(image_vector)
    
    # Save the image2vec embedding matrix
    image_matrix_out = "{}/{}_image_dict.npy".format(args.output_dir, args.output_name)
    np.save(file_io.FileIO(image_matrix_out, 'w'), image_embedding)
    '''

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("word2vec")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out")
    parser.add_argument(
        "--output-name",
        type=str,
        default="")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/a_example.txt")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="log")
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/word2vec')
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10)
    parser.add_argument(
        "--n-epoch",
        type=int,
        default=10000)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=5)
    parser.add_argument(
        "--learning-rate",
        type=int,
        default=0.001)
    args = parser.parse_args()
    main(args)

