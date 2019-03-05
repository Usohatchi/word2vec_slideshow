import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, Reshape, Input, Dense, dot
from tensorflow.keras.models import Model, load_model

import numpy as np
from collections import Counter

from tensorflow.python.lib.io import file_io
import json
import argparse

def parse_file(filename):
    # Read file
    with file_io.FileIO(filename, 'r') as f:
        lines = f.readlines()[1:]
        lines = [line.strip().split() for line in lines]

    horizontal = []
    vertical = []
    tags = []

    max_sample_size = 0.

    # Save frames and build list of all tags
    for i, line in enumerate(lines):
        tags.extend(line[2:])
        if line[0] == 'H':
            horizontal.append(tuple(line[2:]))
            max_sample_size += 1.
        else:
            vertical.append(tuple(line[2:]))
            max_sample_size += 0.5

    # Create a dictionary that allows tag -> index lookup
    # Index in this case is corrlelated to how common it is
    # Ex: if "The" appears the most often in the dataset, its index is 1
    # For the keras skipgram function
    count = [['UNK', -1]]
    count.extend(Counter(tags).most_common())
    tag2int = dict()
    for tag in count:
        tag2int[tag] = len(tag2int)

    # Reduce list to unique values
    tags = list(set(tags))
    vocab_size = len(tags)


    # Build sentence generator. Iterates through all horizontal and vertical tags
    total = len(horizontal) + len(vertical)
    proportions = [len(horizontal) / total, len(vertical) / total]
    h_count = 0
    v_count = 0
    def gen_sentence():
        while True:
            is_horizontal = np.random.choice([True,False],p=proportions)
            if is_horizontal:
                slide = horizontal[h_count]
                h_count = h_count + 1 if h_count < len(horizontal) - 1 else 0
            else:
                slide = horizontal[v_count]
                v_count = v_count + 1 if v_count < len(vertical) - 1 else 0
            slide = [tag2int[tag] for tag in slide]
            yield slide

    return gen_sentence(), int(max_sample_size), tag2int, vocab_size, tags, lines

def build_data_generator(sentence_generator, tag2int, vocab_size, batch_size):
    # Sampling table for keras skipgram function
    # Tells keras what the vocab size is, so it can pick negtive examples without being biased by how common words are
    sampling_table = sequence.make_sampling_table(vocab_size + 1)

    # Build dataset generator for training
    def gen_data():
        while True:
            # Get new sentence
            sentence = next(sentence_generator)
            # Generate target word / context word pairs, and their associated negative or postive labels
            couples, labels = skipgrams(sentence, vocab_size + 1, window_size=15, sampling_table=sampling_table)
            # Weird bug where skipgrams sometimes returns empy couples and labels
            # Might have to do with the fact that we manually process sentences instead of using a tokenizer or something?
            while len(couples) == 0:
                couples, labels = skipgrams(sentence, vocab_size + 1, window_size=15, sampling_table=sampling_table)
            word_target, word_context = zip(*couples)
            # Convert to numpy arrays so our models can use the data
            word_target = np.array(word_target, dtype="int32")
            word_context = np.array(word_context, dtype="int32")
            labels = np.array(labels, dtype="int32")
            yield [word_target, word_context], labels

    return gen_data()

def build_model(vocab_size, embedding_dim):
    #-->Index<-- of input target and input context
    input_target = Input((1,))
    input_context = Input((1,))
    
    # Embedding!
    embedding = Embedding(vocab_size + 1, embedding_dim, input_length=1, name='embedding')

    # Lookup our indexes in our embedding table, then reshape so we can do dot-product
    target = embedding(input_target)
    target = Reshape((embedding_dim, 1))(target)
    context = embedding(input_context)
    context = Reshape((embedding_dim, 1))(context)

    # Cosine similarity - not used for training, but for checking how similar our input and context are
    similarity = dot([target, context], axes=0, normalize=True)
    
    # See how close our target and context vectors are
    dot_product = dot([target, context], axes=1)
    dot_product = Reshape((1,))(dot_product)

    # Sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    
    # Build model!
    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # Second model that gives us distance between input_target and input_context words
    validation_model = Model(inputs=[input_target, input_context], outputs=similarity)

    return model, validation_model, embedding

def main(args):
    # Gen data
    print("Starting sentence generator generation...")
    sentence_generator, total_sentences, tag2int, vocab_size, tags, lines = parse_file(args.input_file)
    print("Sentence generator built and file parsed!")

    # Make training data
    print("Starting data generator building...")
    data_generator = build_data_generator(sentence_generator, tag2int, vocab_size, args.batch_size)
    print("Data generator built!")

    # Build models
    print("Starting model building...")
    if args.restore != None:
        model, validation_model, embedding = build_model(vocab_size, args.embedding_dim)
    else:
        model = load_model(args.restore)
    print("Models built!")

    # Summary writing
    summary_loss = tf.placeholder(dtype=tf.float32, shape=())
    tf.summary.scalar('loss', summary_loss)
    merged_summaries = tf.summary.merge_all()

    # Converting our final image embedding matrix to a tensor to save for tensorboard
    # Called once at the very end of the program
    summary_embedding = tf.placeholder(dtype=tf.float32, shape=(len(lines), args.embedding_dim))
    images = tf.Variable(summary_embedding, name='images')

    with tf.Session() as sess:
        summary_path = "{}/summary".format(args.log_dir)
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
        saver = tf.train.Saver([images])
        
        # Per epoch parse batch_size random sentences and train on the resulting dataset
        for e in range(args.n_epoch):
            features, labels = next(data_generator)
            loss = model.train_on_batch(features, labels)
            print('epoch {}, loss is : {}'.format(e, loss))

            # Save loss and model
            if e % args.checkpoint_period == 0:
                summary = sess.run(merged_summaries, feed_dict={summary_loss: loss})
                summary_writer.add_summary(summary, e)
                save_path = "{}/model_{}.h5".format(args.log_dir, e)
                model.save(save_path)

        # Pull embedding layer weights
        # Embedding layer is the 3rd layer of the model after the 2 inputs
        tag_embedding = model.layers[2].get_weights()[0]
    
        # Save the tag -> int dictionary and the embeddings layer in their own files
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

        sess.run(images.initializer, feed_dict={summary_embedding: image_embedding})
        embedding_path = "{}/image.ckpt".format(args.log_dir)
        saver.save(sess, embedding_path)

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
    parser.add_argument(
        "--restore",
        type=str,
        default=None)
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        default=100)
    args = parser.parse_args()
    main(args)

