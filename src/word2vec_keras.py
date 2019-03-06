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
            horizontal.append((i, tuple(line[2:])))
            max_sample_size += 1.
        else:
            vertical.append((i, tuple(line[2:])))
            max_sample_size += 0.5
        lines[i] = line[2:]

    # Create a dictionary that allows tag -> index lookup
    # Index in this case is corrlelated to how common it is
    # Ex: if "The" appears the most often in the dataset, its index is 1
    # For the keras skipgram function
    count = [['UNK', -1]]
    count.extend(Counter(tags).most_common())
    tag2int = dict()
    for tag, _ in count:
        tag2int[tag] = len(tag2int)

    # Reduce list to unique values
    tags = list(set(tags))
    vocab_size = len(tags)

    # Build sentence generator. Iterates through all horizontal and vertical tags
    def gen_sentence():
        count = 0
        while True:
            slide = lines[count]
            h_count = count + 1 if count < len(lines) - 1 else 0
            slide = [tag2int[tag] for tag in slide]
            yield slide

    return gen_sentence(), tag2int, vocab_size, tags, lines, vertical, horizontal

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

def build_slide_embedding(lines, vertical, horizontal, tag2int, tag_embedding):
    # Lookup the index of the slide embedding matrix and get all photos that make up that slide
    slide2image = {}
    # Lookup an image index relative to line number in the file, and get all slides that the image is part of
    image2slide = {}
    # Lookup slide id and get lower dimension vector
    slide_embedding = []

    print("Starting slide embedding genration...")
    # Horizontals get directly converted
    for index, tags in horizontal:
        slide2image[len(slide_embedding)] = [index]
        image2slide[index] = [len(slide_embedding)]
        slide_vector = np.zeros(args.embedding_dim)
        for tag in tags:
            slide_vector += tag_embedding[tag2int[tag]]
        slide_embedding.append(slide_vector)

    # Building one slide for every single combination of vertical images
    for index, _data in enumerate(vertical):
        index_one, tags_one = _data
        for index_two, tags_two in vertical[index:]:
            # Can't combine image with itself to make a slide
            if index_one == index_two:
                continue
            # Remember what photos made up this slide
            slide2image[len(slide_embedding)] = [index_one, index_two]

            # Remember what slide these photos made up
            # Build a new array if the key doesn't exist, otherwise add to the array
            if index_one in image2slide:
                image2slide[index_one].append(len(slide_embedding))
            else:
                image2slide[index_one] = [len(slide_embedding)]
            if index_two in image2slide:
                image2slide[index_two].append(len(slide_embedding))
            else:
                image2slide[index_two] = [len(slide_embedding)]

            # Build and add slide vector to embedding
            slide_vector = np.zeros(args.embedding_dim)
            for tag in tags_one:
                slide_vector += tag_embedding[tag2int[tag]]
            for tag in tags_two:
                slide_vector += tag_embedding[tag2int[tag]]
            slide_embedding.append(slide_vector)
    
    print("Slide embedding built!")
    return slide_embedding, slide2image, image2slide

def main(args):
    # Gen data
    print("Starting sentence generator generation...")
    sentence_generator, tag2int, vocab_size, tags, lines, vertical, horizontal = parse_file(args.input_file)
    print("Sentence generator built and file parsed!")

    # Make training data
    print("Starting data generator building...")
    data_generator = build_data_generator(sentence_generator, tag2int, vocab_size, args.batch_size)
    print("Data generator built!")

    # Summary writing
    summary_loss = tf.placeholder(dtype=tf.float32, shape=())
    tf.summary.scalar('loss', summary_loss)
    merged_summaries = tf.summary.merge_all()

    # Converting our final image embedding matrix to a tensor to save for tensorboard
    # Called once at the very end of the program
    slide_embedding_size = sum([x for x in range(len(vertical))]) + len(horizontal)
    summary_embedding = tf.placeholder(dtype=tf.float32, shape=(slide_embedding_size, args.embedding_dim))
    images = tf.Variable(summary_embedding, name='images')

    with tf.Session() as sess:
        summary_path = "{}/summary".format(args.log_dir)
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
        saver = tf.train.Saver([images])

        # Build models
        print("Starting model building...")
        if args.restore == None:
            model, validation_model, embedding = build_model(vocab_size, args.embedding_dim)
        else:
            model = load_model(args.restore)
            print("Model restored from {}".format(args.restore))
        print("Models built!")
        
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

        # Build slide embedding and dictionaries
        slide_embedding, slide2image, image2slide = build_slide_embedding(lines, vertical, horizontal, tag2int, tag_embedding)
        
        # Save the slide embedding matrix
        slide_embedding_out = "{}/{}_slide_embedding.npy".format(args.output_dir, args.output_name)
        np.save(file_io.FileIO(slide_embedding_out, 'w'), slide_embedding)

        # Save the lookup dictionaries
        slide2image_out = "{}/{}_slide2image.npy".format(args.output_dir, args.output_name)
        np.save(file_io.FileIO(slide2image_out, 'w'), slide2image)
        image2slide_out = "{}/{}_image2slide.npy".format(args.output_dir, args.output_name)
        np.save(file_io.FileIO(image2slide_out, 'w'), image2slide)

        # Save the slide embeddings for tensorboard
        sess.run(images.initializer, feed_dict={summary_embedding: slide_embedding})
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

