import tensorflow as tf
import numpy as np

from tensorflow.python.lib.io import file_io
import json
import argparse

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

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

    # Reduce list to unique values
    tags = list(set(tags))
    vocab_size = len(tags)

    # Generate tag -> int dictionary
    tag2int = {}
    for i,tag in enumerate(tags):
        tag2int[tag] = i

    # Build sentence generator. Returns all tags associated with a random image in the dataset
    total = len(horizontal)+len(vertical)
    proportions = [len(horizontal) / total, len(vertical) / total]
    def gen_next():
        while True:
            is_horizontal = np.random.choice([True,False],p=proportions)
            is_horizontal = True
            if is_horizontal or len(vertical) <= 1:
                slide = horizontal[np.random.randint(len(horizontal))]
                yield slide
            # If vertical, pick two random vertical photos and add their tags together
            ### SHOULDNT MATTER FOR NOW BECUASE WE AREN'T TESTING ON DATASETS WITH VERTICALS FOR RL ###
            else:
                slide_left = vertical[np.random.randint(len(vertical))]
                slide_right = vertical[np.random.randint(len(vertical))]
                slide = list(slide_left) + list(slide_right)
                yield slide

    return gen_next(), int(max_sample_size), tag2int, vocab_size, tags, lines

def build_data_generator(sentence_generator, tag2int, vocab_size, batch_size):

    # Build dataset generator for training
    def gen_data():
        while True:
            features = [] # input word
            targets = [] # output word
            
            # Process batch_size sentences
            for _ in range(batch_size):
                sentence = next(sentence_generator)
                # For each word, look though the entire sentence and create input word / context word pairs
                for word_index, word in enumerate(sentence):
                    for nb_word in sentence:
                        if nb_word != word:
                            # One hot encode values before adding to array
                            features.append(to_one_hot(tag2int[ word ], vocab_size))
                            targets.append(to_one_hot(tag2int[ nb_word ], vocab_size))
            yield features, targets

    return gen_data()

def main(args):
    print("Starting sentence generator generation...")
    # Gen data
    sentence_generator, total_sentences, tag2int, vocab_size, tags, lines = parse_file(args.input_file)
    print("Sentence generator built and file parsed")

    # Make training data
    data_generator = build_data_generator(sentence_generator, tag2int, vocab_size, args.batch_size)
    print("Data generator build")

    with tf.Graph().as_default() as g:
        # Input and label placeholders
        x = tf.placeholder(tf.float32, shape=(None, vocab_size))
        y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

        # EMBEDDING LAYER!
        W1 = tf.Variable(tf.random_normal([vocab_size, args.embedding_dim]))
        b1 = tf.Variable(tf.random_normal([args.embedding_dim]))

        hidden_representation = tf.add(tf.matmul(x,W1), b1)

        W2 = tf.Variable(tf.random_normal([args.embedding_dim, vocab_size]))
        b2 = tf.Variable(tf.random_normal([vocab_size]))

        # Final prediction
        prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

        # Loss function
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

        # Training function
        train_step = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cross_entropy_loss)

        init = tf.global_variables_initializer()

        # Saving loss
        tf.summary.scalar('loss', cross_entropy_loss)
        merged_summaries = tf.summary.merge_all()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=args.n_epoch // 1000)
        summary_path = "{}/summary".format(args.log_dir)
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
        
        # Per epoch parse batch_size random sentences and train on the resulting dataset
        for e in range(args.n_epoch):
            features, labels = next(data_generator)
            _, loss = sess.run([train_step, cross_entropy_loss], feed_dict={x: features, y_label: labels})
            print('epoch {}, loss is : {}'.format(e, loss))
            if e % 100 == 0:
                summary = sess.run(merged_summaries, feed_dict={x: features, y_label: labels})
                summary_writer.add_summary(summary, e)
            if e % 1000 == 0:
                save_path = "{}/model.ckpt".format(args.log_dir)
                save_path = saver.save(sess, save_path, global_step=e)

        # Pull the embeddings layer
        tag_embedding = sess.run(W1 + b1)
        
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

