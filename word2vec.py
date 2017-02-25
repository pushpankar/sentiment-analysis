from data import get_data
import tensorflow as tf
import random
import numpy as np
import math

data, sentiments, dictionary, reverse_dict = get_data()


def generate_batch(batch_size, num_skips, skip_window):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size), dtype=np.int32)

    random_rev_pos = random.randint(0, len(data) - 1)
    buff = data[random_rev_pos]

    for i in range(batch_size // num_skips):
        center_word_index = random.randint(skip_window,
                                           len(buff) - skip_window - 1)
        surround_word_index = center_word_index
        used_words_index = [surround_word_index]
        for j in range(num_skips):
            while surround_word_index in used_words_index:
                surround_word_index = random.randint(
                    center_word_index - skip_window,
                    center_word_index + skip_window)
            used_words_index.append(surround_word_index)
            batch[i * num_skips + j] = buff[center_word_index]
            labels[i * num_skips + j] = buff[surround_word_index]
    return batch, labels


# Build skip-gram model
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
vocabulary_size = len(dictionary)

valid_size = 16
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # number of negative samples

graph = tf.Graph()
with graph.as_default():
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables
    embedding = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    nce_weigths = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model
    embed = tf.nn.embedding_lookup(embedding, train_dataset)
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weigths, nce_biases, embed,
                                         train_labels, num_sampled,
                                         vocabulary_size))
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # compute similarity between minibatch and all examples
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embedding,
                                              valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embedding))
    saver = tf.train.Saver({"embedding": embedding})

# Train model
num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, num_skips,
                                                  skip_window)
        feed_dict = {train_dataset: batch_data,
                     train_labels: batch_labels.reshape((batch_size, 1))
                     }
        _, l = session.run([optimizer, loss],
                           feed_dict=feed_dict)
        average_loss += l

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step {} is {}'.format(step, average_loss))
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dict[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dict[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embedding.eval()
    path = saver.save(session, "data/embedding.ckpt")
    print("model saved in file {}".format(path))
