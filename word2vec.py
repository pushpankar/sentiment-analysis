from data import get_data
import random
import numpy as np

data, sentiments, dictionary, reverse_dict = get_data()

"""def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size), dtype=np.int32)
    span = 2 * skip_window + 1

    buff = data[data_index]

    for i in range(batch_size // num_skips):
        surround_word = skip_window
        used_words = [skip_window]
        # Generate n surround_word for 1 center_words
        for j in range(num_skips):
            while surround_word in used_words:
                surround_word = random.randint(0, span-1)
            used_words.append(surround_word)
            batch[i * num_skips + j] = buff[skip_window]
            labels[i * num_skips + j] = buff[surround_word]
"""


def generate_batch(batch_size, num_skips, skip_window):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size), dtype=np.int32)

    random_rev_pos = random.randint(0, len(data))
    buff = data[random_rev_pos]

    print("review : ", [reverse_dict[w] for w in buff])

    for i in range(batch_size // num_skips):
        center_word_index = random.randint(skip_window,
                                           len(buff) - skip_window)
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


batch, label = generate_batch(10, 2, 4)
print("batch: ", [reverse_dict[w] for w in batch])
print("label: ", [reverse_dict[w] for w in label])
print(batch)
