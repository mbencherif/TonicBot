import keras
import numpy as np

def vectorize_sequences(maxlen, step):
    sentences = []
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    
    print('Number of sequences:', len(sentences))
    chars = sorted(list(set(text)))

    print('Unique characters:', len(chars))
    char_indices = dict((char, chars.index(char)) for char in chars)

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

if __name__ == '__main__':
    path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read().lower()
    print('Corpus length:', len(text))

    vectorize_sequences(60, 3)