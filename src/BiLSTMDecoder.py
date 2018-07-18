from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Embedding, GRU, LSTM
from keras.optimizers import RMSprop

class BiLSTMDecoder():

    def __init__(self, float_data):
        self.model = Sequential()
        self.model.add(Bidirectional(
            GRU(32), input_shape=(None, float_data.shape[-1])))
        self.model.add(Dense(1))
        self.model.compile(optimizer=RMSprop(), loss='mae')

    def train(self, x_train, y_train):
        history = self.model.fit(x_train, y_train, epochs=10, batch_size=128, 
            validation_split=0.2)
        return history

    def decode(self, source_t, scut_t):
        pass