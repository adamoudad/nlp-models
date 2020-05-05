from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Embedding

def prediction_lstm(n_classes):
    """
    LSTM which predicts next token
    """
    model = Sequential()
    model.add(Embedding(n_classes, 64))
    model.add(LSTM(
        128,
        input_shape=(None, n_classes),
        return_sequences=True,
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
