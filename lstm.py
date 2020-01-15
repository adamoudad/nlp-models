from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

def prediction_lstm(sequence_length, n_classes):
    """
    LSTM which predicts next token
    """
    model = Sequential()
    model.add(LSTM(
        128,
        input_shape=(sequence_length, n_classes),
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
