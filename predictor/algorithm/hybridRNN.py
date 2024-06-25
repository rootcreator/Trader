import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Concatenate, Layer


class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='random_normal',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class HybridRNNModel:
    def __init__(self, units=32):
        self.units = units
        self.model = None

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        lstm_branch = Bidirectional(LSTM(self.units, return_sequences=True))(inputs)
        lstm_branch = Bidirectional(LSTM(self.units, return_sequences=True))(lstm_branch)
        lstm_attention = AttentionLayer()(lstm_branch)
        gru_branch = Bidirectional(GRU(self.units, return_sequences=True))(inputs)
        gru_branch = Bidirectional(GRU(self.units, return_sequences=True))(gru_branch)
        gru_attention = AttentionLayer()(gru_branch)
        merged = Concatenate(axis=1)([lstm_attention, gru_attention])
        outputs = Dense(1)(merged)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def train(self, X, y):
        self.build_model(X.shape[1:])
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(X, y, epochs=10)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return tf.keras.losses.mean_squared_error(y, preds)
