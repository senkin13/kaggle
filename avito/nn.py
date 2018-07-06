import keras
from keras.layers import *
from keras.layers.merge import Dot
from keras.layers.core import Permute, Reshape, SpatialDropout1D, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.engine import Layer
from keras import optimizers, initializers
from keras.models import Input, Model, Sequential
from keras.engine.topology import Layer, InputSpec


def get_bigru_cnn_model(numerical_features, categorical_features, all_columns, maxvalue_dict,
                        maxlen, num_words, embedding_dims, embedding_matrix=None, seed=777):
    dropout_rate = 0.2
    initializer = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=seed)

    # numerical
    numerical = Input(shape=(len(numerical_features),), name='numerical_features')
    numerical_inputs = [numerical]
    n = numerical
    n = BatchNormalization()(n)
    n = Dense(64, kernel_initializer=initializer)(n)
    n = LeakyReLU(alpha=5.5)(n)
    
    # categorical
    embedding_size = 8
    categorical_embeds, categorical_inputs = [], []
    for column in categorical_features:
        inp = Input(shape=(1,), name=column)
        categorical_inputs.append(inp)
        emb = Embedding(maxvalue_dict[column], embedding_size, embeddings_initializer=initializer)(inp)
        categorical_embeds.append(emb)
    c = concatenate(categorical_embeds)
    c = Flatten()(c)
    c = Dropout(dropout_rate)(c)
    
    other_inputs = []

    # text
    text = Input(shape=(maxlen, ), name="seq_text")
    other_inputs.append(text)
    if embedding_matrix is not None:
        t = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(text)
    else:
        t = Embedding(num_words, embedding_dims)(text)
    t = SpatialDropout1D(dropout_rate)(t)
    t = Bidirectional(CuDNNLSTM(40, return_sequences=True))(t)
    t = Bidirectional(CuDNNGRU(40, return_sequences=True))(t)
    t = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer=initializer)(t)
    avg_pool = GlobalAveragePooling1D()(t)
    max_pool = GlobalMaxPooling1D()(t)
    t = concatenate([avg_pool, max_pool])

    # merge
    x = concatenate([n, c, t])
    x = Dense(64, kernel_initializer=initializer)(x)
    x = LeakyReLU(alpha=5.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(numerical_inputs+categorical_inputs+other_inputs, output)

    return model


def get_block(block_input, dropout_rate=0.2, filter_size=64, kernel_size=3, initializer='he_uniform'):
    block = Conv1D(filter_size, kernel_size=kernel_size, padding='same', activation='linear', kernel_initializer=initializer)(block_input)
    block = BatchNormalization()(block)
    block = Dropout(dropout_rate)(block)
    block = PReLU()(block)
    block = LeakyReLU(alpha=5.5)(block)
    block = Conv1D(filter_size, kernel_size=kernel_size, padding='same', activation='linear', kernel_initializer=initializer)(block)
    block = BatchNormalization()(block)
    block = PReLU()(block)
    return block


def get_dpcnn_model(numerical_features, categorical_features, all_columns, maxvalue_dict,
                    maxlen, num_words, embedding_dims, embedding_matrix=None, seed=777):
    initializer = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=seed)
    dropout_rate = 0.2
    filter_size = 64
    kernel_size = 3
    strides = 2
    pool_size = 3
    n_blocks = 4

    # numerical
    numerical = Input(shape=(len(numerical_features),), name='numerical_features')
    numerical_inputs = [numerical]
    n = numerical
    n = BatchNormalization()(n)
    n = Dense(64, kernel_initializer=initializer)(n)
    n = LeakyReLU(alpha=5.5)(n)

    # categorical
    embedding_size = 8
    categorical_embeds, categorical_inputs = [], []
    for column in categorical_features:
        inp = Input(shape=(1,), name=column)
        categorical_inputs.append(inp)
        emb = Embedding(maxvalue_dict[column], embedding_size, embeddings_initializer=initializer)(inp)
        categorical_embeds.append(emb)
    c = concatenate(categorical_embeds)
    c = Flatten()(c)
    c = Dropout(dropout_rate)(c)

    other_inputs = []

    # text
    text = Input(shape=(maxlen, ), name="seq_text")
    other_inputs.append(text)
    if embedding_matrix is not None:
        t = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(text)
    else:
        t = Embedding(num_words, embedding_dims)(text)
    t = SpatialDropout1D(dropout_rate)(t)
    resize_emb = Conv1D(filter_size, kernel_size=1, padding='same', activation='linear',  kernel_initializer=initializer)(t)
    resize_emb = PReLU()(resize_emb)
    
    block = get_block(t, dropout_rate=dropout_rate, filter_size=filter_size, kernel_size=kernel_size, initializer=initializer)
    block_output = add([block, resize_emb])
    block_output = MaxPooling1D(pool_size=pool_size, strides=strides)(block_output)
    
    for _ in range(n_blocks-2): 
        block = get_block(block_output, dropout_rate=dropout_rate, filter_size=filter_size, kernel_size=kernel_size, initializer=initializer)
        block_output = add([block, block_output])
        block_output = MaxPooling1D(pool_size=pool_size, strides=strides)(block_output)

    block = get_block(block_output, dropout_rate=dropout_rate, filter_size=filter_size, kernel_size=kernel_size, initializer=initializer)
    output = add([block, block_output])
    output = GlobalMaxPooling1D()(output)

    # merge
    x = concatenate([n, c, output])
    x = BatchNormalization()(x)
    x = Dense(64, kernel_initializer=initializer)(x)
    x = LeakyReLU(alpha=5.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(numerical_inputs+categorical_inputs+other_inputs, output)

    return model


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs
    
    
def get_model(numerical_features, categorical_features, all_columns, maxvalue_dict,
                        maxlen, num_words, embedding_dims, embedding_matrix=None, seed=777):
    dropout_rate = 0.2
    initializer = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=seed)

    # numerical
    numerical = Input(shape=(len(numerical_features),), name='numerical_features')
    numerical_inputs = [numerical]
    n = numerical
    n = BatchNormalization()(n)
    n = Dense(64, kernel_initializer=initializer)(n)
    n = LeakyReLU(alpha=5.5)(n)

    # categorical
    embedding_size = 8
    categorical_embeds, categorical_inputs = [], []
    for column in categorical_features:
        inp = Input(shape=(1,), name=column)
        categorical_inputs.append(inp)
        emb = Embedding(maxvalue_dict[column], embedding_size, embeddings_initializer=initializer)(inp)
        categorical_embeds.append(emb)
    c = concatenate(categorical_embeds)
    c = Flatten()(c)
    c = Dropout(dropout_rate)(c)

    other_inputs = []

    # text
    text = Input(shape=(maxlen, ), name="seq_text")
    other_inputs.append(text)
    if embedding_matrix is not None:
        t = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(text)
    else:
        t = Embedding(num_words, embedding_dims)(text)
    t = SpatialDropout1D(dropout_rate)(t)
    t = Bidirectional(CuDNNLSTM(40, return_sequences=True))(t)
    t = Bidirectional(CuDNNGRU(40, return_sequences=True))(t)
#     t = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(t)
    t = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(t)
    avg_pool = GlobalAveragePooling1D()(t)
    max_pool = GlobalMaxPooling1D()(t)
    t = concatenate([avg_pool, max_pool])

    # tfidf
    if 'tfidf' in all_columns:
        tfidf = Input(shape=(4362,), sparse=True, name='tfidf')
        other_inputs.append(tfidf)
        tf = Dense(300, activation="relu")(tfidf)
        t = concatenate([t, tf])

    # image
#     image = Input(shape=(1000, ), name="image")
#     i = Dense(256, activation="relu")(image)

    if 'seq_char_text' in all_columns:
        char_text = Input(shape=(4*maxlen, ), name="seq_char_text")
        other_inputs.append(char_text)
        ct = Embedding(num_chars, embedding_dims)(char_text)
        ct = SpatialDropout1D(dropout_rate)(ct)
        ct = Bidirectional(CuDNNLSTM(40, return_sequences=True))(ct)
        ct = Bidirectional(CuDNNGRU(40, return_sequences=True))(ct)
        ct = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer=initializer)(ct)
        avg_pool = GlobalAveragePooling1D()(ct)
        max_pool = GlobalMaxPooling1D()(ct)
        ct = concatenate([avg_pool, max_pool])
        t = concatenate([t, ct])

    # merge
    x = concatenate([n, c, t])
    x = Dense(64, kernel_initializer=initializer)(x)
    x = LeakyReLU(alpha=5.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(numerical_inputs+categorical_inputs+other_inputs, output)

    return model
