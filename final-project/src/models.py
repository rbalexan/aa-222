from sklearn.gaussian_process import GaussianProcessRegressor
import keras


def initialize_gaussian_process(kernel):

    gp = GaussianProcessRegressor(kernel=kernel)

    return gp


def initialize_neural_network(hidden_units, activation):

    # initialize sequential keras model
    model = keras.models.Sequential()

    # add layers using list of hidden layers and activation function
    for i, n_hidden_units in enumerate(hidden_units):

        if i == 0:
            if activation != 'leakyrelu':
                model.add(keras.layers.Dense(n_hidden_units, input_dim=1, activation=activation))
            else:
                model.add(keras.layers.Dense(n_hidden_units, input_dim=1))
                model.add(keras.layers.LeakyReLU())
        else:
            if activation != 'leakyrelu':
                model.add(keras.layers.Dense(n_hidden_units, activation=activation))
            else:
                model.add(keras.layers.Dense(n_hidden_units))
                model.add(keras.layers.LeakyReLU())

    # add output layer with no activation (regression problem)
    model.add(keras.layers.Dense(1, activation=None))

    # specify model loss, optimizer, and initial LR
    model.compile(loss='mse', optimizer=keras.optimizers.adam(learning_rate=0.01), metrics=['mse'])

    return model


def fit_gaussian_process(X_train, y_train, kernel):

    # update gp
    gp = initialize_gaussian_process(kernel)
    gp = gp.fit(X_train, y_train)

    return gp


def fit_neural_network(X_train, y_train, hidden_units, activation, epochs, callbacks):

    # update neural network
    nn = initialize_neural_network(hidden_units, activation)
    nn.fit(X_train, y_train, epochs=epochs, verbose=0, callbacks=callbacks)

    return nn
