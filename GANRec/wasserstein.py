
import tensorflow as tf


def autoencoder_wasserstein(input_data, decoding):
    # config = tf.ConfigProto()
    # sess = tf.Session(config=config)
    # # Définir la valeur initiale de lambda
    # lambda_init = 0.1
    # # Définir lambda comme une variable tensorflow
    # LAMBDA = tf.Variable(lambda_init, dtype=tf.float32, name='lambda')
    # # Définir le facteur de décroissance de lambda
    # lambda_decay = 0.9
    # # Définir la valeur minimale de lambda
    # lambda_min = 0.1
    # # Définir une opération tensorflow pour mettre à jour lambda
    # update_lambda = tf.assign(LAMBDA, tf.maximum(LAMBDA * lambda_decay, lambda_min))

    def discriminator(x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            # h1 = tf.layers.dense(x, units=64, activation='relu', name='h1')
            # h2 = tf.layers.dense(h1, units=32, activation='relu', name='h2')
            score = tf.layers.dense(x, units=1, name='score')
        return score
    # Calculer le score du discriminateur pour input_data
    D_X = discriminator(input_data)
    # Calculer le score du discriminateur pour decoding
    D_decoding = discriminator(decoding)
    # Générer des valeurs aléatoires entre 0 et 1
    epsilon = tf.random_uniform(shape=tf.shape(input_data), minval=0., maxval=1.)
    # Calculer le terme x_hat
    x_hat = epsilon * input_data + (1.0 - epsilon) * decoding
    # Calculer le score du discriminateur pour x_hat
    D_X_hat = discriminator(x_hat)
    # Calculer le gradient du score du discriminateur par rapport à x_hat
    grad_D_X_hat = tf.gradients(D_X_hat, [x_hat])[0]
    # Calculer la norme du gradient
    red_idx = list(range(1, x_hat.shape.ndims))
    grad = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    # Calculer la pénalité du gradient
    gradient_penalty = tf.reduce_mean((grad - 1.)**2)
    # Calculer la fonction de perte
    LAMBDA = 10
    loss = tf.reduce_mean(D_decoding - tf.reduce_mean(D_X) ) + LAMBDA * tf.reduce_mean(gradient_penalty)
    # Mettre à jour la valeur de lambda
    # sess.run(update_lambda)
    return loss

def distance_wasserstein(input_data, decoding):
    # config = tf.ConfigProto()
    # sess = tf.Session(config=config)
    # # Définir la valeur initiale de lambda
    # lambda_init = 0.1
    # # Définir lambda comme une variable tensorflow
    # LAMBDA = tf.Variable(lambda_init, dtype=tf.float32, name='lambda')
    # # Définir le facteur de décroissance de lambda
    # lambda_decay = 0.9
    # # Définir la valeur minimale de lambda
    # lambda_min = 0.1
    # # Définir une opération tensorflow pour mettre à jour lambda
    # update_lambda = tf.assign(LAMBDA, tf.maximum(LAMBDA * lambda_decay, lambda_min))

    def discriminator(x):
        with tf.variable_scope('discriminator_second', reuse=False):
            # h1 = tf.layers.dense(x, units=64, activation='relu', name='h1')
            # h2 = tf.layers.dense(h1, units=32, activation='relu', name='h2')
            score = tf.layers.dense(x, units=1, name='score')
        return score
    # Calculer le score du discriminateur pour input_data
    D_X = discriminator(input_data)
    # Calculer le score du discriminateur pour decoding
    D_decoding = discriminator(decoding)
    # Générer des valeurs aléatoires entre 0 et 1
    epsilon = tf.random_uniform(shape=tf.shape(input_data), minval=0., maxval=1.)
    # Calculer le terme x_hat
    x_hat = epsilon * input_data + (1.0 - epsilon) * decoding
    # Calculer le score du discriminateur pour x_hat
    D_X_hat = discriminator(x_hat)
    # Calculer le gradient du score du discriminateur par rapport à x_hat
    grad_D_X_hat = tf.gradients(D_X_hat, [x_hat])[0]
    # Calculer la norme du gradient
    red_idx = list(range(1, x_hat.shape.ndims))
    grad = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    # Calculer la pénalité du gradient
    gradient_penalty = tf.reduce_mean((grad - 1.)**2)
    # Calculer la fonction de perte
    LAMBDA = 10
    loss = tf.reduce_mean(D_decoding - tf.reduce_mean(D_X) ) + LAMBDA * tf.reduce_mean(gradient_penalty)
    # Mettre à jour la valeur de lambda
    # sess.run(update_lambda)
    return loss
