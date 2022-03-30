import tensorflow as tf
import math

"""
This script has been adapted from Liu et al. (2019)
https://github.com/wy1iu/MHE
"""
def add_thomson_constraint(W, n_filt, model, power):
    """
    MHE implementation for hidden layers
    
    
    :param input: a weights tensor with shape [filter_size, num_channels, num_filters]
    :param model: indicates MHE model to use (standard or half-space)
    :param power: alternatives for power-s parameter, Euclidean or angular distances
    :return: adds the calculated thompson loss for the current layer to a tf collection
    """
    W = tf.reshape(W, [-1, n_filt])
    if model =='half_mhe':
        W_neg = W*-1
        W = tf.concat((W,W_neg), axis=1)
        n_filt *= 2
    W_norm = tf.sqrt(tf.reduce_sum(W*W, [0], keepdims=True) + 1e-4) # it was originally keep_dims
    norm_mat = tf.matmul(tf.transpose(W_norm), W_norm)
    inner_pro = tf.matmul(tf.transpose(W), W)
    inner_pro /= norm_mat

    if power =='0':
        cross_terms = 2.0 - 2.0 * inner_pro
        final = -tf.log(cross_terms + tf.diag([1.0] * n_filt))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_loss = 1 * tf.reduce_sum(final) / cnt
    elif power =='1':
        cross_terms = (2.0 - 2.0 * inner_pro + tf.diag([1.0] * n_filt))
        final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-0.5))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_loss = 1 * tf.reduce_sum(final) / cnt
    elif power =='2':
        cross_terms = (2.0 - 2.0 * inner_pro + tf.diag([1.0] * n_filt))
        final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-1))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_loss = 1* tf.reduce_sum(final) / cnt
    elif power =='a0':
        acos = tf.acos(inner_pro)/math.pi
        acos += 1e-4
        final = -tf.log(acos)
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_loss = 1* tf.reduce_sum(final) / cnt
    elif power =='a1':
        acos = tf.acos(inner_pro)/math.pi
        acos += 1e-4
        final = tf.pow(acos, tf.ones_like(acos) * (-1))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_loss = 1e-1 * tf.reduce_sum(final) / cnt
    elif power =='a2':
        acos = tf.acos(inner_pro)/math.pi
        acos += 1e-4
        final = tf.pow(acos, tf.ones_like(acos) * (-2))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_loss = 1e-1 * tf.reduce_sum(final) / cnt

    tf.add_to_collection('thomson_loss', th_loss)
    

def add_thomson_constraint_final(W, n_filt, power):
    """
    MHE implementation for output layer
    
    :param input: a weights tensor with shape [filter_size, num_channels, num_filters]
    :param power: alternatives for power-s parameter, Euclidean or angular distances
    :return: adds the calculated thompson loss for the current layer to a tf collection
    """
    W = tf.reshape(W, [-1, n_filt])
    W_norm = tf.sqrt(tf.reduce_sum(W*W, [0], keepdims=True) + 1e-4) # it was originally keep_dims
    norm_mat = tf.matmul(tf.transpose(W_norm), W_norm)
    inner_pro = tf.matmul(tf.transpose(W), W)
    inner_pro /= norm_mat

    if power =='0':
        cross_terms = 2.0 - 2.0 * inner_pro
        final = -tf.log(cross_terms + tf.diag([1.0] * n_filt))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_final = 10 * tf.reduce_sum(final) / cnt
    elif power =='1':
        cross_terms = (2.0 - 2.0 * inner_pro + tf.diag([1.0] * n_filt))
        final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-0.5))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_final = 10 * tf.reduce_sum(final) / cnt
    elif power =='2':
        cross_terms = (2.0 - 2.0 * inner_pro + tf.diag([1.0] * n_filt))
        final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-1))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_final = 10 * tf.reduce_sum(final) / cnt
    elif power =='a0':
        acos = tf.acos(inner_pro)/math.pi
        acos += 1e-4
        final = -tf.log(acos)
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_final = 10 * tf.reduce_sum(final) / cnt
    elif power =='a1':
        acos = tf.acos(inner_pro)/math.pi
        acos += 1e-4
        final = tf.pow(acos, tf.ones_like(acos) * (-1))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_final = 1 * tf.reduce_sum(final) / cnt
    elif power =='a2':
        acos = tf.acos(inner_pro)/math.pi
        acos += 1e-4
        final = tf.pow(acos, tf.ones_like(acos) * (-2))
        final -= tf.matrix_band_part(final, -1, 0)
        cnt = n_filt * (n_filt - 1) / 2.0
        th_final = 1 * tf.reduce_sum(final) / cnt

    tf.add_to_collection('thomson_final', th_final)