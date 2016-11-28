import tensorflow as tf
import numpy as np

def __cumprod(l):
    # Get the length and make a copy
    ll = len(l)
    l = [v for v in l]

    # Reverse cumulative product
    for i in range(ll-1):
        l[ll-i-2] *= l[ll-i-1]

    return l

def ravel_multi_index(tensor, multi_idx):
    """
    Returns a tensor suitable for use as the index
    on a gather operation on argument tensor.
    """

    if not isinstance(tensor, (tf.Variable, tf.Tensor)):
        raise TypeError('tensor should be a tf.Variable')

    if not isinstance(multi_idx, list):
        multi_idx = [multi_idx]

    # Shape of the tensor in ints
    shape = [i.value for i in tensor.get_shape()]

    if len(shape) != len(multi_idx):
        raise ValueError("Tensor rank is different "
                        "from the multi_idx length.")

    # Work out the shape of each tensor in the multi_idx
    idx_shape = [tuple(j.value for j in i.get_shape()) for i in multi_idx]
    # Ensure that each multi_idx tensor is length 1
    assert all(len(i) == 1 for i in idx_shape)

    # Create a list of reshaped indices. New shape will be
    # [1, 1, dim[0], 1] for the 3rd index in multi_idx
    # for example.
    reshaped_idx = [tf.reshape(idx, [1 if i !=j else dim[0]
                    for j in range(len(shape))])
                for i, (idx, dim)
                in enumerate(zip(multi_idx, idx_shape))]

    # Figure out the base indices for each dimension
    base = __cumprod(shape)

    # Now multiply base indices by each reshaped index
    # to produce the flat index
    return (sum(b*s for b, s in zip(base[1:], reshaped_idx[:-1]))
        + reshaped_idx[-1])


# # Shape and slice starts and sizes
# shape = (Z, Y, X) = 4, 5, 6
# Z0, Y0, X0 = 1, 1, 1
# ZS, YS, XS = 3, 3, 4

# # Numpy matrix and index
# M = np.random.random(size=shape)
# idx = [
#     np.arange(Z0, Z0+ZS).reshape(ZS,1,1),
#     np.arange(Y0, Y0+YS).reshape(1,YS,1),
#     np.arange(X0, X0+XS).reshape(1,1,XS),
# ]

# # Tensorflow matrix and indices
# TM = tf.Variable(M)
# TF_flat_idx = ravel_multi_index(TM, [
#     tf.range(Z0, Z0+ZS),
#     tf.range(Y0, Y0+YS),
#     tf.range(X0, X0+XS)])
# TF_data = tf.gather(tf.reshape(TM,[-1]), TF_flat_idx)

# with tf.Session() as S:
#     S.run(tf.initialize_all_variables())

#     # Obtain data via flat indexing
#     data = S.run(TF_data)
#     print(M)
#     # Check that it agrees with data obtained
#     # by numpy smart indexing
#     assert np.all(data == M[idx])
#     print(data)









