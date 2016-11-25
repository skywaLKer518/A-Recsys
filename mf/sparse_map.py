import tensorflow as tf

def create_sparse_map(Vals, starts, lens, mb, mtl, ith, prefix):
    '''
    Create sparse tensor given an array and 2 index vectors.
    input:
        Vals: 1-D tensor (mb, ), dense,
        starts: 1-D tensor (mb, ) starting position
        lens: 1-D tensor (mb, ) size
        mb: minibatch size, scalar
        mtl: max text length, a scalar
    return:
        st: sparse tensor, [mb by  mtl]
    '''
    
    sp_shape = [mb, mtl]
    value_list, idx_list = [], []

    for i in xrange(mb):
      l = tf.reshape(tf.slice(lens, [i], [1]), [], name='%s_reshape_i%d_mb%d' % (prefix, ith, i))
      s = tf.slice(starts, [i], [1], name='%s_slice_s_i%d_mb%d'% (prefix, ith, i))
      val = tf.slice(Vals, s, [l], name='%s_slice_v_i%d_mb%d' % (prefix, ith, i))
      value_list.append(val)
      col1 = tf.fill([l, 1], i, name='%s_fill_1_i%d_mb%d' % (prefix, ith, i))
      col2 = tf.reshape(tf.range(0, l), [l,1], name='%s_fill_2_i%d_mb%d' % (prefix, ith, i))
      idx_list.append(tf.concat(1, [col1, col2])) # l1 * 2        

    values = tf.concat(0, value_list, name='%s_concat_v_i%d' % (prefix, ith))
    indices = tf.concat(0, idx_list, name='%s_concat_i_i%d' % (prefix, ith))

    indices = tf.to_int64(indices, name='%s_toint64_i%d' % (prefix, ith))
    sp_shape = tf.to_int64(sp_shape, name='%s_toint64_i%d' % (prefix, ith))
    st = tf.SparseTensor(indices, values, sp_shape)
    return st
