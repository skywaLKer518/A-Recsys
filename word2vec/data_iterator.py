import numpy as np


class DataIterator:
    def __init__(self, model, seq, end_ind, batch_size, n_skips, window):
        self.seq = seq
        self.l_seq = len(seq)
        self.end_ind = end_ind
        self.model = model
        self.batch_size = batch_size
        self.num_skips = n_skips
        self.skip_window = window
        self.index = 0
    def next_random(self):
        while True:
            


            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(self.train_buckets_scale))
                             if self.train_buckets_scale[i] > random_number_01])

            users, inputs, outputs, weights, _ = self.model.get_batch(self.data_set, bucket_id)
            yield users, inputs, outputs, weights, bucket_id

    def next_sequence(self, stop=False, recommend = False):
        users = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        i_items = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        o_items = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        span = 2 * self.skip_window
        buffer = collections.deque(maxlen=span)
        while True:
            self.index




            if bucket_id >= self.n_bucket:
                if stop:
                    break
                bucket_id = 0
            start_id = 0
            while True:
                get_batch_func = self.model.get_batch
                if recommend:
                    get_batch_func = self.model.get_batch_recommend
                users, inputs, outputs, weights, finished = get_batch_func(self.data_set, bucket_id, start_id = start_id)
                yield users, inputs, outputs, weights, bucket_id
                if finished:
                    break
                start_id += self.batch_size
            bucket_id += 1
            
            
