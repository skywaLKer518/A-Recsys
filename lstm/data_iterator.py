import numpy as np

PAD_ID = 0
START_ID = 1

class DataIterator:
    def __init__(self, model, data_set, n_bucket, batch_size, train_buckets_scale):
        self.data_set = data_set
        self.n_bucket = n_bucket
        self.batch_size = batch_size
        self.train_buckets_scale = train_buckets_scale
        self.model = model

    def next_random(self):
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(self.train_buckets_scale))
                             if self.train_buckets_scale[i] > random_number_01])

            users, inputs, outputs, weights, _ = self.model.get_batch(self.data_set, bucket_id)
            yield users, inputs, outputs, weights, bucket_id

    def next_sequence(self, stop=False):
        bucket_id = 0
        while True:
            if bucket_id >= self.n_bucket:
                if stop:
                    break
                bucket_id = 0
            start_id = 0
            while True:
                users, inputs, outputs, weights, finished = self.model.get_batch(self.data_set, bucket_id, start_id = start_id)
                if finished:
                    break
                start_id += self.batch_size
                yield users, inputs, outputs, weights, bucket_id
            bucket_id += 1
            
            
