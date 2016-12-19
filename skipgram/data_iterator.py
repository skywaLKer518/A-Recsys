import numpy as np
import collections

class DataIterator:
    def __init__(self, seq, end_ind, batch_size, n_skips, window, sequence):
        self.seq = seq
        self.l_seq = len(seq)
        self.end_ind = end_ind
        self.batch_size = batch_size
        self.num_skips = n_skips
        self.skip_window = window
        self.index = 0
        self.sequence = sequence

    def get_next(self):
        seq = self.seq
        users = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        i_items = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        o_items = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        span = 2 * self.skip_window + 1
        b = collections.deque(maxlen=span)
        center = self.skip_window
        
        for _ in range(span):
            b.append(seq[self.index])
            self.index = (self.index + 1) % self.l_seq

        while True:
            if self.sequence:
                self.index = np.random.randint(0, self.l_seq)
                for _ in range(span):
                    b.append(seq[self.index])
                    self.index = (self.index + 1) % self.l_seq

            ind = 0
            while ind < self.batch_size:
                u = b[center][0]
                i_i = b[center][1]
                hi = center if i_i == self.end_ind else span
                targets_to_avoid = [center]

                for _ in range(self.num_skips):
                    t = np.random.randint(0, hi) 
                    while t in targets_to_avoid:
                        t = np.random.randint(0, hi)
                    o_i = b[t][1]
                    if b[t][0] != u or o_i == self.end_ind:
                        continue
                    targets_to_avoid.append(t)
                    users[ind] = u
                    i_items[ind] = i_i
                    o_items[ind] = o_i
                    ind += 1
                    if ind >= self.batch_size:
                        break
                b.append(seq[self.index])
                self.index = (self.index + 1) % self.l_seq
            yield users, i_items, o_items

