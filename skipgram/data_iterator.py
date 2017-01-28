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

    def get_next2(self):
        # only predict future based on history
        seq = self.seq
        users = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        i_items = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        o_items = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        span = 2 * self.skip_window + 1
        b = collections.deque(maxlen=span)
        # center = self.skip_window
        center = 0

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
                hi = span
                # hi = center if i_i == self.end_ind else span
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

    def get_next3(self):
        # cbow, only predict current based on history
        seq = self.seq
        users = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        # i_items = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        i_items = [ [0] * self.num_skips] * self.batch_size
        o_items = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        span = self.skip_window + 1
        b = collections.deque(maxlen=span)
        center = span - 1

        for _ in range(span):
            b.append(seq[self.index])
            self.index = (self.index + 1) % self.l_seq

        u_seq_len = -1
        u, o_i = b[center]
        if o_i == self.end_ind:
            u_seq_len = 0
        else:
            for ii in range(center):
                uu = b[ii][0]
                if uu == u:
                    u_seq_len += 1

        while True:
            if self.sequence:
                print('error: not implemented')
                exit(1)
                self.index = np.random.randint(0, self.l_seq)
                for _ in range(span):
                    b.append(seq[self.index])
                    self.index = (self.index + 1) % self.l_seq

            ind = 0
            while ind < self.batch_size:
                u = b[center][0]
                o_i = b[center][1]
                if o_i == self.end_ind: # actually the start
                    b.append(seq[self.index])
                    self.index = (self.index + 1) % self.l_seq
                    u_seq_len = 0
                    continue
                u_seq_len += 1
                u_seq_len = min(u_seq_len, center)

                with_replacement = (u_seq_len < self.num_skips)
                i_samples = np.random.choice(center, self.num_skips, with_replacement)

                i_is = [b[j][1] for j in i_samples]

                i_items[ind] = i_is
                users[ind] = u
                o_items[ind] = o_i
                ind += 1
                
                b.append(seq[self.index])
                self.index = (self.index + 1) % self.l_seq

                i_items_input = batch_major(i_items, self.batch_size, self.num_skips)

            yield users, i_items_input, o_items


def batch_major(l, m, n):
    output = []
    for i in range(n):
        tmp = []
        for j in range(m):
            tmp.append(l[j][i])
        output.append(tmp)
    return output