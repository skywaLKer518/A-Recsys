def calculate_buckets(array, max_length, max_buckets):
    d = {} 
    for u,ll in array:
        length = len(ll)
        if not length in d:
            d[length] = 0
        d[length] += 1
    
    dd = [(x, d[x]) for x in d]
    dd = sorted(dd, key = lambda x: x[0])
    running_sum = []
    s = 0
    for l, n in dd:
        s += n
        running_sum.append((l,s))

    def best_point(ll):
        # return index so that l[:index+1] and l[index+1:]
        index = 0
        maxv = 0
        base = ll[0][1]
        for i in xrange(len(ll)):
            l,n = ll[i]
            v = (ll[-1][0] - l) * (n-base)
            if v > maxv:
                maxv = v
                index = i
        return index, maxv
    
    def arg_max(array,key):
        maxv = -10000
        index = -1
        for i in xrange(len(array)):
            item = array[i]
            v = key(item)
            if v > maxv:
                maxv = v
                index = i
        return index

    end_index = 0
    for i in xrange(len(running_sum)-1,-1,-1):
        if running_sum[i][0] <= max_length:
            end_index = i+1
            break

    print running_sum

    if end_index <= max_buckets:
        buckets = [x[0] for x in running_sum[:end_index]]
    else:
        buckets = []
        # (array,  maxv, index)
        states = [(running_sum[:end_index],0,end_index-1)]
        while len(buckets) < max_buckets:
            index = arg_max(states, lambda x: x[1])
            state = states[index]
            del states[index]
            #split state
            array = state[0]
            split_index = state[2]
            buckets.append(array[split_index][0])
            array1 = array[:split_index+1]
            array2 = array[split_index+1:]
            if len(array1) > 0:
                id1, maxv1 = best_point(array1)
                states.append((array1,maxv1,id1))
            if len(array2) > 0:
                id2, maxv2 = best_point(array2)
                states.append((array2,maxv2,id2))
    return buckets

def main():

    import random
    a = []
    for i in xrange(1000):
        l = random.randint(1,50)
        a.append([0]*l)
    max_length = 40
    max_buckets = 4
    print calculate_buckets(a,max_length, max_buckets)

if __name__ == "__main__":
    main()
