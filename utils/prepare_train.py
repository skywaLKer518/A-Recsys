import numpy as np
import datetime import datetime

def to_week(t):
  return datetime.fromtimestamp(t).isocalendar()[1]

def sample_items(items, n, p, replace=False):
  item_sampled = np.random.choice(items, n, replace=replace, p=p)
  item_sampled_id2idx = {}
  i = 0
  for item in item_sampled:
    item_sampled_id2idx[item] = i
    i += 1
  return item_sampled, item_sampled_id2idx

def item_frequency(data_tr, power):
  ''' count item frequency and compute sampling prob'''
  item_counts = {}  
  item_population = set([])
  for u, i, _ in data_tr:
    item_counts[i] = 1 if i not in item_counts else item_counts[i] + 1
    item_population.add(i)
  item_population = list(item_population)
  counts = [item_counts[v] for v in item_population]
  print(len(item_population))

  count_sum = sum(counts) * 1.0

  p_item_unormalized = [np.power(c / count_sum, power) for c in counts]
  p_item_sum = sum(p_item_unormalized)
  p_item = [f / p_item_sum for f in p_item_unormalized]
  return item_population, p_item

def positive_items(data_tr, data_va):
  hist, hist_va = {}, {}
  for u, i, _ in data_tr:
    if u not in hist:
      hist[u] = set([i])
    else:
      hist[u].add(i)
  for u, i, _ in data_va:
    if u not in hist_va:
      hist_va[u] = set([i])
    else:
      hist_va[u].add(i)

  pos_item_list = {}
  pos_item_list_val = {}
  for u in hist:
    pos_item_list[u] = list(hist[u])
  for u in hist_va:
    pos_item_list_val[u] = list(hist_va[u])

  return pos_item_list, pos_item_list_val