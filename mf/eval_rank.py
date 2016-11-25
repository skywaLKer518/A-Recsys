



def eval_P5(X, T, K=5):
  score = 0
  for uid in T:
    if uid not in X:
      continue
    t = set(T[uid])  
    pred = set(X[uid][:K])
    score += len(pred.intersection(t)) * 1.0 / len(pred)
  return score * 1.0 / len(T)

def eval_R20(X, T):
  '''
  actually it is the success rate of 20
  '''
  POS = 20
  success = 0
  for uid in T:
    if uid not in X:
      continue
    suc = 1 if len(set(X[uid][:POS]).intersection(T[uid])) > 0 else 0
    success += suc
  return 1.0 * success / (len(T))

