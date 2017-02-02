class Task(object):
  ''' a wrapper of datasets/evaluations'''
  def __init__(self, name):
    if name == 'xing':
      from xing_data import data_read as data_read_xing
      from xing_eval import Evaluate as Evaluate_xing
      self.data_read = data_read_xing
      self.evaluate = Evaluate_xing
    elif name == 'ml':
      from ml_data import data_read as data_read_ml
      from ml_eval import Evaluate as Evaluate_ml
      self.data_read = data_read_ml
      self.evaluate = Evaluate_ml
    elif name == 'ml100k':
      from ml100k_data import data_read as data_read_ml100k
      from ml100k_eval import Evaluate as Evaluate_ml100k
      self.data_read = data_read_ml100k
      self.evaluate = Evaluate_ml100k
    elif name == 'ml1m':
      from ml1m_data import data_read as data_read_ml1m
      from ml1m_eval import Evaluate as Evaluate_ml1m
      self.data_read = data_read_ml1m    
      self.evaluate = Evaluate_ml1m
    else:
      print("Error. dataset not used!!")
      exit(-1)
    return

  def get_dataread(self):
    return self.data_read

  def get_evaluation(self):
    return self.evaluate