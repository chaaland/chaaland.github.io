import numpy as np 
from scipy.optimize import minimize
import itertools


def minimize_subject_to_constraints(
  objective, 
  equality_constraints, 
  n: int, 
  x_init: np.ndarray=None, 
  max_iters: int=100,
):
  penalty = 0.1
  n_iters = 0

  if x_init is None:
    x_curr = np.zeros(n)
  elif x_init.size != n:
    raise ValueError(f"x_init has size {x_init.size}, expected {n}")
  else:
    x_curr = x_init

  if isinstance(equality_constraints, (list, tuple)):
    penalised_obj = lambda x: objective(x) + penalty * sum(np.sum(np.square(c(x))) for c in equality_constraints)
  else:
    penalised_obj = lambda x: objective(x) + penalty * np.sum(np.square(equality_constraints(x)))

  x_hist = []
  penalty_schedule = []
  for n_iters in itertools.count():
    x_hist.append(x_curr)
    penalty_schedule.append(penalty)

    if n_iters >= max_iters:
      break

    x_curr = minimize(penalised_obj, x0=x_curr).x
    penalty *= 2
    
  return x_hist, penalty_schedule