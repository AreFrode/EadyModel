from eady import eady
from solvers import forward_euler, runge_kutta2
import numpy as np

eady = eady(64)
eady.initialize(lambda x,y: np.sin(3*x)*np.sin(4*y))
eady.solve_toplayer(10, 0.01, runge_kutta2)
# eady.solve_toplayer(10, 0.01, forward_euler) 
