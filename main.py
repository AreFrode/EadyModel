from eady import eady
import numpy as np
import solvers

eady = eady(64)
eady.initialize(lambda x,y: np.sin(3*x)*np.sin(4*y))
# eady.solve_toplayer(10, 0.01, solvers.runge_kutta4)
# eady.solve_toplayer(10, 0.01, solvers.runge_kutta2)
# eady.solve_toplayer(10, 0.01, solvers.forward_euler) 
# eady.animate_toplayer(10, 0.01, solvers.forward_euler)
# eady.animate_toplayer(10, 0.01, solvers.runge_kutta2)
# eady.animate_toplayer(10, 0.01, solvers.runge_kutta4)
eady.simple_twolayer(1, 0.2, 3, 4, 10, 0.01)
# eady.alternative_twolayer(1, 0.2, 3, 4, 10, 0.01)
