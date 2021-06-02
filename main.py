from eady import eady
import numpy as np
import solvers

eady = eady(64)
eady.initialize(lambda x, y: np.sin(3*x)*np.sin(4*y))

# eady.animate_toplayer(10, 0.01, solvers.forward_euler)

# eady.animate_toplayer(10, 0.01, solvers.runge_kutta4)

# eady.simple_twolayer(1, 0.5, 3, 0, 10, 0.01)

eady.simple_twolayer(1, 0.5, 3, 4, 10, 0.01)

# ------------------------------------------------------

# eady.animate_toplayer(10, 0.01, solvers.runge_kutta2)

# eady.alternative_top(1, 3, 4, 10, 0.01)
# eady.simple_twolayer(1, 1, 3, 4, 10, 0.01)

# eady.simple_twolayer(1, 0.5, 5, 8, 10, 0.01)
# eady.simple_twolayer(1, 0.5, 0.5, 0.1, 10, 0.01)
# eady.simple_twolayer(1, 0.5, 4, 3, 10, 0.01)

# eady.alternative_twolayer(1, 0.2, 3, 4, 10, 0.01)
