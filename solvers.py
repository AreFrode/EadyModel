def forward_euler(psit, alpha, m, dt):
    return psit - ((1j*m*dt)/alpha)*(1.+alpha)*psit

def runge_kutta2(psit, alpha, m, dt):
    k1 = -((1j*m*dt)/alpha)*(1.+alpha)*psit
    k2 = -((1j*m*dt)/alpha)*(1.+alpha)*(psit + 0.5*k1)

    return psit + k2

def runge_kutta4(psit, alpha, m, dt):
    k1 = -((1j*m*dt)/alpha)*(1.+alpha)*psit
    k2 = -((1j*m*dt)/alpha)*(1.+alpha)*(psit+0.5*k1)
    k3 = -((1j*m*dt)/alpha)*(1.+alpha)*(psit+0.5*k2)
    k4 = -((1j*m*dt)/alpha)*(1.+alpha)*(psit + k3)

    return psit + (1./6)*(k1 + 2.*k2 + 2.*k3 + k4)
