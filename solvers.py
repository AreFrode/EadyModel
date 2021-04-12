def forward_euler(psit, alpha, m, dt):
    return psit - ((1j*m*dt)/alpha)*(1+alpha)*psit

def runge_kutta2(psit, alpha, m, dt):
    psit_half = psit - ((1j*m*(0.5)*dt)/(alpha))*(1+alpha)*psit
    return psit - ((1j*m*dt)/alpha)*(1+alpha)*psit_half


