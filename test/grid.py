import numpy as np
import matplotlib.pyplot as plt

# Simple test to set up the wavenumber grid

nx = 8       # assume rectangular grid

m = np.arange(nx)
n = np.arange(nx)

for i in range(1, int(0.5*nx)):
    m[nx-i] = -m[i]
    n[nx-i] = -n[i]

alpha = np.zeros((nx,nx))
psi = np.zeros((nx,nx))
for i in range(nx):
    for j in range(nx):
        # print(f"multiplying {m[i]}^2 + {n[j]}^2")
        alpha[i][j] = np.sqrt(m[i]**2 + n[j]**2)

x = np.arange(1, nx+1)
x = x*2*np.pi/nx - np.pi

print(x)
for i in range(len(x)):
    for j in range(len(x)):
        psi[i][j] = np.sin(3*x[i])*np.sin(4*x[j])
# print(alpha[-1,:])

fig = plt.figure()
fig.add_subplot(2,1,1)
d2psi = -25*psi
plt.contourf(d2psi)
plt.title("Direct")
plt.colorbar()

psit = np.fft.fft2(psi)
psit = -psit*alpha**2
d2psi = np.fft.ifft2(psit)

fig.add_subplot(2,1,2)
plt.contourf(np.real(d2psi))
plt.colorbar()
plt.title("fourier")
# plt.savefig(f"d2-nx{nx}.jpg")
# plt.show()

