import matplotlib.pyplot as plt
import numpy as np

def plot_real(psit):
    plt.contourf(np.real(np.fft.ifft2(psit)))
    plt.colorbar()
    plt.show()

def test_plot():    
    fig = plt.figure()
    fig.add_subplot(2,1,1)
    d2psi = -25*self.psi
    plt.contourf(d2psi)
    plt.title("Direct")
    plt.colorbar()

    psit = np.fft.fft2(self.psi)
    psit = -psit*self.alpha**2
    d2psi = np.fft.ifft2(psit)

    fig.add_subplot(2,1,2)
    plt.contourf(np.real(d2psi))
    plt.colorbar()
    plt.title("fourier")
    # plt.savefig(f"d2-nx{nx}.jpg")
    plt.show() 
