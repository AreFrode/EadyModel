import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from timer import timer

class eady:
    """
    Simple class-representation of the Eady model

    assumes rectangular grid
    """

    def __init__(self, nx):
        self.nx = nx
        self.m = np.arange(nx)
        self.n = np.arange(nx)

        for i in range(1, int(0.5*nx)):
            self.m[nx-i] = -self.m[i]
            self.n[nx-i] = -self.n[i]

        self.alpha = np.zeros((nx,nx))
        for i in range(nx):
            for j in range(nx):
                self.alpha[i][j] = np.sqrt(self.m[i]**2 + self.n[j]**2)

    def initialize(self, streamfunc):
        self.psi = np.zeros_like(self.alpha)
        self.alpha[0,0] = 1e8 # m=n=0 corresponds to constant psi, don't need it
        
        x = np.arange(1, self.nx+1)
        x = x*2*np.pi/self.nx - np.pi

        for i in range(len(x)):
            for j in range(len(x)):
                self.psi[i][j] = streamfunc(x[i], x[j])

    # @timer
    def solve_toplayer(self, nk, dt, solver):
        self.m2 = np.stack((self.m,)*self.nx,axis=0)
        psit = np.fft.fft2(self.psi)
        stop = int(nk/dt)
        counter = 0
        plot_interval = int(0.1*stop)

        for k in range(stop):
            if (counter == plot_interval):
                counter = 0
                self.plot_real(psit)

            psit = solver(psit, self.alpha, self.m2, dt)
            counter += 1

        self.plot_real(psit)

    def animate_toplayer(self, nk, dt, solver):
        self.m2 = np.stack((self.m,)*self.nx,axis=0)
        psit = np.fft.fft2(self.psi)
        G = []
        G.append(self.psi)

        for k in range(int(nk/dt)):
            psit = solver(psit, self.alpha, self.m2, dt)
            G.append(np.real(np.fft.ifft2(psit)))

        fig, ax = plt.subplots()
        
        def animate(i):
            ax.collections = []
            ax.contourf(G[i])

        anim = FuncAnimation(fig, animate, interval=10)

        plt.draw()
        plt.show()
    
    def plot_real(self, psit):
        plt.contourf(np.real(np.fft.ifft2(psit)))
        plt.colorbar()
        plt.show()


    def test_plot(self):    
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
