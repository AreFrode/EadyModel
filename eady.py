import plotter
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
                plotter.plot_real(psit)

            psit = solver(psit, self.alpha, self.m2, dt)
            counter += 1

        self.plot_real(psit)


    def animate_toplayer(self, nk, dt, solver):
        self.m2 = np.stack((self.m,)*self.nx,axis=0)
        psit = np.fft.fft2(self.psi)

        # Computing the frames
        frames = []
        frames.append(self.psi)

        for k in range(int(nk/dt)):
            psit = solver(psit, self.alpha, self.m2, dt)
            frames.append(np.real(np.fft.ifft2(psit)))
    
        # setting up the canvas
        fig = plt.figure()
        ax = fig.add_subplot(111)

        frame = frames[0]
        cf = ax.contourf(frame)
        tx = ax.set_title('Frame 0, max = {np.max(frame):,.2f}')

        def _animate(i):
            ax.collections = []
            frame = frames[i]
            cf = ax.contourf(frame) 
            tx.set_text(f'Frame {i}, max = {np.max(frame):,.2f}')
            
        anim = FuncAnimation(fig, _animate, interval=10)

        plt.draw()
        plt.show()
 
    def simple_twolayer(self, A, B, m, n, nk, dt):
        # Hardwired model, results seem to be wrong
        alpha = np.sqrt(m**2 + n**2)
        C = np.array([[A],[B]])
        N = np.array([[np.exp(-alpha), -1], [1, -np.exp(-alpha)]])
        M = np.array([[((1j*m)/alpha)*np.exp(-alpha), (1j*m)/alpha], [-(1-(1./alpha))*1j*m, (1 + (1./alpha))*1j*m*np.exp(-alpha)]])
        P = np.linalg.inv(N)@M
        I = np.identity(2)

        psif_bot = A*np.exp(-self.alpha) + B
        psif_top = A + B*np.exp(-self.alpha)

        frames = []
        frames.append([np.real(np.fft.ifft2(psif_bot)),np.real(np.fft.ifft2(psif_top))])

        for i in range(int(nk/dt)):
            C = (I + dt*P)@C
            psif_bot = C[0]*np.exp(-self.alpha) + C[-1]
            psif_top = C[0] + C[-1]*np.exp(-self.alpha)
            frames.append([np.real(np.fft.ifft2(psif_bot)),np.real(np.fft.ifft2(psif_top))])
                
        # setting up the canvas
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        frame = frames[0]
        cf1 = ax1.contourf(frame[-1])
        tx1 = ax1.set_title('Frame 0, max = {np.max(frame[-1]):,.2f}')

        cf2 = ax2.contourf(frame[0])
        tx2 = ax2.set_title('         max = {np.max(frame[0]):,.2f}')

        def _animate(i):
            ax1.collections = []
            ax2.collections = []
            frame = frames[i]
            cf1 = ax1.contourf(frame[-1]) 
            tx1.set_text(f'Frame {i}, max = {np.max(frame[-1]):,.2f}')

            cf2 = ax2.contourf(frame[0])
            tx2 = ax2.set_title(f'         max = {np.max(frame[0]):,.2f}')


        anim = FuncAnimation(fig, _animate, interval=10)

        plt.draw()
        plt.show()

        # print(np.real(np.fft.ifft2(C)))
