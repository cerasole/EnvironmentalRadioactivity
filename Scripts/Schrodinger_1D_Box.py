import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation

from astropy.constants import h, m_e
h_bar = h.to_value("J s") / (2*np.pi)
m_e = m_e.to_value("kg")

plt.rc("font", size = 16)

def get_norm(psi):
    return (psi * psi.conjugate()).real

def Schrodinger_1Dbox (xmin, xmax, N, V, tau, tstop):

    # Spatial grid definition
    x, h = np.linspace(xmin, xmax, N, retstep = True)

    # Time grid
    t = np.arange(0., tstop+tau, tau)

    # Definition of the position wave function
    psi = np.zeros(N, dtype = np.complex128)

    # Initial condition for the wave function
    ymax = 1.
    p = 2 * np.pi * h_bar
    sigma = 2.
    prefactor = 1. / (np.sqrt(sigma * np.sqrt(np.pi)))
    psi = np.exp(1j * p * x / h_bar) * np.exp( - (x)**2 / (2 * sigma**2))

    # Boundary condition for the wave function (we assume an infinite potential at x = xmin and at x = xmax)
    psi[0] = psi[-1] = 0.

    # Definition of the A and B matrices
    A, B = np.zeros((N, N), dtype = np.complex128), np.zeros((N, N), dtype = np.complex128)
    term1 = (4 * m_e * h**2) / (h_bar * tau) * 1j
    term2 = 2 + 2 * m_e * h**2 * V / h_bar**2
    A[0][0] = A[-1][-1] = B[0][0] = B[-1][-1] = 1.
    for i in range(1, N-1):
        A[i][i-1] = A[i][i+1] = 1.
        B[i][i-1] = B[i][i+1] = -1.
        A[i][i] = term1 - term2[i]
        B[i][i] = term1 + term2[i]

    skip = int(len(t) / 100)

    plt.figure(figsize = (8, 6))

    # Numerical solution
    for k in range(len(t)):
        r = B.dot(psi)
        r[0] = r[-1] = 0.
        psi_new = scipy.linalg.solve(A, r)
        psi = psi_new.copy()

        if k % skip == 0:
            # Plot
            plt.clf()
            plt.subplot(1,1,1)
            plt.plot(x, get_norm(psi))
            #plt.plot(x, psi.real)
            plt.ylim(0., ymax * 1.5)
            plt.title("Time = %.1e s" % t[k])
            plt.xlabel("$x$")
            plt.ylabel("$|\\psi (t, x)|^{2}$")
            plt.pause(0.1)

    print ("Successfully completed the execution!")
    plt.show()
    return


def Schrodinger_1Dbox_animation (xmin, xmax, N, V, tau, tstop, outname = "schrodinger_1D.gif"):

    # Spatial grid definition
    x, h = np.linspace(xmin, xmax, N, retstep = True)

    # Time grid
    t = np.arange(0., tstop+tau, tau)

    # Definition of the position wave function
    psi = np.zeros(N, dtype = np.complex128)

    # Initial condition for the wave function
    ymax = 1.
    p = 2 * np.pi * h_bar
    sigma = 2.
    prefactor = 1. / (np.sqrt(sigma * np.sqrt(np.pi)))
    psi = np.exp(1j * p * x / h_bar) * np.exp( - (x)**2 / (2 * sigma**2))

    # Boundary condition for the wave function (we assume an infinite potential at x = xmin and at x = xmax)
    psi[0] = psi[-1] = 0.

    # Stato iniziale globale
    psi_current = psi.copy()  # psi iniziale

    # Definition of the A and B matrices
    A, B = np.zeros((N, N), dtype = np.complex128), np.zeros((N, N), dtype = np.complex128)
    term1 = (4 * m_e * h**2) / (h_bar * tau) * 1j
    term2 = 2 + 2 * m_e * h**2 * V / h_bar**2
    A[0][0] = A[-1][-1] = B[0][0] = B[-1][-1] = 1.
    for i in range(1, N-1):
        A[i][i-1] = A[i][i+1] = 1.
        B[i][i-1] = B[i][i+1] = -1.
        A[i][i] = term1 - term2[i]
        B[i][i] = term1 + term2[i]

    skip = int(len(t) / 100)

    fig, ax = plt.subplots(figsize = (8, 6))
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0., ymax * 1.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|\\psi (t, x)|^{2}$")
    title = ax.set_title("")

    # Prepara lista dei frame
    frame_indices = list(range(0, len(t), skip))

    def init():
        line.set_data([], [])
        title.set_text("")
        return line, title

    def update(k_idx):
        nonlocal psi_current
        k_real = frame_indices[k_idx]

        for _ in range(skip):
            r = B.dot(psi_current)
            r[0] = r[-1] = 0.
            psi_current = scipy.linalg.solve(A, r)

        #psi_current = psi_new.copy()
        line.set_data(x, get_norm(psi_current))
        title.set_text("Time = %.1e s" % t[k_real])
        return line, title

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                  init_func=init, blit=True, interval=100)

    if ".gif" in outname:
        ani.save(outname, writer="pillow", fps=10)
        plt.close()
    elif ".mp4" in outname:
        ani.save(outname, writer="ffmpeg", fps=30)
        plt.close()

    print ("Successfully completed the execution!")

    return


# Particle in a 1D box, surrounded by infinite walls of potential
L = 10
xmin, xmax = -L, L
N = 200
V = np.zeros(N)
tau = 10
tstart, tstop = 0., tau*10000.
#Schrodinger_1Dbox (xmin, xmax, N, V, tau, tstop)
outname = "schrodinger_1D.gif"
outname = "schrodinger_1D.mp4"
#Schrodinger_1Dbox_animation (xmin, xmax, N, V, tau, tstop, outname)


# Particle in an infinite wall of potential, with a finite barrier of potential in the box
xmin, xmax = -2*L, 2*L
N = 400
V0 = 1e-36
x, h = np.linspace(xmin, xmax, N, retstep = True)
V = np.where(
    x < 10,
    0.,
    np.where(
        x > 10.1,
        0,
        V0,
    )
)
tau = 10
tstart, tstop = 0., tau*10000.
#Schrodinger_1Dbox (xmin, xmax, N, V, tau, tstop)
outname = "schrodinger_1D_tunnel.gif"
outname = "schrodinger_1D_tunnel.mp4"
Schrodinger_1Dbox_animation (xmin, xmax, N, V, tau, tstop, outname)

# E = p**2 / (2*m) + h_bar**2 / (4 * m * sigma**2) =
#   = 2.4e-37 J + 0.6e-37 J =
#   = 3e-37 J
