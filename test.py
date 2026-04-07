import numpy as np
import matplotlib.pyplot as plt

T0 = 1
R = 1 / (3 * T0) # c^2 = 3RT_0 = 1

H = 20
Pr = 0.71

Ra = 10000

nu = H * np.sqrt(0.1 * Pr / Ra)
chi = nu / Pr

tau_v = nu / (R * T0)
tau_c = chi / (2 * R * T0)

T1 = 1.000625

T_m = (T1 + T0) / 2
dT = T1 - T0

g0 = -9.81

beta = 0.1 / (g0 * dT * H)

rho0 = 1

Nx = 80
Ny = (40+1) + 2 # Top and bottom are ghost nodes
NL = 9
Nt = 4000

def LBM():
    show_animation = False
    is_output_data = False
    output_data = None

    #   6 --- 2 --- 5
    #   |     |     |
    #   3 --- 0 --- 1
    #   |     |     |
    #   7 --- 4 --- 8
    idxs = np.arange(NL)
    cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    weights = (1 / 36) * np.array([16, 4, 4, 4, 4, 1, 1, 1, 1])  # sums to 1

    # Initialize density distribution
    F = np.ones((Ny, Nx, NL))

    # Initialize internal distribution
    G = np.ones((Ny, Nx, NL))

    # Initial temperature field
    T = np.ones((Ny, Nx))
    for y in range(1, Ny - 1):
        T[y, :] = T0 - dT * y / H

    # Initial pressure field
    P = np.ones((Ny, Nx))
    # Corrected vectorized initialization
    X_coords = np.arange(Nx)
    for y in range(1, Ny - 1):
        # perturbation should be across the horizontal length L
        P[y, :] = (1 + 0.5 * rho0 * beta * g0 * dT * y * (1 - y / H)) * (1 + 0.001 * np.cos(2 * np.pi * X_coords / Nx))

    # Initialize F and G to equilibrium distributions at t=0
    def get_Feq(rho, p, ux, uy):
        feq = np.zeros((Ny, Nx, NL))
        for i, (cx, cy, w) in enumerate(zip(cxs, cys, weights)):
            # Using the incompressible Feq from the paper's appendix
            feq[:, :, i] = rho * w * (3 * p / rho + 3 * (cx * ux + cy * uy) + 4.5 * (cx * ux + cy * uy) ** 2 - 1.5 * (
                        ux ** 2 + uy ** 2))
        return feq

    def get_Geq(rho, t, ux, uy):
        geq = np.zeros((Ny, Nx, NL))
        for i, (cx, cy, w) in enumerate(zip(cxs, cys, weights)):
            c2 = cx ** 2 + cy ** 2
            u2 = ux ** 2 + uy ** 2
            # Using the Geq formula from Eq (45) / (40)
            geq[:, :, i] = rho * R * t * w * (
                        1.5 * c2 + 4.5 * c2 * (cx * ux + cy * uy) - 3 * (cx * ux + cy * uy) + 4.5 * (
                            cx * ux + cy * uy) ** 2 - 1.5 * u2)
        return geq

    # Inside LBM function, replace F = np.ones and G = np.ones with:
    F = get_Feq(rho0, P, 0, 0)  # Initial velocity is zero
    G = get_Geq(rho0, T, 0, 0)

    if show_animation:
        # [AI] : Visualization
        # Before the loop starts, initialize the plot and colorbar
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(np.zeros((Ny - 2, Nx)), cmap="magma", aspect='auto',
                       origin='lower')  # Using magma for better visibility
        cbar = fig.colorbar(im, ax=ax, label='Velocity $u_x$')

    # Simulation Main Loop
    for it in range(Nt):
        print(it)

        ux = np.sum(F * cxs, 2) / rho0
        uy = np.sum(F * cys, 2) / rho0 + 0.5 * beta * g0 * (T - T_m)

        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho0*w*(3*P/rho0 + 3*(cx*ux + cy*uy) + 4.5*(cx*ux + cy*uy)**2 - 1.5*(
                        ux**2 + uy**2))

        P = np.sum(Feq, 2) / 3

        Force = np.zeros(F.shape)
        for i, cx, cy in zip(idxs, cxs, cys):
            vy_rel = cy - uy
            Force[:, :, i] = 3 * (beta * g0 * (T - T_m)) * vy_rel * Feq[:, :, i]



        # Corrected collision update for F
        F += -(F - Feq) / (tau_v + 0.5) + (tau_v * Force) / (tau_v + 0.5)

        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Compute unknown f2, f5 and f6 at Bottom
        F[1, :, 2] = F[1, :, 4]
        F[1, :, 5] = F[1, :, 7] - 0.5 * (F[1, :, 1] - F[1, :, 3]) + 0.5 * rho0 * beta * g0 * (T[1, :] - T_m)
        F[1, :, 6] = F[1, :, 8] + 0.5 * (F[1, :, 1] - F[1, :, 3]) + 0.5 * rho0 * beta * g0 * (T[1, :] - T_m)

        # Compute unknown f4, f7 and f8 at Top
        F[-2, :, 4] = F[-2, :, 2]
        F[-2, :, 7] = F[-2, :, 5] + 0.5 * (F[-2, :, 1] - F[-2, :, 3]) - 0.5 * rho0 * beta * g0 * (T[-2, :] - T_m)
        F[-2, :, 8] = F[-2, :, 6] - 0.5 * (F[-2, :, 1] - F[-2, :, 3]) - 0.5 * rho0 * beta * g0 * (T[-2, :] - T_m)

        # [4] Compute macroscopic variables using for G distribution
        f = (tau_v * F + 0.5 * Feq + 0.5 * tau_v * Force) / (tau_v + 0.5)

        dux_dx = 0.5 * (np.roll(ux, 1, 1) - np.roll(ux, -1, 1))

        dux_dy = np.zeros_like(ux)
        dux_dy[2:-2, :] = (ux[3:-1, :] - ux[1:-3, :]) / 2.0
        dux_dy[1, :] = (ux[2, :] - ux[1, :])
        dux_dy[-2, :] = (ux[-2, :] - ux[-3, :])

        duy_dx = 0.5 * (np.roll(uy, 1, 1) - np.roll(uy, -1, 1))

        duy_dy = np.zeros_like(uy)
        duy_dy[2:-2, :] = (uy[3:-1, :] - uy[1:-3, :]) / 2.0
        duy_dy[1, :] = (uy[2, :] - uy[1, :])
        duy_dy[-2, :] = (uy[-2, :] - uy[-3, :])

        # PI
        Txx = (tau_v / 3) * rho0 * (2 * dux_dx)
        Txy = (tau_v / 3) * rho0 * (duy_dx + dux_dy)
        Tyy = (tau_v / 3) * rho0 * (2 * duy_dy)

        # divergence PI
        dTxx_dx = 0.5 * (np.roll(Txx, 1, 1) - np.roll(Txx, -1, 1))
        dTxy_dx = 0.5 * (np.roll(Txy, 1, 1) - np.roll(Txy, -1, 1))

        dTxy_dy = np.zeros_like(Txy)
        dTxy_dy[2:-2, :] = (Txy[3:-1, :] - Txy[1:-3, :]) / 2.0
        dTxy_dy[1, :] = (Txy[2, :] - Txy[1, :])
        dTxy_dy[-2, :] = (Txy[-2, :] - Txy[-3, :])

        dTyy_dy = np.zeros_like(Tyy)
        dTyy_dy[2:-2, :] = (Tyy[3:-1, :] - Tyy[1:-3, :]) / 2.0
        dTyy_dy[1, :] = (Tyy[2, :] - Tyy[1, :])
        dTyy_dy[-2, :] = (Tyy[-2, :] - Tyy[-3, :])

        dP_dx = 0.5 * (np.roll(P, 1, 1) - np.roll(P, -1, 1))

        dP_dy = np.zeros_like(dP_dx)
        dP_dy[2:-2, :] = (P[3:-1, :] - P[1:-3, :]) / 2.0
        dP_dy[1, :] = (P[2, :] - P[1, :])
        dP_dy[-2, :] = (P[-2, :] - P[-3, :])

        q1 = np.zeros((Ny, Nx, NL))
        for i, cx, cy in zip(idxs, cxs, cys):
            vx_rel = cx - ux
            vy_rel = cy - uy

            q1[:, :, i] = (1 / rho0) * (vx_rel * (-dP_dx + dTxx_dx + dTxy_dy) + vy_rel * (-dP_dy + dTxy_dx + dTyy_dy))

        q = q1

        Q = 0.5 * np.sum(f * q, 2)

        # [4] Stream G distribution
        for i, cx, cy in zip(idxs, cxs, cys):
            G[:, :, i] = np.roll(G[:, :, i], cx, axis=1)
            G[:, :, i] = np.roll(G[:, :, i], cy, axis=0)

        T = (np.sum(G, 2) - Q) / (rho0 * R)

        # Fixed temperature at bottom
        T_bottom = np.zeros((1, Nx)) + T0
        # Fixed temperature at top
        T_top = np.zeros((1, Nx)) + T1

        # Correcting temperature at boundaries
        T[1, :] = T_bottom
        T[-2, :] = T_top

        # Add this after correcting T[1,:] and T[-2,:]
        T[0, :] = T[1, :]
        T[-1, :] = T[-2, :]
        P[0, :] = P[1, :]
        P[-1, :] = P[-2, :]

        # 5.1 Compute unknown G distribution at boundaries
        f_neq_4 = (1 / 2) * Tyy
        f_neq_7 = (1 / 8) * (Txx + 2 * Txy + Tyy)
        f_neq_8 = (1 / 8) * (Txx - 2 * Txy + Tyy)

        # Bottom
        G[1, :, 2] = -G[1, :, 4] + (1 / 3) * rho0 * R * T_bottom + 2 * f_neq_4[1, :]
        G[1, :, 5] = -G[1, :, 7] + (1 / 6) * rho0 * R * T_bottom + 4 * f_neq_7[1, :]
        G[1, :, 6] = -G[1, :, 8] + (1 / 6) * rho0 * R * T_bottom + 4 * f_neq_8[1, :]

        # Top
        G[-2, :, 4] = -G[-2, :, 2] + (1 / 3) * rho0 * R * T_top + 2 * f_neq_4[-2, :]
        G[-2, :, 7] = -G[-2, :, 5] + (1 / 6) * rho0 * R * T_top + 4 * f_neq_7[-2, :]
        G[-2, :, 8] = -G[-2, :, 6] + (1 / 6) * rho0 * R * T_top + 4 * f_neq_8[-2, :]

        print(T)

        if it == 10:
            exit()

        # [6] Calculate Geq
        Geq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            c2 = cx ** 2 + cy ** 2
            u2 = ux ** 2 + uy ** 2
            Geq[:, :, i] = rho0*R*T*w*(1.5*c2 + 4.5*c2*(cx*ux + cy*uy) - 3*(cx*ux + cy*uy) + 4.5*(cx*ux + cy*uy)**2 - 1.5*u2)

        # [7] Update next distributions
        G += -(G - Geq) / (tau_c + 0.5) - (tau_c / (tau_c + 0.5)) * (f * q)

    return 0

if __name__ == '__main__':
    LBM()
