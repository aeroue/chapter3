# common/dynamics.py

import numpy as np
from scipy.linalg import expm

class CWDynamics:
    """
    Implements the 2D Clohessy-Wiltshire (CW) dynamics for spacecraft relative motion.
    The state vector is [x, y, vx, vy].
    The control input is [ax, ay].
    Uses Zero-Order Hold (ZOH) for accurate discretization.
    """

    def __init__(self, n: float, ts: float):
        """
        Initializes the CW dynamics model.

        Args:
            n (float): Mean angular velocity of the reference orbit (n = sqrt(mu/a^3)).
            ts (float): Sampling time for discretization.
        """
        if not isinstance(n, (int, float)) or n < 0:
            raise ValueError("Mean angular velocity 'n' must be a non-negative number.")
        if not isinstance(ts, (int, float)) or ts <= 0:
            raise ValueError("Sampling time 'ts' must be a positive number.")

        self.n = n
        self.ts = ts

        self.A_cont = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [3 * self.n**2, 0, 0, 2 * self.n],
            [0, 0, -2 * self.n, 0]
        ])
        self.state_dim = self.A_cont.shape[0]

        self.B_cont = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        self.control_dim = self.B_cont.shape[1]

        M_upper = np.concatenate((self.A_cont, self.B_cont), axis=1)
        M_lower = np.zeros((self.control_dim, self.state_dim + self.control_dim))
        M = np.concatenate((M_upper, M_lower), axis=0)

        try:
            phi = expm(M * self.ts)
            self.Ad = phi[:self.state_dim, :self.state_dim]
            self.Bd = phi[:self.state_dim, self.state_dim:]
        except Exception as e:
            print(f"Error during matrix exponentiation for ZOH: {e}")
            print("Falling back to Euler discretization for dynamics.")
            self.Ad = np.eye(self.state_dim) + self.A_cont * self.ts
            self.Bd = self.B_cont * self.ts


    def step(self, current_state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """
        Propagates the state one step forward using the discretized dynamics.
        """
        if not isinstance(current_state, np.ndarray):
            current_state = np.array(current_state, dtype=float)
        if not isinstance(control_input, np.ndarray):
            control_input = np.array(control_input, dtype=float)

        if current_state.shape != (self.state_dim,):
            raise ValueError(f"Current state shape must be ({self.state_dim},), got {current_state.shape}")
        if control_input.shape != (self.control_dim,):
            raise ValueError(f"Control input shape must be ({self.control_dim},), got {control_input.shape}")

        next_state = self.Ad @ current_state + self.Bd @ control_input
        return next_state

    def get_discrete_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.Ad.copy(), self.Bd.copy()

    def get_continuous_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.A_cont.copy(), self.B_cont.copy()

if __name__ == '__main__':
    mu_earth = 3.986004418e14
    a_ref = 7000e3
    n_orbital = np.sqrt(mu_earth / a_ref**3)
    sampling_time = 1.0

    cw_model = CWDynamics(n=n_orbital, ts=sampling_time)
    initial_state_2d = np.array([100.0, 50.0, 1.0, -0.5])
    control_thrust_2d = np.array([0.01, -0.005])

    print(f"Orbital n: {cw_model.n:.6e} rad/s, Ts: {cw_model.ts} s")
    Ad, Bd = cw_model.get_discrete_matrices()
    print("\nAd (ZOH):\n", np.round(Ad, 6))
    print("\nBd (ZOH):\n", np.round(Bd, 6))
    next_state_2d = cw_model.step(initial_state_2d, control_thrust_2d)
    print(f"\nNext State: {np.round(next_state_2d, 6)}")