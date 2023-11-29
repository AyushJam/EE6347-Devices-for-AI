import numpy as np

t_program = 1.5
V_on = 3
dt = 0.001
V1_set = np.concatenate(
    (
        np.linspace(0, V_on, int(t_program / 3 / dt)),
        np.ones(int(t_program / dt)) * V_on,
        np.linspace(V_on, 0, int(t_program / 3 / dt)),
    )
)
V2_reset = np.concatenate(
    (
        np.linspace(0, -V_on, int(t_program / 3 / dt)),
        np.ones(int(t_program / dt)) * -V_on,
        np.linspace(-V_on, 0, int(t_program / 3 / dt)),
    )
)
V_program = np.concatenate(
    (V1_set, np.zeros(int(t_program / dt)), V2_reset)
)  # Programming voltage
