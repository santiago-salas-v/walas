import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import ode

def dydx(y, t0, epsilon):
    dy0_dx = y[1]
    dy1_dx = ((1-y[0]**2)*y[1]-y[0])/epsilon
    return np.array([
        dy0_dx,
        dy1_dx
    ])

t_span = np.linspace(0, 2, 200)
y0 = np.array([2.0, 0.0])
# odeint takes f(y,t)
soln = odeint(lambda y, t: dydx(y, t, 1.0e-03), y0, t_span)

t0 = t_span[0]
t1 = t_span[-1]
dt = np.mean(np.diff(t_span))
# ode takes f(t, y) !! not f(y,t)
soln_oo = ode(
    lambda t, y, epsilon: dydx(y, t, epsilon)
).set_integrator('dopri5', atol=1.0e-03, rtol=1.0e-03)
soln_oo.set_initial_value(y0, t0).set_f_params(1.0e-03)
soln_oo_val = np.zeros([len(t_span),3])
i = 0
soln_oo_val[0, 0] = t0
soln_oo_val[0, 1] = y0[0]
soln_oo_val[0, 2] = y0[1]
while soln_oo.successful() and soln_oo.t < t1:
    i += 1
    print(
        soln_oo.t+dt,
        soln_oo.integrate(soln_oo.t + dt)
    )
    soln_oo_val[i, 0] = soln_oo.t+dt
    soln_oo_val[i, 1] = soln_oo.y[0]
    soln_oo_val[i, 2] = soln_oo.y[1]



ax = plt.subplot2grid([2,2], [0,0])
ax.plot(t_span, np.array(soln[:, 0]))
ax.set_xlabel('x')
ax.set_ylabel('y0')
ax = plt.subplot2grid([2,2], [0,1])
ax.plot(t_span, np.array(soln[:, 1]))
ax.set_xlabel('x')
ax.set_ylabel('y1')

ax = plt.subplot2grid([2,2], [1,0])
ax.plot(soln_oo_val[:, 0], soln_oo_val[:, 1])
ax = plt.subplot2grid([2,2], [1,1])
ax.plot(soln_oo_val[:, 0], soln_oo_val[:, 2])

plt.tight_layout()
plt.show()