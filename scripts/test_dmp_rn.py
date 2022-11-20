import numpy as np
import matplotlib.pyplot as plt
from lie_group_dmp import *

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class RnGroup(object):
  def __init__(self, n):
    self.n = n
    self.nq = n
    self.nv = n

  def err_fn(self, q, v, qd, vd):
    return q - qd, v - vd

  def neutral(self):
    return np.zeros(self.n)

  def inv(self, q):
    return -q

  def integrate(self, q, dq):
    return q + dq

  def clip_dq(self, q, dq):
    return dq

  def difference(self, q1, q2):
    return q2 - q1

  def dIntegrate(self, q, dq, first):
    if first:
      return np.eye(self.nv)
    else:
      return np.eye(self.nv)

  def dDifference(self, q1, q2, first):
    if first:
      return -np.eye(self.nv)
    else:
      return np.eye(self.nv)

  def adj(self, q, v):
    return np.eye(self.n)

  def adj_inv(self, q, v):
    return np.eye(self.n)

  def act(self, q1, q2):
    return q1 + q2

  def act_inv(self, q1, q2):
    return q2 - q1

def demonstrate_sine(t, T):
  x = 2*np.pi/T*t
  y = np.sin(x)
  q = np.array([x, y])

  xdot = 2*np.pi/T
  ydot = np.cos(x)*xdot
  v = np.array([xdot, ydot])

  xddot = 0
  yddot = -np.sin(x)*xdot**2 + np.cos(x)*xddot
  vdot = np.array([xddot, yddot])

  return q, v, vdot

def demonstrate_line(t, T):
  x = 2*np.pi/T*t
  y = x
  q = np.array([x, y])

  xdot = 2*np.pi/T
  ydot = xdot
  v = np.array([xdot, ydot])

  xddot = 0
  yddot = xddot
  vdot = np.array([xddot, yddot])

  return q, v, vdot

if __name__ == "__main__":
  lie_group = RnGroup(2)

  ### Initialize DMP ###
  # PD gains
  K = 100*np.eye(lie_group.nv) 
  D = 2*np.sqrt(K)
  num_psi = 10 # Number of window functions
  alpha = 3 # Canonical system decay rate

  dmp = DMP(lie_group, K, D, alpha, num_psi)

  ### Demonstration parameters ###
  t_0_dem = 0 # Start time (s)
  T_dem = 1 # Duration (s)
  trj_len = 101
  dt_dem = T_dem/(trj_len - 1)

  ### Tests ###
  # Plot window functions over the demonstration interval, then 5 times the demonstration interval, just to see if they scale
  dmp.plot_window_functions(1)
  dmp.plot_window_functions(5)

  # Try tracking a sine curve while reaching a goal. Configuration consists of (x, y)
  t_list_dem = np.linspace(t_0_dem, T_dem, trj_len)

  q_list_dem = []
  v_list_dem = []
  vdot_list_dem = []
  for t in t_list_dem:
    q_dem, v_dem, vdot_dem = demonstrate_sine(t, T_dem)
    # q_dem, v_dem, vdot_dem = demonstrate_line(t, T_dem)
    q_list_dem.append(q_dem)
    v_list_dem.append(v_dem)
    vdot_list_dem.append(vdot_dem)
  q_list_dem = np.array(q_list_dem)
  v_list_dem = np.array(v_list_dem)
  vdot_list_dem = np.array(vdot_list_dem)

  q_0_dem = q_list_dem[0]
  v_0_dem = v_list_dem[0]
  q_goal_dem = q_list_dem[-1]
  v_goal_dem = np.zeros(lie_group.nv)
  vdot_goal_dem = np.zeros(lie_group.nv)

  dmp.fit(q_list_dem, v_list_dem, vdot_list_dem, q_0_dem, q_goal_dem, v_goal_dem, vdot_goal_dem, T_dem)

  dmp.check_approx_error(q_list_dem, v_list_dem, vdot_list_dem, t_0_dem)

  ### Tests ###
  # Integrate, see how the system behaves
  t_list, q_list, v_list = dmp.simulate(q_0_dem, v_0_dem, t_0_dem, 1, trj_len)
  plt.plot(q_list[:, 0], q_list[:, 1], label='Imitation')
  plt.plot(q_list_dem[:, 0], q_list_dem[:, 1], label='Demonstration')
  plt.scatter([q_goal_dem[0]], [q_goal_dem[1]], s=100, label='Goal')
  plt.legend()
  plt.xlabel('x (m)')
  plt.ylabel('y (m)')
  plt.title('Imitation of Demonstrated Trajectory')
  plt.show()

  # Change the goal and integrate
  q_goal = np.array([4*np.pi, 0])
  transformed_dmp = dmp.transform(q_goal, q_0_dem)
  t_list, q_list, v_list = transformed_dmp.simulate(q_0_dem, v_0_dem, t_0_dem, 1, trj_len)
  plt.plot(q_list[:, 0], q_list[:, 1], label='DMP')
  plt.plot(q_list_dem[:, 0], q_list_dem[:, 1], label='Demonstration')
  plt.scatter([q_goal[0]], [q_goal[1]], s=100, label='Goal')
  plt.legend()
  plt.xlabel('x (m)')
  plt.ylabel('y (m)')
  plt.title('DMP with New Goal')
  plt.show()

  # Change the goal and integrate
  q_goal = np.array([2*np.pi, 1])
  transformed_dmp = dmp.transform(q_goal, q_0_dem)
  t_list, q_list, v_list = transformed_dmp.simulate(q_0_dem, v_0_dem, t_0_dem, 1, trj_len)
  plt.plot(q_list[:, 0], q_list[:, 1], label='DMP')
  plt.plot(q_list_dem[:, 0], q_list_dem[:, 1], label='Demonstration')
  plt.scatter([q_goal[0]], [q_goal[1]], s=100, label='Goal')
  plt.legend()
  plt.xlabel('x (m)')
  plt.ylabel('y (m)')
  plt.title('DMP with New Goal')
  plt.show()

  # Apply a translation to the whole system and integrate to verify translation invariance
  delta_q = np.ones(2)
  q_0 = q_0_dem + delta_q
  q_goal = q_goal_dem + delta_q
  transformed_dmp = dmp.transform(q_goal, q_0)
  t_list, q_list, v_list = transformed_dmp.simulate(q_0, v_0_dem, t_0_dem, 1, trj_len)
  plt.plot(q_list[:, 0], q_list[:, 1], label='DMP')
  plt.plot(q_list_dem[:, 0], q_list_dem[:, 1], label='Demonstration')
  plt.scatter([q_goal[0]], [q_goal[1]], s=100, label='Goal')
  plt.legend()
  plt.xlabel('x (m)')
  plt.ylabel('y (m)')
  plt.title('Translated DMP')
  plt.show()

  # Change the starting time and integrate to verify time-invariance
  t_list, q_list, v_list = dmp.simulate(q_0_dem, v_0_dem, 5, 1, trj_len)
  plt.plot(q_list[:, 0], q_list[:, 1], label='DMP')
  plt.plot(q_list_dem[:, 0], q_list_dem[:, 1], label='Demonstration')
  plt.scatter([q_goal_dem[0]], [q_goal_dem[1]], s=100, label='Goal')
  plt.legend()
  plt.xlabel('x (m)')
  plt.ylabel('y (m)')
  plt.title('Imitating the Demonstration with a Different Start Time')
  plt.show()

  plt.plot(t_list, q_list[:, 0], label='DMP')
  plt.plot(t_list_dem, q_list_dem[:, 0], label='Demonstration')
  plt.legend()
  plt.xlabel('t (s)')
  plt.ylabel('x (m)')
  plt.title('Imitating the Demonstration with a Different Start Time: x-axis Behavior')
  plt.show()

  plt.plot(t_list, q_list[:, 1], label='DMP')
  plt.plot(t_list_dem, q_list_dem[:, 1], label='Demonstration')
  plt.legend()
  plt.xlabel('t (s)')
  plt.ylabel('y (m)')
  plt.title('Imitating the Demonstration with a Different Start Time: y-axis Behavior')
  plt.show()

  # Slow down time and integrate
  tau = 2
  t_list, q_list, v_list = dmp.simulate(q_0_dem, v_0_dem, t_0_dem, tau, trj_len)
  plt.plot(q_list[:, 0], q_list[:, 1], label='DMP')
  plt.plot(q_list_dem[:, 0], q_list_dem[:, 1], label='Demonstration')
  plt.scatter([q_goal_dem[0]], [q_goal_dem[1]], s=100, label='Goal')
  plt.legend()
  plt.xlabel('x (m)')
  plt.ylabel('y (m)')
  plt.title('DMP with Slower Time Scale')
  plt.show()

  plt.plot(t_list, q_list[:, 0], label='DMP')
  plt.plot(t_list_dem, q_list_dem[:, 0], label='Demonstration')
  plt.legend()
  plt.xlabel('t (s)')
  plt.ylabel('x (m)')
  plt.title('DMP with Slower Time Scale: x-axis Behavior')
  plt.show()

  plt.plot(t_list, q_list[:, 1], label='DMP')
  plt.plot(t_list_dem, q_list_dem[:, 1], label='Demonstration')
  plt.legend()
  plt.xlabel('t (s)')
  plt.ylabel('y (m)')
  plt.title('DMP with Slower Time Scale: y-axis Behavior')
  plt.show()

  # Add an obstacle
  o_list = [np.array([np.pi, 0])]
  t_list, q_list, v_list = dmp.simulate_with_obstacles(q_0_dem, v_0_dem, t_0_dem, 1, trj_len, o_list)
  plt.plot(q_list[:, 0], q_list[:, 1], label='Imitation')
  plt.plot(q_list_dem[:, 0], q_list_dem[:, 1], label='Demonstration')
  plt.scatter([q_goal_dem[0]], [q_goal_dem[1]], s=100, label='Goal')
  plt.scatter([o_list[0][0]], [o_list[0][1]], s=100, label='Obstacle')
  plt.legend()
  plt.xlabel('x (m)')
  plt.ylabel('y (m)')
  plt.title('DMP with Obstacle')
  plt.show()
