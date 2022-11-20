import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class DMP(object):
  # CONSTRUCTOR
  # ARGUMENTS
  # K: proportional gain (nv x nv matrix)
  # D: derivative gain (nv x nv matrix)
  # alpha: decay rate of canonical system
  # num_psi: number of window functions
  def __init__(self, lie_group, K, D, alpha, num_psi):
    self.lie_group = lie_group
    self.K = np.copy(K)
    self.D = np.copy(D)
    self.alpha = alpha
    self.num_psi = num_psi

    ### Store the window functions ###

    # Where are the functions centered in time?
    t_centers = np.linspace(0, 1, self.num_psi)

    # Where are the functions centered in phase space?
    s_centers = np.exp(-self.alpha*t_centers)

    # Variances for the functions
    h = num_psi/(s_centers)**2

    self.psi = []
    for i, (c, h_i) in enumerate(zip(s_centers, h)):
      self.psi.append(lambda s, h_i=h_i, c=c : np.exp(-h_i*(s - c)**2))

    ### Properties determined by fitting to a demonstration. Updated using the "fit" function ###

    # Configuration, velocity, and acceleration goals
    self.q_goal = self.lie_group.neutral()
    self.v_goal = np.zeros(self.lie_group.nv)
    self.vdot_goal = np.zeros(self.lie_group.nv)

    # Locations of acceleration fields associated with the window functions,
    # relative to the initial configuration of the DMP. These are
    # determined using linear regression in the "fit" function
    self.w = np.array([self.lie_group.neutral() for i in range(self.num_psi)])

    # Duration of demonstration
    self.T_dem = 1

    # Initial configuration
    self.q_0 = self.lie_group.neutral()

  # transform: returns a transformation of the DMP
  # ARGUMENTS
  # q_goal: new goal
  # q_0: new starting point
  # RETURN: a new dmp object with the transformation applied
  def transform(self, q_goal, q_0):
    # New goal relative to new q_0
    q_goal_rel = self.lie_group.act_inv(q_0, q_goal)
    # Old goal relative to old q_0
    cur_q_goal_rel = self.lie_group.act_inv(self.q_0, self.q_goal)
    # Transformation applied to q_goal, relative to q_0
    dq_goal = self.lie_group.difference(cur_q_goal_rel, q_goal_rel)

    transformed_dmp = DMP(self.lie_group, self.K, self.D, self.alpha, self.num_psi)
    transformed_dmp.q_goal = q_goal
    transformed_dmp.v_goal = np.copy(self.v_goal)
    transformed_dmp.vdot_goal = np.copy(self.vdot_goal)
    transformed_dmp.w = np.array([self.lie_group.integrate(self.w[i], i*dq_goal/self.num_psi) for i in range(self.num_psi)])
    transformed_dmp.q_0 = q_0
    return transformed_dmp

  # get_window_functions: returns a list of window functions, which can be evaluated at any phase value s
  def get_window_functions(self):
    return self.psi

  # canonical_system: evaluates the canonical system s(t) at time t, assuming the 
  # initial condition s(t_0) = 1. Decay rate is self.alpha/T
  def canonical_system(self, t, T, t_0):
    return np.exp(-self.alpha/T*(t - t_0))

  # eval_window_functions: returns a trajectory of window function values
  # ARGUMENTS
  # trj_len: number of evaluations in trajectory
  # T: duration of trajectory (scales the canonical system in time)
  # t_0: initial time
  # RETURN
  # t_list: list of times
  # s_list: list of phase (canonical system) values
  # psi_matrix: trj_len x num_psi numpy array with the window function values
  def eval_window_functions(self, trj_len, T, t_0):
    t_list = np.linspace(0, T, trj_len)
    s_list = self.canonical_system(t_list, T, t_0)
    psi_lists = [psi_i(s_list) for psi_i in self.psi]
    psi_matrix = np.array(psi_lists).transpose()
    return t_list, s_list, psi_matrix

  # plot window_functions: plots the window functions, starting at t = 0,
  # ending at t = T_dem, evaluating at num_eval_points along the interval
  def plot_window_functions(self, tau, num_eval_points=101):
    t_list, _, psi_matrix = self.eval_window_functions(num_eval_points, tau*self.T_dem, 0)
    for i in range(self.num_psi):
      plt.plot(t_list, psi_matrix[:, i], label='psi_' + str(i))
    plt.xlabel('t (s)')
    plt.ylabel('Window Function Value')
    plt.title('Window Functions')
    plt.legend()
    plt.show()

  # get_linreg_eqn: gets LHS and RHS for linear regression. Assumes goals and q_0 have
  # already been populated
  # q_list: trj_len x nq numpy array, containing demonstration configuration trajectory
  # v_list: trj_len x nv numpy array, containing demonstration velocity trajectory
  # vdot_list: trj_len x nv numpy array, containing demonstration acceleration trajectory
  # psi_matrix: trj_len x num_psi numpy array, containing trajectory of window function values
  def get_linreg_eqn(self, q_list, v_list, vdot_list, s_list, psi_matrix):
    nq = self.lie_group.nq
    nv = self.lie_group.nv

    trj_len = len(q_list)

    # Linear equation consists of trj_len x num_psi blocks, where each block is nv x nv
    LHS = np.zeros((trj_len*nv, self.num_psi*nv))
    RHS = np.zeros((trj_len*nv))
    wds = []
    ws = []
    for blockrow, (q, v, vdot, s, psi_vals) in enumerate(zip(q_list, v_list, vdot_list, s_list, psi_matrix)):
      startrow = blockrow*nv
      endrow = (blockrow + 1)*nv
      sum_psi = sum(psi_vals)
      # q0inv*q
      rel_q = self.lie_group.act_inv(self.q_0, q)
      for blockcol, psi_val in enumerate(psi_vals):
        startcol = blockcol*nv
        endcol = (blockcol + 1)*nv
        # Derivative of (q0inv*q) - w_i wrt w_i
        ddiff = self.lie_group.dDifference(self.w[blockcol], rel_q, True)
        LHS[startrow:endrow, startcol:endcol] = -s*psi_val*self.K@ddiff/sum_psi

      q_err, v_err = self.lie_group.err_fn(q, v, self.q_goal, self.v_goal)
      # RHS[startrow:endrow] = vdot - self.vdot_goal + self.D@v_err + (1 - s)*self.K@q_err + s*self.K@self.lie_group.difference(self.q_0, q)
      RHS[startrow:endrow] = vdot - self.calc_tau_vdot(q, v, s)
      # wds.append(vdot[4])
      # ws.append(self.calc_tau_vdot(q, v, s)[4])

    # print(wds)
    # print(ws)
    # print(RHS[4::self.lie_group.nv])
    # print([R.from_quat(wi[3:7]).as_rotvec()[1] for wi in self.w])
    # print()

    return LHS, RHS

  def calc_tau_vdot(self, q, v, s):
    psi_vals = [psi_i(s) for psi_i in self.psi]
    q_err, v_err = self.lie_group.err_fn(q, v, self.q_goal, self.v_goal)
    # q0inv*q
    rel_q = self.lie_group.act_inv(self.q_0, q)
    return s*sum([-psi_val*self.K@self.lie_group.difference(w_i, rel_q) for psi_val, w_i in zip(psi_vals, self.w)])/sum(psi_vals) - (1 - s)*self.K@q_err - self.D@v_err + self.vdot_goal

  # simulate: simulates the DMP for a specified number of steps, with the given time scaling
  # ARGUMENTS
  # q_0, v_0, t_0: initial condition
  # tau: scales time, e.g. tau = 2 makes time run 2x slower
  # trj_len: returned trajectories will contain this many states, including initial state
  def simulate(self, q_0, v_0, t_0, tau, trj_len):
    T = tau*self.T_dem # Trajectory duration, given that the demonstration duration is being scaled by tau
    t_list = np.linspace(t_0, t_0 + T, trj_len)
    dt = t_list[1] - t_list[0]
    q_list = [q_0]
    v_list = [v_0]
    for t in t_list[:-1]:
      s = self.canonical_system(t, T, t_0)
      vdot = self.calc_tau_vdot(q_list[-1], v_list[-1], s)/tau
      v_list.append(v_list[-1] + vdot*dt)
      q_list.append(self.lie_group.integrate(q_list[-1], v_list[-1]/tau*dt))

    return t_list, np.array(q_list), np.array(v_list)

  # simulate_with_obstacles: simulates the DMP for a specified number of steps, with the given time scaling, avoiding
  # the given obstacle list (assumes 2D, for now)
  # ARGUMENTS
  # q_0, v_0, t_0: initial condition
  # tau: scales time, e.g. tau = 2 makes time run 2x slower
  # trj_len: returned trajectories will contain this many states, including initial state
  # o_list: num_obstacles x 2 
  def simulate_with_obstacles(self, q_0, v_0, t_0, tau, trj_len, o_list):
    T = tau*self.T_dem # Trajectory duration, given that the demonstration duration is being scaled by tau
    t_list = np.linspace(t_0, t_0 + T, trj_len + 1)
    dt = t_list[1] - t_list[0]
    q_list = [q_0]
    v_list = [v_0]
    for t in t_list[:-1]:
      s = self.canonical_system(t, T, t_0)
      vdot = self.calc_tau_vdot(q_list[-1], v_list[-1], s)/tau
      gamma = 100
      beta = 20/np.pi
      for o in o_list:
        q = q_list[-1]
        v = v_list[-1]
        if np.linalg.norm(v) < 1e-5:
          break

        b1 = o - q
        b1 /= np.linalg.norm(b1)
        b2 = v - v@b1*b1
        b2 /= np.linalg.norm(b2)
        phi = np.arctan2(v@b2, v@b1)
        phidot = gamma*phi*np.exp(-beta*np.abs(phi))
        rmat = np.array([[0, -1], [1, 0]])
        vplane = np.linalg.norm(v)*np.array([np.cos(phi), np.sin(phi)])
        vdot_plane = rmat@vplane*phidot
        vdot += vdot_plane[0]*b1 + vdot_plane[1]*b2

      v_list.append(v_list[-1] + vdot*dt)
      q_list.append(self.lie_group.integrate(q_list[-1], v_list[-1]/tau*dt))

    return t_list, np.array(q_list), np.array(v_list)

  def check_approx_error(self, q_list, v_list, vdot_list, t_0):
    trj_len = len(q_list)
    t_list = np.linspace(t_0, t_0 + self.T_dem, trj_len)
    for t, q, v, vdot in zip(t_list, q_list, v_list, vdot_list):
      s = self.canonical_system(t, self.T_dem, t_0)
      vdot_approx = self.calc_tau_vdot(q, v, s)
      print(vdot - vdot_approx)

  def fit(self, q_list, v_list, vdot_list, q_0, q_goal, v_goal, vdot_goal, T):
    self.q_0 = np.copy(q_0)
    self.q_goal = np.copy(q_goal)
    self.v_goal = np.copy(v_goal)
    self.vdot_goal = np.copy(vdot_goal)
    self.T_dem = T

    trj_len = len(q_list)

    t_list, s_list, psi_matrix = self.eval_window_functions(trj_len, self.T_dem, 0)
    max_iter = 1000

    w_trj = [np.copy(self.w)]
    for it in range(max_iter):
      LHS, RHS = self.get_linreg_eqn(q_list, v_list, vdot_list, s_list, psi_matrix)
      dw_flat, _, _, _ = np.linalg.lstsq(LHS, RHS)
      dw = 0.01*dw_flat.reshape(self.num_psi, self.lie_group.nv)
      for i in range(self.num_psi):
        dw[i] = self.lie_group.clip_dq(self.w[i], dw[i])
        self.w[i] = self.lie_group.integrate(self.w[i], dw[i])
      w_trj.append(np.copy(self.w))

      if np.linalg.norm(dw, ord=np.inf) < 1e-4:
        break

    return np.array(w_trj)

    # wflat, _, _, _ = np.linalg.lstsq(LHS, RHS)
    # self.w = wflat.reshape(self.num_psi, self.lie_group.nq)
