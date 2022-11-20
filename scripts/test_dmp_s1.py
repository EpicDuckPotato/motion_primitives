#!/usr/bin/env python3

import numpy as np
from lie_group_dmp import *
from angle_mod import *
import rospy
import rospkg
from geometry_msgs.msg import Pose, PoseStamped, Point
from visualization_msgs.msg import Marker

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class S1Group(object):
  def __init__(self):
    self.nq = 1
    self.nv = 1

  def err_fn(self, q, v, qd, vd):
    q_err = np.array([angdiff(qd[0], q[0])])
    v_err = v - vd
    return q_err, v_err

  def neutral(self):
    return np.zeros(1)

  def inv(self, q):
    return -q

  def integrate(self, q, dq):
    return np.array([angle_mod(q[0] + dq, np.pi)])

  def clip_dq(self, q, dq):
    qmod = angle_mod(q[0], np.pi)
    if qmod + dq > np.pi:
      return (np.pi - qmod - 1e-5)*np.ones(self.nv)
    elif qmod + dq < -np.pi:
      return (-np.pi - qmod + 1e-5)*np.ones(self.nv)
    return dq

  def difference(self, q1, q2):
    return np.array([angdiff(q1[0], q2[0])])

  def dIntegrate(self, q, dq, first):
    return np.ones((1, 1))

  def dDifference(self, q1, q2, first):
    if first:
      return -np.ones((1, 1))
    else:
      return np.ones((1, 1))

  def adj(self, q, v):
    return np.ones((1, 1))

  def adj_inv(self, q, v):
    return np.ones((1, 1))

  def act(self, q1, q2):
    return self.integrate(q1, q2)
    
  def act_inv(self, q1, q2):
    return self.difference(q1, q2)

def demonstrate_q(t, T):
  # In xy plane
  x = 2*np.pi/T*t
  y = np.sin(x)

  xdot = 2*np.pi/T
  ydot = np.cos(x)*xdot

  q = np.zeros(1)
  q[0] = np.arctan2(ydot, xdot)
  return q

rospy.init_node('sim_node', anonymous=True)

rospack = rospkg.RosPack()
path = rospack.get_path('humrs_control')

lie_group = S1Group()

# PD gains
K = 100*np.eye(lie_group.nv) 
D = 2*np.sqrt(K)
num_psi = 10 # Number of window functions
alpha = 3 # Canonical system decay rate

dmp = DMP(lie_group, K, D, alpha, num_psi)

t_0_dem = 0
T_dem = 5
trj_len = 101
dt_dem = T_dem/(trj_len - 1)
t_list_dem = np.linspace(t_0_dem, T_dem, trj_len)

extended_t_list_dem = np.concatenate((t_list_dem, [t_list_dem[-1] + dt_dem, t_list_dem[-1] + 2*dt_dem]))
q_list_dem = [demonstrate_q(t, T_dem) for t in extended_t_list_dem]
v_list_dem = [lie_group.difference(q1, q2)/dt_dem for q1, q2 in zip(q_list_dem[:-1], q_list_dem[1:])]
vdot_list_dem = [(v2 - v1)/dt_dem for v1, v2 in zip(v_list_dem[:-1], v_list_dem[1:])]
q_list_dem = np.array(q_list_dem[:-2])
v_list_dem = np.array(v_list_dem[:-1])
vdot_list_dem = np.array(vdot_list_dem)

q_0_dem = q_list_dem[0]
v_0_dem = v_list_dem[0]
q_goal_dem = q_list_dem[-1]
v_goal_dem = np.zeros(lie_group.nv)
vdot_goal_dem = np.zeros(lie_group.nv)

w_trj = dmp.fit(q_list_dem, v_list_dem, vdot_list_dem, q_0_dem, q_goal_dem, v_goal_dem, vdot_goal_dem, T_dem)
np.save(path + '/scripts/s1_w_trj.npy', w_trj)

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 30})

plt.figure(figsize=(20, 15))
for i in range(num_psi):
  plt.plot(w_trj[:, i, 0]*180/np.pi, label='Force Field Center ' + str(i), linewidth=10)

plt.legend()
plt.xlabel('Gradient Descent Iteration', labelpad=100)
plt.ylabel('Force Field Location (deg)', labelpad=100)
ax = plt.gca()
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.2, bottom=0.4, top=0.8, right=0.7)
ax.set_title('Gradient Descent on S1', y=1.05)
ax.legend(bbox_to_anchor=(1.0, 1.05))
plt.savefig(path + '/scripts/gd_s1.png')
plt.show()

dmp_pub = rospy.Publisher('/dmp', Marker, queue_size=1)
dmp_marker = Marker()
dmp_marker.type = Marker.ARROW
dmp_marker.action = Marker.ADD
dmp_marker.pose.position.x = 0
dmp_marker.pose.position.y = 0
dmp_marker.pose.position.z = 0
dmp_marker.pose.orientation.x = 0
dmp_marker.pose.orientation.y = 0
dmp_marker.pose.orientation.z = 0
dmp_marker.pose.orientation.w = 1
dmp_marker.scale.x = 0.1
dmp_marker.scale.y = 0.2
dmp_marker.color.r = 1
dmp_marker.color.g = 0
dmp_marker.color.b = 0
dmp_marker.color.a = 1
dmp_marker.header.frame_id = "world"
dmp_marker.ns = "sim_node"
dmp_marker.points.append(Point())
dmp_marker.points[0].x = 0
dmp_marker.points[0].y = 0
dmp_marker.points[0].z = 0
dmp_marker.points.append(Point())

demonstration_pub = rospy.Publisher('/demonstration', Marker, queue_size=1)
demonstration_marker = Marker()
demonstration_marker.type = Marker.ARROW
demonstration_marker.action = Marker.ADD
demonstration_marker.pose.position.x = 0
demonstration_marker.pose.position.y = 0
demonstration_marker.pose.position.z = 0
demonstration_marker.pose.orientation.x = 0
demonstration_marker.pose.orientation.y = 0
demonstration_marker.pose.orientation.z = 0
demonstration_marker.pose.orientation.w = 1
demonstration_marker.scale.x = 0.1
demonstration_marker.scale.y = 0.2
demonstration_marker.color.r = 0
demonstration_marker.color.g = 1
demonstration_marker.color.b = 0
demonstration_marker.color.a = 1
demonstration_marker.header.frame_id = "world"
demonstration_marker.ns = "sim_node"
demonstration_marker.points.append(Point())
demonstration_marker.points[0].x = 0
demonstration_marker.points[0].y = 0
demonstration_marker.points[0].z = 0
demonstration_marker.points.append(Point())

input('Press enter to display planned trajectory')
rospy.sleep(5)

# Integrate, see how the system behaves
t_list, q_list, v_list = dmp.simulate(q_0_dem, v_0_dem, t_0_dem, 1, trj_len)
rate = rospy.Rate(1/dt_dem)
for t, q, v, q_dem, v_dem in zip(t_list, q_list, v_list, q_list_dem, v_list_dem):
  if rospy.is_shutdown():
    break

  dmp_marker.points[1].x = np.cos(q[0])
  dmp_marker.points[1].y = np.sin(q[0])
  dmp_marker.header.stamp = rospy.Time.now()
  dmp_pub.publish(dmp_marker)

  demonstration_marker.points[1].x = np.cos(q_dem[0])
  demonstration_marker.points[1].y = np.sin(q_dem[0])
  demonstration_marker.header.stamp = rospy.Time.now()
  demonstration_pub.publish(demonstration_marker)

  rate.sleep()
