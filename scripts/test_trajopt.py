import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import crocoddyl as croc
import rospkg
import itp_contact.itp_contact_bindings as itp
import motion_primitives.motion_primitives_bindings as mp
import numpy as np
import time

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

# print(croc.StdVec_VectorX)

rospack = rospkg.RosPack()
humrs_path = rospack.get_path('humrs_control')
urdf_file = humrs_path + '/../urdf/snake.urdf'
contact_info = itp.ContactInfo()

pin_model = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer()).model
trajopt = mp.Trajopt(pin_model, contact_info)
q_trj = croc.StdVec_VectorX()
num_q = 10
q_ref = croc.StdVec_VectorX()
for i in range(num_q):
  q = pin.neutral(pin_model)
  q[2] = i/(num_q - 1)
  q_ref.append(q)
qstart = pin.neutral(pin_model)
goal_pos = np.array([0, 1, 1], dtype=np.float64)
dt = 0.05
q_cost = 1.0
v_cost = 1.0
vdot_cost = 1.0
# vdot_cost = 0.0

bt = time.perf_counter()
solved = trajopt.optimize(q_trj, q_ref, qstart, goal_pos, dt, q_cost, v_cost, vdot_cost)
at = time.perf_counter()
print(solved)
for q in q_trj:
  print(q)
print('Time: ', at - bt)
