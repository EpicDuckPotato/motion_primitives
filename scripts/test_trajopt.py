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

pin_model = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer()).model
pin_data = pin.Data(pin_model)

# Populate obstacles
max_rads = [0.059 for j in range(pin_model.njoints)]

contact_info = itp.ContactInfo()

pin.forwardKinematics(pin_model, pin_data, pin.neutral(pin_model))
tail_fid = pin_model.getFrameId('tail_rear_propeller')
pin.updateFramePlacement(pin_model, pin_data, tail_fid)
link_primitive_idx = []
for j, link_rad in enumerate(max_rads):
  if j == 0:
    # We aren't aggregating things connected to the universe
    continue

  if j == pin_model.njoints - 1:
    link_length = pin_data.oMi[j].translation[2] - pin_data.oMf[tail_fid].translation[2]
  else:
    link_length = pin_data.oMi[j].translation[2] - pin_data.oMi[j + 1].translation[2]

  # Aggregates the links into cylinders
  link_translation = np.array([0, 0, -link_length/2])
  link_rmat = np.eye(3)
  placement = pin.SE3(link_rmat, link_translation)

  jfid = pin_model.getFrameId(pin_model.names[j])

  link_frame = pin.Frame('ag_link' + str(j), j, jfid, placement, pin.FrameType.BODY)
  link_frame_id = pin_model.addFrame(link_frame)

  link_primitive_idx.append(contact_info.add_primitive(itp.Cylinder(link_rad, link_length), link_frame_id))

universe_jid = pin_model.getJointId('universe')
universe_fid = pin_model.getFrameId('universe')

boxes = [np.array([0.1, 0.1, 0.5, 0.9, 0.9, 0.9])]

for b, box in enumerate(boxes):
  placement = pin.SE3(np.eye(3), box[:3])
  box_frame = pin.Frame('box_link' + str(b), universe_jid, universe_fid, placement, pin.FrameType.BODY)
  box_frame_id = pin_model.addFrame(box_frame)

  box_idx = contact_info.add_primitive(itp.Box(box[3], box[4], box[5]), box_frame_id)

  for idx in link_primitive_idx:
    contact_info.add_contact_pair(idx, box_idx)
    print('here')


trajopt = mp.Trajopt(pin_model, contact_info)
q_trj = croc.StdVec_VectorX()
num_q = 3
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
