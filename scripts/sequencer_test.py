#!/usr/bin/env python3

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import crocoddyl as croc
import hppfcl
import rospkg
import motion_primitives.motion_primitives_bindings as mp
import numpy as np
import time

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

rospy.init_node('test_trajopt', anonymous=True)

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

rospack = rospkg.RosPack()
humrs_path = rospack.get_path('humrs_control')
urdf_file = humrs_path + '/../urdf/snake.urdf'

pin_model = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer()).model
pin_data = pin.Data(pin_model)

pin_geom = pin.buildGeomFromUrdf(pin_model, urdf_file, './', pin.GeometryType.COLLISION)

# Populate obstacles
max_rads = [0.059 for j in range(pin_model.njoints)]

pin.forwardKinematics(pin_model, pin_data, pin.neutral(pin_model))
tail_fid = pin_model.getFrameId('tail_rear_propeller')
pin.updateFramePlacement(pin_model, pin_data, tail_fid)
link_primitive_idx = []
aggregate_geom_ids = []
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

  aggregate_geom_ids.append(pin_geom.ngeoms)
  pin_geom.addGeometryObject(pin.GeometryObject('cyl' + str(j), jfid, j, hppfcl.Cylinder(link_rad, link_length), placement))

universe_jid = pin_model.getJointId('universe')
universe_fid = pin_model.getFrameId('universe')

boxes = [np.array([-0.2, -0.1, 0.5, 0.9, 0.9, 0.9])]
# boxes = []

for b, box in enumerate(boxes):
  placement = pin.SE3(np.eye(3), box[:3])
  box_frame = pin.Frame('box_link' + str(b), universe_jid, universe_fid, placement, pin.FrameType.BODY)
  box_frame_id = pin_model.addFrame(box_frame)

  placement = pin.SE3(np.eye(3), box[:3])
  collision_geometry = hppfcl.Box(box[3], box[4], box[5])
  box_id = pin_geom.ngeoms
  pin_geom.addGeometryObject(pin.GeometryObject('box' + str(b), universe_fid, universe_jid, collision_geometry, placement))
  for geom_id in aggregate_geom_ids:
    pin_geom.addCollisionPair(pin.CollisionPair(box_id, geom_id))

library = mp.PrimitiveLibrary()

dt = 0.05
thetas = np.pi/4*np.arange(8)
for theta in thetas:
  primitive = mp.Primitive()
  dq = np.zeros(pin_model.nv)
  dq[2] = dt
  dq[3:6] = np.array([np.cos(theta), np.sin(theta), 0])*dt
  q = pin.neutral(pin_model)
  primitive.append(np.copy(q))
  for step in range(10):
    q = pin.integrate(pin_model, q, dq)
    primitive.append(np.copy(q))

  library.append(primitive)

primitive = mp.Primitive()
dq = np.zeros(pin_model.nv)
dq[2] = dt
q = pin.neutral(pin_model)
primitive.append(np.copy(q))
for step in range(10):
  q = pin.integrate(pin_model, q, dq)
  primitive.append(np.copy(q))

library.append(primitive)

sequencer = mp.PrimitiveSequencer(library)

r3_path = mp.R3Path()
r3_path.append(np.zeros(3))
r3_path.append(np.array([0.5, 0.0, 0.5]))
r3_path.append(np.array([1.0, 0.0, 1.0]))

solution = mp.Primitive()
q0 = pin.neutral(pin_model)

sequencer.sequence(solution, r3_path, q0)

for q in solution:
  print(q)
