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

# print(croc.StdVec_VectorX)

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

trajopt = mp.Trajopt(pin_model, pin_geom)
q_trj = croc.StdVec_VectorX()
num_q = 100
# num_q = 3
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

bt = time.perf_counter()
solved = trajopt.optimize(q_trj, q_ref, qstart, goal_pos, dt, q_cost, v_cost, vdot_cost)
at = time.perf_counter()
print(solved)
'''
for q in q_trj:
  print(q)
'''
print('Time: ', at - bt)

br = TransformBroadcaster()

head_tf = TransformStamped()
head_tf.header.frame_id = 'world'
head_tf.child_frame_id = 'head'
head_tf.transform.rotation.w = 1

joints_pub = rospy.Publisher('/humrs/fbk/joint_state', JointState, queue_size=1)
joints_msg = JointState()
for j in range(2, pin_model.njoints):
  joints_msg.name.append(pin_model.names[j])
  joints_msg.position.append(0)
  joints_msg.velocity.append(0)
  joints_msg.effort.append(0)

boxes_pub = rospy.Publisher('/humrs/boxes', MarkerArray, queue_size=1)
boxes_array = MarkerArray()
for box in boxes:
  box_marker = Marker()
  box_marker.header.frame_id = 'world'
  box_marker.ns = 'test_trajopt'
  box_marker.action = Marker.ADD
  box_marker.pose.position.x = box[0]
  box_marker.pose.position.y = box[1]
  box_marker.pose.position.z = box[2]
  box_marker.pose.orientation.w = 1

  box_marker.type = Marker.CUBE
  box_marker.scale.x = box[3]
  box_marker.scale.y = box[4]
  box_marker.scale.z = box[5]

  color = [0.5, 0.5, 0.5]
  box_marker.color.b = color[0]
  box_marker.color.g = color[1]
  box_marker.color.r = color[2]
  box_marker.color.a = 0.5
  box_marker.id = len(boxes_array.markers)
  boxes_array.markers.append(box_marker)

rate = rospy.Rate(1/dt)
for rep in range(100):
  # for q in q_ref:
  for q in q_trj:
    print(q)
    if rospy.is_shutdown():
      break

    head_tf.transform.translation.x = q[0]
    head_tf.transform.translation.y = q[1]
    head_tf.transform.translation.z = q[2]

    head_tf.transform.rotation.x = q[3]
    head_tf.transform.rotation.y = q[4]
    head_tf.transform.rotation.z = q[5]
    head_tf.transform.rotation.w = q[6]

    head_tf.header.stamp = rospy.Time.now()
    br.sendTransform(head_tf)

    joints_msg.position = [qi for qi in q[7:]]
    joints_msg.header.stamp = rospy.Time.now()
    joints_pub.publish(joints_msg)

    boxes_pub.publish(boxes_array)

    rate.sleep()
