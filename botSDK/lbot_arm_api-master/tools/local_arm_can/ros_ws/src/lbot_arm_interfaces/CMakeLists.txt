cmake_minimum_required(VERSION 3.8)
project(lbot_arm_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmEffort.msg"
  "msg/ArmState.msg"
  "msg/JointTemperature.msg"
  "msg/RobotRunState.msg"
  "srv/FK.srv"
  "srv/IK.srv"
  "srv/FollowP.srv"
  "srv/GetJointLimit.srv"
  "srv/MoveC.srv"
  "srv/MoveJ.srv"
  "srv/MoveJP.srv"
  "srv/MoveL.srv"
  DEPENDENCIES geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
