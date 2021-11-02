cmake_minimum_required(VERSION 2.8.3)
project(map_postprocess)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
)

catkin_package()
catkin_python_setup()
