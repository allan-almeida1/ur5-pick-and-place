# UR5 Pick and Place

<b>Author:</b> Allan Almeida

## Overview

This project implements a pick and place application with a UR5 robot arm. 
It uses Python and ZMQRemote API to communicate with the robot on CoppeliaSim.

Forward and inverse kinematics are implemented in Python. The robot is controlled
by sending joint velocities to the robot using a quintic polynomial trajectory.
The robot is able to pick up a cup from a table
and place it on another table.

The computer vision part of the project is implemented via a CNN. The CNN is trained
to detect the cup position on the image. The CNN is implemented in Tensorflow and Keras, 
and it uses a VGG16 pre-trained model as a base. The CNN is modified to output the cup position
on the image, and it is trained with a dataset of 5000 images.
