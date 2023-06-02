# UR5 Pick and Place

<b>Author:</b> <a href="https://orcid.org/my-orcid?orcid=0009-0006-2253-4195" target="_blank">Allan Almeida</a>

## Overview

This project implements a pick and place application with a UR5 robot arm. 
It uses Python and ZMQRemote API to communicate with the robot on CoppeliaSim.

Forward and inverse kinematics are implemented in Python. The robot is controlled
by sending joint velocities to the robot using a quintic polynomial trajectory.
The robot is able to pick up a cup from a table
and place it on another table.

The computer vision part of the project is implemented via a CNN. The CNN is trained
to detect the cup position on the image. The CNN is implemented in Tensorflow and Keras, 
and it uses a VGG16 pre-trained model as a base. The CNN is modified to predict the cup
relative to the image and convert it to the real XYZ coordinates, and it is trained 
with a dataset of 5000 images.

Resulting metrics of the CNN:

| Accuracy | R2 Score | MAE | MAE x | MAE y | Max Error x | Max Error y |
|----------|----------|-----|-------|-------|-------------|-------------|
| 0.9912   | 0.996    | 1.694 px | 2.054 px | 1.335 px | 10.1 px  | 8.8 px   |


## Dependencies

- CoppeliaSim (V-REP)
- Python >= 3.6
- Jupyter Notebook (Anaconda or pip)

## Usage

1. Open CoppeliaSim and load the scene `UR5.ttt`
2. Create a virtual environment and install the dependencies from `requirements.txt`
    ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3. Open the Jupyter Notebook `Trabalho.ipynb` and run the first cell to import the dependencies and start the simulation
4. Run the other cells, one by one, to see the robot in action
5. You can use the functions from `ur5.py` to control the robot and perform other tasks you want

Have fun! :sparkles: :robot: