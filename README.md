# drlnd-continuous_control
My implementation of the Continuos Control Project in the scope of Udacity's Deep Reinforcement Learning Nanodegree

The task was to train a robotic arm to move into a target area using the PyTorch package (https://pytorch.org/).
A description of the results can be found in the Report.ipynb jupyter notebook.

# Installation
In order to run the training script you'll need a a working Python environment with a couple of different non-standard libraries.
One simple way to get everything set-up correctly is:
- Install Python, for example using Anaconda from https://www.anaconda.com/
- Follow the instructions from the Deep Reinforcement Learning Repository: https://github.com/udacity/deep-reinforcement-learning
to install the base packages needed.
- You'll also need to download a pre-compiled Unity Environment, links to versions for different operating systems can be found under "Getting started" at
https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control
While I only tested this approach on Ubuntu Linux, it should work on Windows or Mac OS in the same way.

# Description of the Reacher Environment
The goal is to train a robotic arm with two joints to move into a dyncamically moving target zone.
A video of the environment can be found at https://www.youtube.com/watch?v=2N9EoF6pQyE&feature=youtu.be.
A detailed description of the environment can be found at https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control.  
According to the link above, the agent has an observation space with 33 continuos dimensions. The observations include, for example, position, rotation and velocity of
the moving arm. Even though it is not mentioned in the source above, the observations also include the location of the target zone according to the youtube video.
The action space consists of four different continuos actions that correspond to torque that can be applied to the two joints.

# Running the Agent
To run the agent you just need to run the "main.py" file. You might need to update the path of the Unity environment in line 11.
The settings are given starting from line 18, the comments behind each line give a brief description of the parameters.

