# PMLDL project - [repo](github.com/lvjonok/f23-pmldl-project)

## Team:

Lev Kozlov - l.kozlov@innopolis.university
Anton Kirilin - a.kirilin@innopolis.university
Ilia Milyoshin - i.mileshin@innopolis.university

## Description

Implementation of service for system identification problem using deep learning. The top view on the problem is to infer dynamic parameters of dynamic model given data with trajectories. Although in control engineering this problem is mostly tackled with the use of physically-inspired regressors which might limit complexity of the model, we will try to utilize the use of deep learning and provide comparison between results.

Service stages:
1. Request structural description of dynamic model (URDF)
2. Request trajectories from experiments
3. Train model and evaluate
4. Output description of dynamic model with inferred parameters (URDF)

## What have been done so far?

- reviewed the [paper](https://www.sciencedirect.com/science/article/pii/S2405896320317353) we were inspired by
- discussed what we would like to see at the end of project
- agreed on the list of dynamic models we would like to experiment with
