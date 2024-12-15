# PMLDL project - [repo](github.com/lvjonok/f23-pmldl-project)

## Initial team:

* Lev Kozlov - l.kozlov@innopolis.university
* Anton Kirilin - a.kirilin@innopolis.university
* Ilia Milyoshin - i.mileshin@innopolis.university

## Description

This is a repository with code for experiments on solving the trajectory optimization problem with using neural networks for calculating the inverse dynamics of a system. 
in the repo you can find:
- [`paper draft`](https://github.com/lvjonok/f23-pmldl-project/blob/ilia/ArticleDraft.pdf)
- [`Final solution demonstration`](https://github.com/lvjonok/f23-pmldl-project/blob/ilia/notebooks/final/cart_pole.ipynb)
- [`Demonstartion dataset geenration`](https://github.com/lvjonok/f23-pmldl-project/blob/ilia/simulation/cart_pole.ipynb)

You are free to explore other parts of the repo and ask question about it. Simply create an issue and we'll answer it later.



## What have been done so far?

- Did experiments with noiseless data
- tried out a gaussian noise models
- wrote a draft of the paper for submission
- reviewd a few articles for exploring the problem

## After 18.09.2023

- after the review we agreed that project idea needs reformulation and we dropped this idea

## New idea

- collect data from experiments, train the model to predict the next state of system given current state and applied control
- formulate trajectory optimization task where the dynamics of model is given by neural network prediction
- we aim to use [`CasADi`](https://web.casadi.org/) to create a nonlinear program from optimization task and [`l4casadi`](https://github.com/Tim-Salzmann/l4casadi) as framework to integrate `pyTorch` model

## After 11.11.2024

- Added noisy dataset creation
- Comparison of two models (trained on noisy data and on initial one)
