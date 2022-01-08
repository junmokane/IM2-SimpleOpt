import GPy

import time
import os
from util import scale_from_unit_square_to_domain, scale_from_domain_to_unit_square
from objectives import bra_var, bra_max_min_var, gprice_var, gprice_max_min_var, hm3_var, hm3_max_min_var
from acq_fn import expected_improvement, probability_improvement, ucb
import matplotlib.pyplot as plt
from scipy import optimize
import sobol_seq
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--f_type', default='BRA-var', help='f type: BRA-var, GPRICE-var, HM3-var')
parser.add_argument('--acq_type', default='EI', help='acq type: PI, EI, UCB')
parser.add_argument('--bound_translation', type=float, default=0.1, help='translation bound')
parser.add_argument('--bound_scaling', type=float, default=0.1, help='scaling bound')
parser.add_argument('--M', type=int, default=50, help='# of fixed source datasets')


args = parser.parse_args()

rng = np.random.RandomState()
rng.seed(None)
exp = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]_exp_f_{args.f_type}_acq_{args.acq_type}'
os.makedirs(f"./results/{exp}", exist_ok=True)

if args.acq_type == 'PI':
    acq = probability_improvement
elif args.acq_type == 'EI':
    acq = expected_improvement
elif args.acq_type == 'UCB':
    acq = ucb
else:
    exit()

if args.f_type == 'BRA-var':
    M = args.M
    D = 2
    bound_scaling = args.bound_scaling
    bound_translation = args.bound_translation
    fct_params_domain = np.array([[-bound_translation, bound_translation],
                                  [-bound_translation, bound_translation],
                                  [1 - bound_scaling, 1 + bound_scaling]])
    fct_params_grid = sobol_seq.i4_sobol_generate(dim_num=3, n=M)  # 2 translations, 1 scaling
    fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

    param_idx = rng.choice(M)
    t = fct_params_grid[param_idx, 0:2]
    s = fct_params_grid[param_idx, 2]

    f = lambda x: bra_var(x, t=t, s=s)

    max_pos, max, _, min = bra_max_min_var(t=t, s=s)
    x_max = max_pos
    y_max = max
    y_min = min

elif args.f_type == 'GPRICE-var':
    M = args.M
    D = 2
    bound_scaling = args.bound_scaling
    bound_translation = args.bound_translation
    fct_params_domain = np.array([[-bound_translation, bound_translation],
                                  [-bound_translation, bound_translation],
                                  [1 - bound_scaling, 1 + bound_scaling]])
    fct_params_grid = sobol_seq.i4_sobol_generate(dim_num=3, n=M)  # 2 translations, 1 scaling
    fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

    param_idx = rng.choice(M)
    t = fct_params_grid[param_idx, 0:2]
    s = fct_params_grid[param_idx, 2]

    f = lambda x: gprice_var(x, t=t, s=s)

    max_pos, max, _, min = gprice_max_min_var(t=t, s=s)
    x_max = max_pos
    y_max = max
    y_min = min

elif args.f_type == 'HM3-var':
    # use M fixed source datasets evenly spread over the training set
    M = args.M
    D = 3
    bound_scaling = args.bound_scaling
    bound_translation = args.bound_translation
    fct_params_domain = np.array([[-bound_translation, bound_translation],
                                  [-bound_translation, bound_translation],
                                  [-bound_translation, bound_translation],
                                  [1 - bound_scaling, 1 + bound_scaling]])
    fct_params_grid = sobol_seq.i4_sobol_generate(dim_num=4, n=M)  # 3 translations, 1 scaling
    fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

    param_idx = rng.choice(M)
    t = fct_params_grid[param_idx, 0:3]
    s = fct_params_grid[param_idx, 3]

    f = lambda x: hm3_var(x, t=t, s=s)

    max_pos, max, _, min = hm3_max_min_var(t=t, s=s)
    x_max = max_pos
    y_max = max
    y_min = min

else:
    exit()


'''
GP regression
'''
# initialization
n_eval = 30
traj = {'x': [], 'y': [], 'regret': [], 'perf': []}

# random init
for i in range(2):
    x_init = rng.uniform(size=[1, D])
    y_init = f(x_init)
    traj['x'].append(x_init)
    traj['y'].append(y_init)
    traj['regret'].append(y_max - y_init)
    traj['perf'].append(np.concatenate(traj['regret']).squeeze().min())
    print(f'random starting point: [x: {x_init}, y: {f(x_init)}]')


for i in range(1, n_eval):
    # GP regression
    kernel = GPy.kern.Matern52(input_dim=D, variance=1., lengthscale=1.)
    X = np.concatenate(traj['x'], axis=0)  # (i,D)
    Y = f(X)
    gp_model = GPy.models.GPRegression(X, Y, kernel)
    gp_model.optimize(messages=False)
    gp_model.optimize_restarts(num_restarts=5)

    # print(traj)
    # print(np.concatenate(traj['y']).max())
    # print(traj['x'][0])
    # print(expected_improvement(f=gp_model.predict,
    #                            y_current=np.concatenate(traj['y']).max(),
    #                            x_proposed=traj['x'][0]))
    def f_(x):
        return -acq(f=gp_model.predict,
                    y_current=np.concatenate(traj['y']).max(),
                    x_proposed=x[None, :])

    # this searching part can be modified (different from paper)
    x0 = np.random.uniform(low=0., high=1., size=(25, D))
    proposal = None
    best_ei = np.inf
    for x0_ in x0:
        res = optimize.minimize(f_, x0_, bounds=np.array([[0.0, 1.0] for i in range(D)]))
        if res.success and res.fun < best_ei:
            best_ei = res.fun
            proposal = res.x
        if np.isnan(res.fun):
            raise ValueError("NaN within bounds")
    x_new = proposal[None, :]
    y_new = f(x_new)
    traj['x'].append(x_new)
    traj['y'].append(y_new)
    traj['regret'].append(y_max - y_new)
    traj['perf'].append(np.concatenate(traj['regret']).squeeze().min())
    print(f'proposed point: {proposal}, regret: {y_max - y_new}')

    # evaluation

    if not args.f_type == 'HM3-var':
        # plot objective
        a = b = np.arange(0, 1, 0.01)
        A, B = np.meshgrid(a, b)  # (100,100), (100,100)
        cs = f(np.stack([np.ravel(A), np.ravel(B)], axis=1))
        C = cs.squeeze().reshape(A.shape)
        fig = plt.figure()
        cp = plt.contourf(A, B, C, levels=np.linspace(C.reshape(-1, 1).min(), C.reshape(-1, 1).max(), 30))
        plt.colorbar(cp)
        # plot optimum
        plt.scatter(max_pos[:, 0], max_pos[:, 1], marker='X', color='white')
        # plot optimization trajectory
        for x in traj['x'][:-1]:
            plt.scatter(x[:, 0], x[:, 1], marker='o', color='blue')
        x_last = traj['x'][-1]
        plt.scatter(x_last[:, 0], x_last[:, 1], marker='o', color='red')
        x_best = traj['x'][np.concatenate(traj['y']).argmax()]
        plt.scatter(x_best[:, 0], x_best[:, 1], marker='x', color='red')
        plt.xlabel('red circle: current, red x: best, white x: max, blue circle: previous')
        plt.savefig(f'./results/{exp}/objective_plot_{i}.png')
        plt.close()

        # plot acquisition function
        a = b = np.arange(0, 1, 0.01)
        A, B = np.meshgrid(a, b)  # (100,100), (100,100)
        cs = expected_improvement(f=gp_model.predict,
                                  y_current=np.concatenate(traj['y']).max(),
                                  x_proposed=np.stack([np.ravel(A), np.ravel(B)], axis=1))
        C = cs.squeeze().reshape(A.shape)
        fig = plt.figure()
        cp = plt.contourf(A, B, C, levels=np.linspace(C.reshape(-1, 1).min(), C.reshape(-1, 1).max(), 30))
        plt.colorbar(cp)
        # plot optimum
        plt.scatter(max_pos[:, 0], max_pos[:, 1], marker='X', color='white')
        # plot optimization trajectory
        for x in traj['x'][:-1]:
            plt.scatter(x[:, 0], x[:, 1], marker='o', color='blue')
        x_last = traj['x'][-1]
        plt.scatter(x_last[:, 0], x_last[:, 1], marker='o', color='red')
        x_best = traj['x'][np.concatenate(traj['y']).argmax()]
        plt.scatter(x_best[:, 0], x_best[:, 1], marker='x', color='red')
        plt.xlabel('red circle: current, red x: best, white x: max, blue circle: previous')
        plt.savefig(f'./results/{exp}/acquisition_plot_{i}.png')
        plt.close()


# plot regret function
fig = plt.figure()
regret_vals = np.concatenate(traj['regret']).squeeze()
plt.plot(np.arange(len(regret_vals)), regret_vals)
plt.yscale('log')
plt.savefig(f'./results/{exp}/regret_curve.png')
plt.close()

# plot performance curve
fig = plt.figure()
regret_vals = traj['perf']
plt.plot(np.arange(len(regret_vals)), regret_vals)
plt.yscale('log')
plt.savefig(f'./results/{exp}/performance_curve.png')
plt.close()