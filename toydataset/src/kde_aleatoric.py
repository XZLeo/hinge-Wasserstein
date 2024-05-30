'''
plot KDE for comparing in-distribution test set and OOD test set, or aleatoric uncertainty
'''
import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.utils import make_directory

test1_metric_paths = ['/home/zilxi06/debug/data/toy_aleatoric_plainW/resnet_entropy_index_index_test1.pickle',
                '/home/zilxi06/debug/data/toy_aleatoric_hingeW/resnet_entropy_index_index_test1.pickle',
                '/home/zilxi06/debug/data/toy_aleatoric_NLL/resnet_entropy_index_index_test1.pickle']

test2_metric_paths = ['/home/zilxi06/debug/data/toy_aleatoric_plainW/resnet_entropy_index_index_test2.pickle',
                    '/home/zilxi06/debug/data/toy_aleatoric_hingeW/resnet_entropy_index_index_test2.pickle',
                    '/home/zilxi06/debug/data/toy_aleatoric_NLL/resnet_entropy_index_index_test2.pickle']

bandwidth = 5e-3
scale = 5

label1 = 'oneline'
label2 = 'twoline'
img_path = f'toydataset/img/{label1}_{label2}/'
make_directory(img_path)

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

colors = ['tab:blue', 'tab:orange', 'tab:green']

for idx, test1_metric_path in enumerate(test1_metric_paths):
    test2_metric_path = test2_metric_paths[idx]
    with open(test1_metric_path, 'rb') as f:
        result_1 = pickle.load(f)
    with open(test2_metric_path, 'rb') as f:
        result_2 = pickle.load(f)    

    loss_name = test1_metric_path.split('/')[5].split('_')[2]
    # test set 1
    theta_entropy_1 = np.array(result_1['theta metrics'])[:, np.newaxis]
    rho_entropy_1 = np.array(result_1['rho metrics'])[:, np.newaxis]
    joint_entropy_1 = np.array(result_1['rho metrics']) + np.array(result_1['theta metrics'])
    joint_entropy_1 = joint_entropy_1[:, np.newaxis]
    # test set 2
    theta_entropy_2 = np.array(result_2['theta metrics'])[:, np.newaxis]
    rho_entropy_2 = np.array(result_2['rho metrics'])[:, np.newaxis]
    joint_entropy_2 = np.array(result_2['rho metrics']) + np.array(result_2['theta metrics'])
    joint_entropy_2 = joint_entropy_2[:, np.newaxis]

    max_theta_entropy = np.maximum(np.max(result_1['theta metrics']), np.max(result_2['theta metrics']))
    max_rho_entropy = np.maximum(np.max(result_1['rho metrics']), np.max(result_2['rho metrics']))
    max_joint_entropy = np.maximum(np.max(joint_entropy_1), np.max(joint_entropy_2))
    min_theta_entropy = np.minimum(np.min(result_1['theta metrics']), np.min(result_2['theta metrics']))
    min_rho_entropy = np.minimum(np.min(result_1['rho metrics']), np.min(result_2['rho metrics']))
    min_joint_entropy = np.minimum(np.min(joint_entropy_1), np.min(joint_entropy_2))
    theta_range = np.abs(max_theta_entropy-min_theta_entropy)
    rho_range = np.abs(max_rho_entropy-min_rho_entropy)
    joint_range = np.abs(max_joint_entropy-min_joint_entropy)

    
    X_plot = np.linspace(min_theta_entropy-theta_range/scale, max_theta_entropy+theta_range/scale, 100)[:, np.newaxis] # use the range above 
    kde = KernelDensity(bandwidth=bandwidth).fit(theta_entropy_1)
    log_dens = kde.score_samples(X_plot)
    ax1.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        linestyle="-",
        lw=1,
        label=f'1 line {loss_name}',
        color=mcolors.TABLEAU_COLORS[colors[idx]]
    )
    kde = KernelDensity(bandwidth=bandwidth).fit(theta_entropy_2)
    log_dens = kde.score_samples(X_plot)
    ax1.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        linestyle="-.",
        lw=0.5,
        label=f'2 line {loss_name}',
        color=mcolors.TABLEAU_COLORS[colors[idx]]
    )
    
    X_plot = np.linspace(min_rho_entropy-rho_range/scale, max_rho_entropy+rho_range/scale, 100)[:, np.newaxis]
    kde = KernelDensity(bandwidth=bandwidth).fit(rho_entropy_1)
    log_dens = kde.score_samples(X_plot)
    ax2.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        linestyle="-",
        lw=1,
        label=f'1 line {loss_name}',
        color=mcolors.TABLEAU_COLORS[colors[idx]]
    )
    kde = KernelDensity(bandwidth=bandwidth).fit(rho_entropy_2)
    log_dens = kde.score_samples(X_plot)
    ax2.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        linestyle="-.",
        lw=0.5,
        label=f'2 line {loss_name}',
        color=mcolors.TABLEAU_COLORS[colors[idx]]
    )

    X_plot = np.linspace(min_joint_entropy-joint_range/scale, max_joint_entropy+joint_range/scale, 100)[:, np.newaxis]
    kde = KernelDensity(bandwidth=bandwidth).fit(joint_entropy_1)
    log_dens = kde.score_samples(X_plot)
    ax3.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        linestyle="-",
        lw=1,
        label=f'1 line {loss_name}',
        color=mcolors.TABLEAU_COLORS[colors[idx]]
    )
    kde = KernelDensity(bandwidth=bandwidth).fit(joint_entropy_2)
    log_dens = kde.score_samples(X_plot)
    ax3.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        linestyle="-.",
        lw=0.5,
        label=f'2 line {loss_name}',
        color=mcolors.TABLEAU_COLORS[colors[idx]]
    )
    
ax1.legend(loc="upper right", fontsize="5")
ax1.set_ylabel('Density', fontsize="8")
ax1.set_xlim(0, 5.7)
ax1.set_ylim(0, 5.0)
ax1.set_title(r'$\alpha$ entropy', fontsize='10')
ax1.tick_params(axis='both', which='major', labelsize=5)

ax2.legend(loc="upper right", fontsize="5")
ax2.set_ylabel('Density', fontsize="8")
ax2.set_xlim(0, 5.7)
ax2.set_ylim(0, 4.5)
ax2.set_title(r'$\rho$ entropy', fontsize='10')
ax2.tick_params(axis='both', which='major', labelsize=5)

ax3.legend(loc="upper right", fontsize="5")
ax3.set_xlabel('Entropy', fontsize="8")
ax3.set_ylabel('Density', fontsize="8")
ax3.set_xlim(0, 11.3)
ax3.set_ylim(0, 4.5)
ax3.set_title('Joint entropy', fontsize='10')
ax3.tick_params(axis='both', which='major', labelsize=5)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)


plt.savefig(f'{img_path}/aleatoric_kde.pdf')

plt.close()
