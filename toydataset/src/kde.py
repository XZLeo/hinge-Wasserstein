'''
plot KDE for comparing in-distribution test set and OOD test set, or aleatoric uncertainty
'''
import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import matplotlib.pyplot as plt

from src.utils import make_directory

metric_path_1 = '/home/zilxi06/debug/data/toy_aleatoric_HingeEMD.0/resnet_entropy_index_index_test1.pickle'
metric_path_2 = '/home/zilxi06/debug/data/toy_aleatoric_HingeEMD.0/resnet_entropy_index_index_test2.pickle'
bandwidth = 5e-3
scale = 5

# label1 = metric_path_1.split('/')[5]
# label2 = metric_path_2.split('/')[5]
label1 = 'oneline'
label2 = 'twoline'
loss_name = metric_path_1.split('/')[6]

img_path = f'toydataset/img/{label1}_{label2}/{loss_name}'
print(img_path)
make_directory(img_path)

with open(metric_path_1, 'rb') as f:
    result_1 = pickle.load(f)
print('***Out of Distribution***')
print('theta entropy range: [{}, {}]'.format(np.min(result_1['theta metrics']), np.max(result_1['theta metrics'])))
print('rho entropy range: [{}, {}]'.format(np.min(result_1['rho metrics']), np.max(result_1['rho metrics'])))

theta_entropy_1 = np.array(result_1['theta metrics'])[:, np.newaxis]
rho_entropy_1 = np.array(result_1['rho metrics'])[:, np.newaxis]
joint_entropy_1 = np.array(result_1['rho metrics']) + np.array(result_1['theta metrics'])
joint_entropy_1 = joint_entropy_1[:, np.newaxis]
print('joint entropy range: [{}, {}]'.format(np.min(joint_entropy_1), np.max(joint_entropy_1)))

with open(metric_path_2, 'rb') as f:
    result_2 = pickle.load(f)
print('***In Distribution***')
print('theta entropy range: [{}, {}]'.format(np.min(result_2['theta metrics']), np.max(result_2['theta metrics'])))
print('rho entropy range: [{}, {}]'.format(np.min(result_2['rho metrics']), np.max(result_2['rho metrics'])))

theta_entropy_2 = np.array(result_2['theta metrics'])[:, np.newaxis]
rho_entropy_2 = np.array(result_2['rho metrics'])[:, np.newaxis]
joint_entropy_2 = np.array(result_2['rho metrics']) + np.array(result_2['theta metrics'])
joint_entropy_2 = joint_entropy_2[:, np.newaxis]
print('joint entropy range: [{}, {}]'.format(np.min(joint_entropy_2), np.max(joint_entropy_2)))

max_theta_entropy = np.maximum(np.max(result_1['theta metrics']), np.max(result_2['theta metrics']))
max_rho_entropy = np.maximum(np.max(result_1['rho metrics']), np.max(result_2['rho metrics']))
max_joint_entropy = np.maximum(np.max(joint_entropy_1), np.max(joint_entropy_2))
min_theta_entropy = np.minimum(np.min(result_1['theta metrics']), np.min(result_2['theta metrics']))
min_rho_entropy = np.minimum(np.min(result_1['rho metrics']), np.min(result_2['rho metrics']))
min_joint_entropy = np.minimum(np.min(joint_entropy_1), np.min(joint_entropy_2))
theta_range = np.abs(max_theta_entropy-min_theta_entropy)
rho_range = np.abs(max_rho_entropy-min_rho_entropy)
joint_range = np.abs(max_joint_entropy-min_joint_entropy)

fig, ax = plt.subplots()
X_plot = np.linspace(min_theta_entropy-theta_range/scale, max_theta_entropy+theta_range/scale, 100)[:, np.newaxis] # use the range above 
kde = KernelDensity(bandwidth=bandwidth).fit(theta_entropy_1)
log_dens = kde.score_samples(X_plot)
ax.plot(
    X_plot[:, 0],
    np.exp(log_dens),
    linestyle="-",
    lw=1,
    label=label1
)
kde = KernelDensity(bandwidth=bandwidth).fit(theta_entropy_2)
log_dens = kde.score_samples(X_plot)
ax.plot(
    X_plot[:, 0],
    np.exp(log_dens),
    linestyle="-",
    lw=1,
    label=label2
)
ax.legend(loc="upper right")
ax.set_xlabel('Entropy')
ax.set_ylabel('Density')
# plt.title('Plain Wasserstein')
plt.xlim(0, 4.7)
plt.ylim(0, 4.0)
plt.savefig(f'{img_path}/theta_entropy.pdf')
plt.close()

X_plot = np.linspace(min_rho_entropy-rho_range/scale, max_rho_entropy+rho_range/scale, 100)[:, np.newaxis]
fig, ax = plt.subplots()
kde = KernelDensity(bandwidth=bandwidth).fit(rho_entropy_1)
log_dens = kde.score_samples(X_plot)
ax.plot(
    X_plot[:, 0],
    np.exp(log_dens),
    linestyle="-",
    lw=1,
    label=label1
)
kde = KernelDensity(bandwidth=bandwidth).fit(rho_entropy_2)
log_dens = kde.score_samples(X_plot)
ax.plot(
    X_plot[:, 0],
    np.exp(log_dens),
    linestyle="-",
    lw=1,
    label=label2
)
ax.legend(loc="upper right")
ax.set_xlabel('entropy')
plt.xlim(0, 4.7)
plt.ylim(0, 4.0)
# plt.title('rho entropy')
plt.savefig(f'{img_path}/rho_entropy.pdf')

X_plot = np.linspace(min_joint_entropy-joint_range/scale, max_joint_entropy+joint_range/scale, 100)[:, np.newaxis]
fig, ax = plt.subplots()
kde = KernelDensity(bandwidth=bandwidth).fit(joint_entropy_1)
log_dens = kde.score_samples(X_plot)
ax.plot(
    X_plot[:, 0],
    np.exp(log_dens),
    linestyle="-",
    lw=1,
    label=label1
)
kde = KernelDensity(bandwidth=bandwidth).fit(joint_entropy_2)
log_dens = kde.score_samples(X_plot)
ax.plot(
    X_plot[:, 0],
    np.exp(log_dens),
    linestyle="-",
    lw=1,
    label=label2
)
ax.legend(loc="upper right")
ax.set_xlabel('entropy')
plt.xlim(0, 9.2)
plt.ylim(0, 4.0)
# plt.title('Joint entropy')
plt.savefig(f'{img_path}/joint_entropy.pdf')