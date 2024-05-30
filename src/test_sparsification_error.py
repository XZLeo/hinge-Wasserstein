from matplotlib.pyplot import plot, xlabel, ylabel, figure, title, savefig, legend, axis, xlim, ylim, gca
import numpy as np

from src.utils import make_directory

# read errors from sparsification_error.txt
with open('data/epestimic_sparsification.txt') as f:
        error_path_list = f.read().splitlines() 
        
        
figure()
fractions = np.arange(0, 1, 0.01)
legend_list = []
for error_path in error_path_list:
    name = error_path.split('/')[1].split('_')[1]+'_'+error_path.split('/')[1].split('_')[2]
    error = np.load(error_path)    
    plot(fractions, error)
    legend_list.append(name)
    
ylabel('Sparsification Error')
xlabel('Fraction of removed samples')
axes = gca()
axes.set_xlim([0, 1])
legend(legend_list, fontsize="7", loc ="upper left")
axis('equal')

make_directory(f'./img/sparsification_error/')
savefig(f'./img/theta_sparsification_error.pdf')
    