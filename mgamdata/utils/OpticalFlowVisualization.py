import os
import numpy as np
from matplotlib import pyplot as plt




def quiver_flow(flow, save_folder:str=''):
    """
    flow : np.float32, [h, w, 2], (x, y) format
    """
    h, w = flow.shape[:2]
    xx, yy = np.meshgrid(range(w), range(h))

    s = 16
    plt.figure()
    plt.quiver(xx[::s, ::s], yy[::s, ::s], flow[..., 0][::s, ::s], flow[..., 1][::s, ::s], color='green')
    if save_folder: 
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, 'flow.png'))
    plt.show()




