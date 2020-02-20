import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

def regist_colormap():
    startcolor = '#D3D3D3'
    endcolor = '#555555'
    cmap2 = col.LinearSegmentedColormap.from_list('own_cm',[startcolor,endcolor])
    cm.register_cmap(cmap=cmap2)


def show_traj_image(tra_1, tra_2, tra_1_, tra_2_, normal=True):
    plt.figure(figsize=(25, 6))
    column = 2
    color_dict = ['#d33e4c', '#007672']
    s = 0.2*np.ones((50, 1))

    for i_seq in range(10):
        # before showing trajectory on road, normalize it
        xy_1 = tra_1[i_seq]
        xy_2 = tra_2[i_seq]
        x_1 = xy_1.T[0]
        y_1 = xy_1.T[1]
        x_2 = xy_2.T[0]
        y_2 = xy_2.T[1]

        plt.subplot(column, 10, i_seq+1)
        plt.scatter(x_1, y_1, s=s, color=color_dict[0])
        plt.scatter(x_2, y_2, s=s, color=color_dict[1])
        # draw the end point
        plt.plot(x_1[-1], y_1[-1], color=color_dict[0], marker='o', markersize=4)
        plt.plot(x_2[-1], y_2[-1], color=color_dict[1], marker='o', markersize=4)

        #plt.axis('off')
        if i_seq == 0:
            plt.ylabel('Original trajectory')
            if normal:
                plt.title('Normal dataset')
            else:
                plt.title('Collision dataset')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

    for i_seq in range(10):
        # before showing trajectory on road, normalize it
        xy_1 = tra_1_[i_seq]
        xy_2 = tra_2_[i_seq]
        x_1 = xy_1.T[0]
        y_1 = xy_1.T[1]
        x_2 = xy_2.T[0]
        y_2 = xy_2.T[1]

        plt.subplot(column, 10, i_seq+11)
        plt.scatter(x_1, y_1, s=s, color=color_dict[0])
        plt.scatter(x_2, y_2, s=s, color=color_dict[1])
        # draw the end point
        plt.plot(x_1[-1], y_1[-1], color=color_dict[0], marker='o', markersize=4)
        plt.plot(x_2[-1], y_2[-1], color=color_dict[1], marker='o', markersize=4)

        #plt.axis('off')
        if i_seq == 0:
            plt.ylabel('Recondtruction trajectory')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.97, top=0.95, hspace=0.2, wspace=0.2)
    plt.show()



regist_colormap()
sample_path = '../samples'

for i in range(10):
    x_s_file = os.path.join(sample_path, 'x_s.'+str(i)+'.npy')
    x_t_file = os.path.join(sample_path, 'x_t.'+str(i)+'.npy')
    x_s__file = os.path.join(sample_path, 'x_s_.'+str(i)+'.npy')
    x_t__file = os.path.join(sample_path, 'x_t_.'+str(i)+'.npy')

    with open(x_s_file, 'rb') as file_op:
        x_s = np.load(file_op)
    with open(x_t_file, 'rb') as file_op:
        x_t = np.load(file_op)
    with open(x_s__file, 'rb') as file_op:
        x_s_ = np.load(file_op)
    with open(x_t__file, 'rb') as file_op:
        x_t_ = np.load(file_op)

    traj_1_s = x_s[:,:,0:2]
    traj_2_s = x_s[:,:,2:4]
    traj_1_s_ = x_s_[:,:,0:2]
    traj_2_s_ = x_s_[:,:,2:4]

    show_traj_image(traj_1_s, traj_2_s, traj_1_s_, traj_2_s_, True)
    
    traj_1_s = x_t[:,:,0:2]
    traj_2_s = x_t[:,:,2:4]
    traj_1_s_ = x_t_[:,:,0:2]
    traj_2_s_ = x_t_[:,:,2:4]

    show_traj_image(traj_1_s, traj_2_s, traj_1_s_, traj_2_s_, False)
