import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

def regist_colormap():
    startcolor = '#BEBEBE'
    endcolor = '#555555'
    cmap2 = col.LinearSegmentedColormap.from_list('own_cm',[startcolor,endcolor])
    cm.register_cmap(cmap=cmap2)


def show_traj_image(tra_1, tra_2, condition):
    plt.figure(figsize=(11, 6))
    column = 4
    color_dict = ['#d33e4c', '#007672']
    s = 0.2*np.ones((50, 1))

    for i_col in range(column):
        for i_seq in range(10):
            # before showing trajectory on road, normalize it
            xy_1 = tra_1[i_seq*10+i_col]
            x_1 = (xy_1.T[0]+1)*(condition.shape[-1]/2)-1
            y_1 = (xy_1.T[1]+1)*(condition.shape[-1]/2)-1
            xy_2 = tra_2[i_seq*10+i_col]
            x_2 = (xy_2.T[0]+1)*(condition.shape[-1]/2)-1
            y_2 = (xy_2.T[1]+1)*(condition.shape[-1]/2)-1

            plt.subplot(column, 10, i_seq+1+i_col*10)
            plt.imshow(1-condition[i_col][0], cmap='own_cm')
            plt.scatter(x_1, y_1, s=s, color=color_dict[0])
            plt.scatter(x_2, y_2, s=s, color=color_dict[1])

            # draw the start point
            #plt.plot(x_1[0], y_1[0], color='red', marker='o')
            #plt.plot(x_2[0], y_2[0], color='red', marker='o')
            # draw the end point
            plt.plot(x_1[-1], y_1[-1], color=color_dict[0], marker='o', markersize=4)
            plt.plot(x_2[-1], y_2[-1], color=color_dict[1], marker='o', markersize=4)

            '''
            if i_seq == 0:
                plt.title('Collision')
            elif i_seq == 9:
                plt.title('Normal')
            else:
                plt.title('--------------->')
            '''

            plt.axis('off')
            plt.xlim([0, 128])
            plt.ylim([0, 128])

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0.1)
    plt.show()



regist_colormap()
sample_path = './samples'

for i in range(2):
    c_s_file = os.path.join(sample_path, 'c_s.'+str(i)+'.npy')
    c_t_file = os.path.join(sample_path, 'c_t.'+str(i)+'.npy')
    f_s_file = os.path.join(sample_path, 'x_f_s.'+str(i)+'.npy')
    f_t_file = os.path.join(sample_path, 'x_f_t.'+str(i)+'.npy')

    with open(c_s_file, 'rb') as file_op:
        c_s = np.load(file_op)
    with open(c_t_file, 'rb') as file_op:
        c_t = np.load(file_op)
    with open(f_s_file, 'rb') as file_op:
        f_s = np.load(file_op)
    with open(f_t_file, 'rb') as file_op:
        f_t = np.load(file_op)

    traj_1_s = f_s[:,:,0:2]
    traj_2_s = f_s[:,:,2:4]
    traj_1_t = f_t[:,:,0:2]
    traj_2_t = f_t[:,:,2:4]

    # traj_1_s and traj_2_s are generated from c_s
    # traj_1_t and traj_2_t are generated from c_t
    # maps are only used for displaying
    show_traj_image(traj_1_s, traj_2_s, c_s)
    show_traj_image(traj_1_t, traj_2_t, c_t)
