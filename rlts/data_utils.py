import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def to_traj(file):
    traj = []
    f = open(file)
    for line in f:
        temp = line.strip().split(' ')
        if len(temp) < 2:
            continue
        traj.append([float(temp[0]),  int(float(temp[1]))])
    f.close()
    return traj

def get_point(ps, pe, segment, index):
    syn_time = segment[index][1]  # Now index 1 is for time
    time_ratio = 1 if (pe[1]- ps[1]) == 0 else (syn_time-ps[1]) / (pe[1]-ps[1])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    return [syn_x], syn_time

def sed_op(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        #print('segment', segment)
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            syn_time = segment[i][1]
            time_ratio = 1 if (pe[1]- ps[1]) == 0  else (syn_time-ps[1]) / (pe[1]-ps[1])
            syn_value = ps[0] + (pe[0] - ps[0]) * time_ratio
            e = max(e, abs(segment[i][0] - syn_value))
        #print('segment error', e)
        return e

def sed_error(ori_traj, sim_traj):
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    # 计算最大绝对误差
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, sed_op(ori_traj[start: c+1]))
            start = c
    # 计算原始轨迹第一维的值域（max–min）
    values = [pt[0] for pt in ori_traj]
    data_range = max(values) - min(values)
    # 防止除以 0
    pct_error = (error / data_range * 100.0) if data_range != 0 else 0.0
    return t_map, error, pct_error

def speed_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(0, len(segment)-1):
            p_1, t_1 = get_point(ps, pe, segment, i)
            p_2, t_2 = get_point(ps, pe, segment, i+1)
            time = 1 if t_2 - t_1 == 0 else abs(t_2-t_1)
            est_speed = abs(p_1[0] - p_2[0]) / time  # p_1 and p_2 are now one-dimensional
            rea_speed = abs(segment[i][0] - segment[i+1][0]) / time
            e = max(e, abs(est_speed - rea_speed))
        return e

def speed_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, speed_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def draw_sed_op(segment):
    """
    计算同步欧氏距离误差（一维）。
    segment: 轨迹段 [[value, t], ...]
    """
    if len(segment) <= 2:
        return 0.0, segment[0], segment[0], segment[0], segment[0]
    ps = segment[0]
    pe = segment[-1]
    e = 0.0
    e_points = segment[0]
    syn = ps[0]
    for i in range(1, len(segment) - 1):
        syn_time = segment[i][1]
        time_ratio = 1 if (pe[1] - ps[1]) == 0 else (syn_time - ps[1]) / (pe[1] - ps[1])
        syn_value = ps[0] + (pe[0] - ps[0]) * time_ratio
        t = abs(segment[i][0] - syn_value)  # 一维数据的差值
        if t >= e:
            e = t
            e_points = segment[i]
            syn = syn_value
    return e, e_points, ps, pe, syn

def draw_error(ori_traj, sim_traj, label):
    """
    计算简化轨迹的误差（一维）。
    ori_traj, sim_traj: [[value, t], ...]
    label: 误差类型（例如 'sed'）
    """
    dict_traj = {}
    t_map = [0 for _ in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    error_points = ori_traj[0]
    error_left   = ori_traj[0]
    error_right  = ori_traj[-1]
    error_syn    = ori_traj[0][0]
    for c, value in enumerate(t_map):
        if value == 1:
            if label == 'sed':
                e, e_points, ps, pe, syn = draw_sed_op(ori_traj[start:c + 1])
                if e > error:
                    error = e
                    error_points = e_points
                    error_syn = syn
                    error_left = ps
                    error_right = pe
            start = c
    return error, error_points, error_left, error_right, error_syn

def draw(ori_traj, sim_traj, label='sed'):
    error, error_points, error_left, error_right, error_syn = draw_error(ori_traj, sim_traj, label)
    print(error_points, error_left, error_right, error_syn)
    # pdf = PdfPages('vis_rlts_online.pdf')
    plt.figure(figsize=(10.5/2,6.8/2))
    plt.plot(np.array(ori_traj)[:,1],np.array(ori_traj)[:,0],color="blue", linewidth=0.7, label='raw traj')
    plt.scatter(np.array(sim_traj)[:,1],np.array(sim_traj)[:,0],color="red", s=2)
    plt.plot(np.array(sim_traj)[:,1],np.array(sim_traj)[:,0], '--', color="red", linewidth=0.5, label='simplified traj')
    plt.plot([error_points[1],error_points[1]],[error_points[0],error_syn], '--', color="black", label='SED')
    plt.plot([error_left[1],error_right[1]],[error_left[0],error_right[0]], color="green", linewidth=2, label='anchor seg')
    plt.title('simplified traj length: '+str(len(sim_traj)))
    plt.legend(loc='best', prop = {'size': 9})
    #plt.show()
    plt.savefig(f"vis_rlts_{error:.6f}_{len(sim_traj)}.png", dpi=300)
    return error

if __name__ == '__main__':
    print('The required tools are implemented here!')