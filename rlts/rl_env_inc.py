import os
import numpy as np
import data_utils as F
import copy
#import heapq
#from heapq import heappush, heappop, _siftdown, _siftup
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
import math

class TrajComp():
    def __init__(self, path, amount, a_size, s_size):
        self.n_actions = a_size
        self.n_features = s_size
        self._load(path, amount)

    def _load(self, path, amount):
        self.ori_traj_set = []
        for num in range(amount):
            self.ori_traj_set.append(F.to_traj(path + str(num)))
            
    def read(self, p, episode):
        self.F_ward[self.link_tail] = [0.0, p]
        self.B_ward[p] = [0.0, self.link_tail]
        s = self.B_ward[self.link_tail][1]
        m = self.link_tail
        e = self.F_ward[self.link_tail][1]
        self.err_record[(s, e)] = F.sed_op(self.ori_traj_set[episode][s: e + 1])
        self.F_ward[m][0] = self.err_record[(s, e)]
        self.B_ward[m][0] = self.err_record[(s, e)]
        #heapq.heappush(self.heap, (self.F_ward[m][0], m))# save (state_value, point index of ori traj)
        self.sortedlist.add((self.F_ward[m][0], m))
        self.link_tail = p
    
    def reset(self, episode, buffer_size):
        #self.heap = []
        self.last_error = 0.0
        self.current = 0.0
        self.c_left = 0
        self.c_right = 0
        #self.copy_traj = copy.deepcopy(self.ori_traj_set[episode]) # for testing the correctness of inc rewards
        self.start = {}
        self.end = {}
        self.err_seg = {}
        self.err_record = {}
        steps = len(self.ori_traj_set[episode])
        self.F_ward = {0:[0.0,1]} # save (state_value, next_point)
        self.B_ward = {1:[0.0,0]} # save (state_value, last_point)
        # self.F_ward[0] = [0.0, 1]
        # self.B_ward[1] = [0.0, 0]
        self.link_head = 0
        self.link_tail = 1
        self.sortedlist = SortedList({})
        for i in range(2, buffer_size + 1):
            self.read(i, episode)
        #t = heapq.nsmallest(self.n_features, self.heap)
        t = self.sortedlist[:self.n_features]
        # print(len(t))
        # print(self.n_features)
        # print(t)
        if len(t) < self.n_features:
            self.check = [t[0][1],t[0][1],t[1][1]]
            self.state = [t[0][0],t[0][0],t[1][0]]
        else:
            self.check = [t[0][1], t[1][1],t[2][1]]
            self.state = [t[0][0], t[1][0],t[2][0]]
        
        return steps, np.array(self.state).reshape(1, -1)           
        
    def reward_update(self, episode, rem):
        if (rem not in self.start) and (rem not in self.end):
            #interval insert
            a = self.B_ward[rem][1]
            b = self.F_ward[rem][1]
            self.start[a] = b
            self.end[b] = a
            NOW = self.err_record[(a,b)]
            self.err_seg[(a,b)] = NOW
            if NOW >= self.last_error:
                self.current = NOW
                self.current_left, self.current_right = a, b
        
        elif (rem in self.start) and (rem not in self.end):
            #interval expand left
            a = self.B_ward[rem][1]
            b = rem
            c = self.start[rem]
            BEFORE = self.err_record[(b, c)]
            NOW = self.err_record[(a, c)]
            del self.err_seg[(b,c)]
            self.err_seg[(a,c)] = NOW
            
            if  math.isclose(self.last_error,BEFORE):
                if NOW >= BEFORE:
                    #interval expand left_case1
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #interval expand left_case2
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #interval expand left_case3
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
            self.end[c] = a
            self.start[a] = c
            del self.start[b]
            
        # interval expand right
        elif (rem not in self.start) and (rem in self.end):
            #interval expand right
            a = self.end[rem]
            b = rem
            c = self.F_ward[rem][1]
            BEFORE = self.err_record[(a, b)]
            NOW = self.err_record[(a, c)]
            del self.err_seg[(a,b)]
            self.err_seg[(a,c)] = NOW
            if math.isclose(self.last_error,BEFORE):
                if NOW >= BEFORE:
                    #interval expand right_case1
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #interval expand right_case2
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #interval expand right_case3
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
            self.start[a] = c
            self.end[c] = a
            del self.end[b]
        
        # interval merge
        elif (rem in self.start) and (rem in self.end):
            #interval merge
            b = rem
            a = self.end[b]
            c = self.start[b]
            # get values quickly
            BEFORE_1 = self.err_record[(a, b)]
            BEFORE_2 = self.err_record[(b, c)]
            NOW = self.err_record[(a, c)]
            del self.err_seg[(a,b)]
            del self.err_seg[(b,c)]
            self.err_seg[(a,c)] = NOW            
            if math.isclose(self.last_error,BEFORE_1):
                if NOW >= BEFORE_1:
                    #interval merge_case1
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #interval merge_case2
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
                    
            elif math.isclose(self.last_error,BEFORE_2):
                if NOW >= BEFORE_2:
                    #interval merge_case3
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #interval merge_case4
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #interval merge_case5
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                    
            self.start[a] = c
            self.end[c] = a
            del self.start[b]
            del self.end[b]
        else:
            print('Here is a bug!!!')
        
    def step(self, episode, action, index, done, label='T'):        
        # update state and compute reward

        rem = self.check[action] # point index in ori traj

        NEXT_P = self.F_ward[rem][1]
        NEXT_V = self.B_ward[NEXT_P][0]
        LAST_P = self.B_ward[rem][1]
        LAST_V = self.F_ward[LAST_P][0]

        if LAST_P > self.link_head:
            #self.delete_heap(self.heap, (LAST_V, LAST_P))
            self.sortedlist.remove((LAST_V, LAST_P))
            self.err_record[(self.B_ward[LAST_P][1], NEXT_P)] = F.sed_op(self.ori_traj_set[episode][self.B_ward[LAST_P][1]: NEXT_P + 1])
            self.F_ward[LAST_P][0] = self.err_record[(self.B_ward[LAST_P][1], NEXT_P)]
            self.B_ward[LAST_P][0] = self.err_record[(self.B_ward[LAST_P][1], NEXT_P)]
            #heapq.heappush(self.heap, (self.F_ward[LAST_P][0], LAST_P))
            self.sortedlist.add((self.F_ward[LAST_P][0], LAST_P))
        if NEXT_P < self.link_tail:
            #self.delete_heap(self.heap, (NEXT_V, NEXT_P))
            self.sortedlist.remove((NEXT_V, NEXT_P))
            self.err_record[(LAST_P, self.F_ward[NEXT_P][1])] = F.sed_op(self.ori_traj_set[episode][LAST_P: self.F_ward[NEXT_P][1] + 1])
            self.F_ward[NEXT_P][0] = self.err_record[(LAST_P, self.F_ward[NEXT_P][1])]
            self.B_ward[NEXT_P][0] = self.err_record[(LAST_P, self.F_ward[NEXT_P][1])]
            #heapq.heappush(self.heap, (self.F_ward[NEXT_P][0], NEXT_P))
            self.sortedlist.add((self.F_ward[NEXT_P][0], NEXT_P))
        
        #self.copy_traj.remove(self.ori_traj_set[episode][rem]) # for testing the correctness of inc rewards
        self.reward_update(episode, rem)
        
        self.F_ward[LAST_P][1] = NEXT_P
        self.B_ward[NEXT_P][1] = LAST_P
        #self.delete_heap(self.heap, (self.F_ward[rem][0], rem))
        self.sortedlist.remove((self.F_ward[rem][0], rem))
        del self.F_ward[rem]
        del self.B_ward[rem]     
        
        #_,  self.current = F.sed_error(self.ori_traj_set[episode], self.copy_traj) # for testing the correctness of inc rewards
        rw = self.last_error - self.current
        self.last_error = self.current
        #print('self.current',self.current)
        
        if not done:
            self.read(index + 1, episode)
            #t = heapq.nsmallest(self.n_features, self.heap)
            t = self.sortedlist[:self.n_features]
            if len(t) < self.n_features:
                self.check = [t[0][1],t[0][1],t[1][1]]
                self.state = [t[0][0],t[0][0],t[1][0]]
            else:
                self.check = [t[0][1], t[1][1],t[2][1]]
                self.state = [t[0][0], t[1][0],t[2][0]]

        return np.array(self.state).reshape(1, -1), rw
    
    output_dir_norm = "dataset/norm_dims/sim"
    orig_dir = "dataset/dims/"

    def compute_weights(self, sim_traj):
        values = np.array([pt[0] for pt in sim_traj], dtype=float)
        times  = np.array([pt[1] for pt in sim_traj], dtype=float)
        raw_w = []
        for i in range(len(values)):
            if i == 0:
                dv = abs(values[1] - values[0])
                dt = times[1] - times[0]
            else:
                dv = abs(values[i] - values[i-1])
                dt = times[i] - times[i-1]
            # 防止除以 0
            w = dv/(dt if dt>0 else 1e-8)
            raw_w.append(w)
        raw_w = np.array(raw_w, dtype=float)

        # z-score 标准化
        mu, sigma = raw_w.mean(), raw_w.std(ddof=0)
        if sigma > 0:
            z = (raw_w - mu) / sigma
        else:
            z = np.zeros_like(raw_w)

        # Min–Max 归一化到 [0,1]
        zmin, zmax = z.min(), z.max()
        if zmax > zmin:
            norm_w = ((z - zmin) / (zmax - zmin)).tolist()
        else:
            norm_w = [0.0] * len(z)

        return norm_w

    def output(self, episode, label='T',
            output_dir_norm="dataset/norm_dims/sim",
            orig_dir="dataset/dims/"):
        if label == 'V-VIS':
            # 1) 构造简化后的轨迹点列表
            start = 0
            sim_traj = []
            while start in self.F_ward:
                sim_traj.append(self.ori_traj_set[episode][start])
                start = self.F_ward[start][1]
            sim_traj.append(self.ori_traj_set[episode][start])

            # 2) 计算误差
            _, final_error, pct_error = F.sed_error(
                self.ori_traj_set[episode], sim_traj)

            norm_w = self.compute_weights(sim_traj)

            # 4) 输出归一化简化结果
            os.makedirs(output_dir_norm, exist_ok=True)
            fn_norm = os.path.join(output_dir_norm, f"sim_{episode}")
            with open(fn_norm, 'w') as f:
                for (pt, w) in zip(sim_traj, norm_w):
                    f.write(f"{pt[0]} {pt[1]} {w:.6f}\n")
            print(f"Saved normalized sim to {fn_norm}")

            # 5) 读取对应原始维度文件，构建 seq->原始值 映射
            path_orig = os.path.join(orig_dir, str(episode))
            orig_map = {}
            with open(path_orig, 'r') as rf:
                for line in rf:
                    v, s = line.strip().split()
                    orig_map[int(s)] = v

            # 6) 输出未归一化的简化结果
            output_dir_orig = os.path.join(orig_dir, "sim")
            os.makedirs(output_dir_orig, exist_ok=True)
            fn_orig = os.path.join(output_dir_orig, f"sim_{episode}")
            with open(fn_orig, 'w') as f:
                for (pt, w) in zip(sim_traj, norm_w):
                    seq = pt[1]
                    raw_v = orig_map[seq]
                    f.write(f"{raw_v} {seq} {w:.6f}\n")
            print(f"Saved raw sim to {fn_orig}")

            # 7) 打印误差并可视化
            print('Validation at episode {} with error {:.6e} (percentage {:.6e}%)'
                .format(episode, final_error, pct_error))
            F.draw(self.ori_traj_set[episode], sim_traj)

            return final_error, pct_error
        if label == 'V':
            start = 0
            sim_traj = []
            while start in self.F_ward:
                sim_traj.append(self.ori_traj_set[episode][start])
                start = self.F_ward[start][1]
                # print(sim_traj)
            sim_traj.append(self.ori_traj_set[episode][start])
            # print(sim_traj)
            _, final_error, pct_error = F.sed_error(self.ori_traj_set[episode], sim_traj)
            return final_error,pct_error
        if label == 'T':
            print('Training at episode {} with error {:.6e}'.format(episode, self.current))
            return self.current
