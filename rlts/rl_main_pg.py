import os
from rl_env_inc import TrajComp
from rl_brain import PolicyGradient
import matplotlib.pyplot as plt
import time

def run_online(elist): # Validation
    eva,pct_eva = [],[]
    total_len = []
    for episode in elist:
        #print('online episode', episode)
        total_len.append(len(env.ori_traj_set[episode]))
        buffer_size = int(ratio*len(env.ori_traj_set[episode]))
        if buffer_size < 3:
            continue
        steps, observation = env.reset(episode, buffer_size)
        for index in range(buffer_size, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            action = RL.pro_choose_action(observation)
            #action = RL.quick_time_action(observation) #use it when your model is ready for efficiency
            observation_, _ = env.step(episode, action, index, done, 'V') #'T' means Training, and 'V' means Validation
            observation = observation_
        env_out_put_err, pct_error=env.output(episode, 'V')
        eva.append(env_out_put_err) 
        pct_eva.append(pct_error)
    return eva,pct_eva
        
def run_comp(): #Training
    check = 999999
    check_pct = 1
    training = []
    validation = []
    valid_pct = []
    Round = 10
    while Round!=0:
        Round = Round - 1
        for episode in range(0, traj_amount):
            #print('training: ', episode)
            buffer_size = int(ratio*len(env.ori_traj_set[episode]))
            # extreme cases
            if buffer_size < 3:
                print("buffer_size < 3")
                continue
            steps, observation = env.reset(episode, buffer_size)
            for index in range(buffer_size, steps):
                #print('index', index)
                if index == steps - 1:
                    done = True
                else:
                    done = False
                
                # RL choose action based on observation
                action = RL.pro_choose_action(observation)
                #print('action', action)
                # RL take action and get next observation and reward
                observation_, reward = env.step(episode, action, index, done, 'T') #'T' means Training, and 'V' means Validation
                
                RL.store_transition(observation, action, reward)
                
                if done:
                    vt = RL.learn()
                    break
                # swap observation
                observation = observation_
            train_e = env.output(episode, 'T') #'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
            show = dims-1
            if episode == show:
                #eva,pct_eva = run_online([i for i in range(traj_amount, traj_amount + valid_amount)])
                eva,pct_eva = run_online([i for i in range(0, traj_amount + valid_amount)])
                print('eva: ', eva, 'eva pencentage: ', pct_eva)
                res = sum(eva)/len(eva)
                res_pct = sum(pct_eva)/len(pct_eva)
                training.append(train_e)
                validation.append(res)
                valid_pct.append(res_pct)
                print('Training error: {:.6e}, Validation error: {:.6e}, Validation percentage error: {:.6e}%'.format(sum(training[-show:])/len(training[-show:]), res, res_pct))
                #RL.save('./save/'+ str(res) + '_ratio_' + str(ratio) + '/trained_model.ckpt')
                if res < check:
                    check = res
                    check_pct = res_pct
                    print('==>update current best model {}, percentage {:.6e}% with ratio {}'.format(check, check_pct, ratio))
                    model_dir = model_prefix + '{:.6e}_ratio_{:.2f}'.format(res, ratio)
                    os.makedirs(model_dir, exist_ok=True)
                    RL.save(os.path.join(model_dir, 'trained_model.ckpt'))
                    print('Save model at round {} episode {} with error {:.6e}, percentage error {:.6e}%'.format(10 - Round, episode, res, res_pct))
    print('Best model is {}, percentage {:.6e}% with ratio {}'.format(check, check_pct, ratio))
    return training, validation, valid_pct

if __name__ == "__main__":

    ratio = input("请输入比率 ratio（浮点数，默认 0.2）：").strip()
    ratio = float(ratio) if ratio else 0.2
    
    file_prefix = input("请输入文件路径前缀（默认 '6980-main/'）：").strip()
    if not file_prefix:
        file_prefix = '6980-main/'
        # file_prefix = './'

    traj_path1 = 'dataset/norm_dims/'
    traj_path = file_prefix + traj_path1

    model_dir1 = 'save/'
    model_prefix = file_prefix + model_dir1

    absolute_traj_path = os.path.abspath(traj_path)
    if not os.path.exists(absolute_traj_path):
        print("Path not exists: ", absolute_traj_path)
        raise FileNotFoundError()
    
    files = [f for f in os.listdir(traj_path)
             if os.path.isfile(os.path.join(traj_path, f))]
    dims = len(files)
    print(f"数据维度： {dims}")

    # choose = input("是否需要调整参数（默认 'N'）：").strip()
    # if not choose:
    #     choose = 'N'
    # if choose=='N':
        
    # else: 
    #     # traj_amount = input("请输入训练轨迹数量（整数，默认使用数据维度）：").strip()
    #     # traj_amount = int(traj_amount) if traj_amount else dims

    #     # valid_amount = input("请输入验证轨迹数量（整数，默认 1）：").strip()
    #     # valid_amount = int(valid_amount) if valid_amount else 1

    #     # a_size = input("请输入动作空间维度 a_size（整数，默认 3）：").strip()
    #     # a_size = int(a_size) if a_size else 3

    #     # s_size = input("请输入状态空间维度 s_size（整数，默认 3）：").strip()
    #     # s_size = int(s_size) if s_size else 3
    
    traj_amount = dims
    valid_amount = 0
    a_size = 3
    s_size = 3

    env = TrajComp(traj_path, traj_amount + valid_amount, a_size, s_size)
    RL = PolicyGradient(env.n_features, env.n_actions)
    #RL.load('./save/your_model/')
    start = time.time()
    training, validation, valid_pct = run_comp()
    print("Training elapsed time = ", float(time.time() - start))

