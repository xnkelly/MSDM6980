import os
from rl_env_inc import TrajComp
from rl_brain import PolicyGradient
import matplotlib.pyplot as plt

def evaluate(elist): # Evaluation
    effectiveness, eff_pct = [], []
    for episode in elist:
        #print('online episode', episode)
        buffer_size = int(ratio*len(env.ori_traj_set[episode]))
        if buffer_size < 3:
            continue
        steps, observation = env.reset(episode, buffer_size)
        for index in range(buffer_size, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            action = RL.quick_time_action(observation) #matrix implementation for fast efficiency when the model is ready
            observation_, _ = env.step(episode, action, index, done, 'V') #'T' means Training, and 'V' means Validation
            observation = observation_
        env_output_error, err_pct =env.output(episode, 'V-VIS', output_sim, orig_dir)    # 输出简化数据
        effectiveness.append(env_output_error) #'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
        eff_pct.append(err_pct)
    return sum(effectiveness)/len(effectiveness), sum(eff_pct)/len(eff_pct)

if __name__ == "__main__":

    ratio = input("请输入比率 ratio（浮点数，默认 0.1）：").strip()
    ratio = float(ratio) if ratio else 0.1
    
    file_prefix = input("请输入文件路径前缀（默认 '6980-main/'）：").strip() or './'
    #file_prefix = input("请输入文件路径前缀（默认 '6980-main/'）：").strip() or '6980-main/'

    model_prefix = file_prefix + 'save/'

    # 自动筛选 model_dir
    ratio_str = f"{ratio:.2f}"
    suffix = f"_ratio_{ratio_str}"
    # 列出所有候选目录
    cands = [d for d in os.listdir(model_prefix) if d.endswith(suffix)]
    if not cands:
        model_dir = input("请输入模型文件名：").strip()
    else: 
        # 提取前缀数值并选最小
        # 目录名形如 '2.210530e-01_ratio_0.10'
        pairs = []
        for d in cands:
            # d.split(suffix)[0] 就是 '2.210530e-01'
            try:
                val = float(d.split(suffix)[0])
            except ValueError:
                continue
            pairs.append((val, d))
        if not pairs:
            model_dir = input("请输入模型文件名：").strip()
        # 取最小的那个目录名
        _, model_dir = min(pairs, key=lambda x: x[0])
        print(f"自动选择 model_dir = {model_dir}")

    # building subtrajectory env 
    # model_dir = input("请输入模型文件名：").strip()
    model_path = model_prefix +  model_dir + '/'
    
    traj_path1 = 'dataset/norm_dims/'
    traj_path = file_prefix + traj_path1

    output_sim1 = 'dataset/norm_dims/sim'
    output_sim = file_prefix + output_sim1

    orig_dir1 = 'dataset/dims/'
    orig_dir = file_prefix + orig_dir1

    # 计算默认的 test_amount
    files = [f for f in os.listdir(traj_path)
            if os.path.isfile(os.path.join(traj_path, f))]
    default_test = len(files)
    print(f"简化轨迹文件数：{default_test}")

    # choose = input("是否需要调整参数（默认 'N'）：").strip()
    # if not choose:
    #     choose = 'N'
    # if choose=='N':
    #     test_amount = default_test
    # else: 
    #     test_amount = input("请输入评估轨迹数量（整数，默认使用数据维度）：").strip()
    #     test_amount = int(test_amount) if test_amount else default_test
    
    test_amount = default_test
    a_size = 3 #RLTS 3, RLTS-Skip 5
    s_size = 3 #RLTS and RLTS-Skip are both 3 online
    
    elist = [i for i in range(test_amount)]
    env = TrajComp(traj_path, test_amount, a_size, s_size)
    RL = PolicyGradient(env.n_features, env.n_actions)
    RL.load(model_path)
    #RL.load('./save/0.00490_ratio_0.20/') #your_trained_model your_trained_model_skip
    effectiveness, eff_pct = evaluate(elist) #evaluate evaluate_skip
    print("Error: {0:.6e}, percentage: {1:.6e}%".format(effectiveness, eff_pct))
