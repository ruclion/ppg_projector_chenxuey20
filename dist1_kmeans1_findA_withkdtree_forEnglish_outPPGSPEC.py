# 为了验证代码正确性，找ppg英文自己的top3的第3个, 然后把所有的ppg和linear和wav都存起来，共之后对比NN版本

import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from audio import hparams as audio_hparams
from audio import load_wav, wav2unnormalized_mfcc, wav2normalized_db_mel, wav2normalized_db_spec
from audio import write_wav, normalized_db_mel2wav, normalized_db_spec2wav
eps = 1e-6

# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 80,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  
    'min_db': -80.0,  
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,  
    'silence_db': -28.0,
    'center': True,
}


assert hparams == audio_hparams

Linear_DIM = 201
PPG_DIM = 345
# !!!!!!
K_small = 1     #类
K = 20000     #类

en_all_cnt = 3
# cn_all_cnt = 12
cn_all_cnt = 12000


cn_raw_list_path = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/meta_good.txt'
cn_raw_ppg_path = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/ppg_from_generate_batch'
cn_raw_linear_dir ='/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/spec_5ms_by_audio_2'

en_raw_list_path = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/meta_good.txt'
en_raw_ppg_path = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/ppg_from_generate_batch'
en_raw_linear_dir = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/spec_5ms_by_audio_2'


# 写
en_final_cn_log_path = '/datapool/home/hujk17/chenxueyuan/en_final_cn_log_withkdtree_kmeans1_findA_dist1_forEnglish_outPPGSPE'
if os.path.exists(en_final_cn_log_path) is False:
    os.makedirs(en_final_cn_log_path)
en_final_cn_idx_path = os.path.join(en_final_cn_log_path, 'en_final_cn_idx_withkdtree_kmeans1_findA_dist1_forEnglish_outPPGSPE.npy')
# 写
projected_wav_dir = '/datapool/home/hujk17/chenxueyuan/projected_wavs_16000_withkdtree_kmeans1_findA_dist1_forEnglish_outPPGSPE'
if os.path.exists(projected_wav_dir) is False:
    os.makedirs(projected_wav_dir)



def en_text2list(file):  
    en_file_list = []
    global en_all_cnt
    with open(file, 'r') as f: 
        for i, line in enumerate(f.readlines()):
            en_file_list.append(line.strip())
            if i >= en_all_cnt - 1:
                break
    print('en len:', len(en_file_list), 'en:', en_file_list[:min(3, en_all_cnt)])
    return en_file_list


# 000001	那些#1庄稼#1田园#2在#1果果#1眼里#2感觉#1太亲切了#4
#	na4 xie1 zhuang1 jia5 tian2 yuan2 

def cn_text2list(file):
    cn_file_list = []
    global cn_all_cnt
    with open(file, 'r') as f:
        a = [i.strip() for i in f.readlines()]
        i = 0
        while i < len(a):
            fname = a[i]
            cn_file_list.append(fname)
            i += 1
            if i >= cn_all_cnt:
                break
    print('cn len:', len(cn_file_list), 'cn:', cn_file_list[:min(3, cn_all_cnt)])
    return cn_file_list




def get_single_data_pair(fname, ppgs_dir, linears_dir):
    assert os.path.isdir(ppgs_dir) and os.path.isdir(linears_dir)

    ppg_f = os.path.join(ppgs_dir, fname+'.npy')
    linear_f = os.path.join(linears_dir, fname+'.npy')
    ppg = np.load(ppg_f)
    linear = np.load(linear_f)
    assert ppg.shape[0] == linear.shape[0],fname+' 维度不相等'
    assert ppg.shape[1] == PPG_DIM and linear.shape[1] == Linear_DIM
    return ppg, linear



def for_loop_en(): 
    en_file_list = en_text2list(file=en_raw_list_path)
    en_ppgs_ls = []
    en_linears_ls = []
    for f in tqdm(en_file_list):
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=en_raw_ppg_path, linears_dir=en_raw_linear_dir)
        en_ppgs_ls.extend(list(wav_ppgs))
        en_linears_ls.extend(list(linears))
    return en_ppgs_ls, en_linears_ls


def for_loop_cn():
    cn_file_list = cn_text2list(file=cn_raw_list_path)
    cn_ppgs_ls = []
    cn_linears_ls = []
    for f in tqdm(cn_file_list):
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=cn_raw_ppg_path, linears_dir=cn_raw_linear_dir)
        cn_ppgs_ls.extend(list(wav_ppgs))
        cn_linears_ls.extend(list(linears))
    return cn_ppgs_ls, cn_linears_ls


def dist(ppg_e, ppg_c):
    ans = np.linalg.norm(ppg_e - ppg_c)
    return ans

    
def cluster_kmeans(all, K): 
    class_as_index = KMeans(n_clusters=K, random_state=0).fit_predict(all)
    return class_as_index


def bruce_find_closest(i, now_class, en_l, cn_l, class_cn_ppgs):
    ans = 1e100 
    ans_id = -1     
    for j in class_cn_ppgs[now_class]:      #class_cn_ppgs[now_class] = [2,8,19,...]或[3,48,79,...]或[4,5,36,...]或... eg,[2,8,19,...]
        e = en_l[i]                         #e = en_l[0], en_l[1], en_l[2], ...
        c = cn_l[j]                         #c = cn_l[2], cn_l[8], ...
        dist_e_c = dist(e, c)
        if dist_e_c < ans:
            ans = dist_e_c
            ans_id = j                      #cn_id  距离每一个en_ppg最近的cn_ppg的帧id
    return ans, ans_id

def kdtree_find_closest(i, en_l, now_class_cn_ppgs_value_kdtree, now_class_cn_ppgs):
    e = en_l[i]
    e_2d = np.expand_dims(e, axis=0)
    # print(e_2d.shape)
    dist, ind = now_class_cn_ppgs_value_kdtree.query(e_2d, k=3)
    print(dist, ind)
    dist = dist[0][2]
    ind = ind[0][2]
    real_ind = now_class_cn_ppgs[ind]
    return dist, real_ind


# id list
def ppg_project(e_ppg_id, project_array):
    ans = list()
    for i in e_ppg_id:
        j = int(project_array[i] + eps)
        ans.append(j)
    print('cn num from 0:', ans[:10])
    return ans


def main():
    print('start program')
    program_time = time.time()
    last_time = time.time()

    en_ppg_l, en_linear_l = for_loop_en()     #英文每一帧ppg的列表 en_l = [en_ppg1,en_ppg2,...]
    cn_ppg_l, cn_linear_l = for_loop_cn()     #中文每一帧ppg的列表 cn_l = [cn_ppg1,cn_ppg2,...]
    all_ppg_l = en_ppg_l + cn_ppg_l          #中英文混合后的ppg的列表
    print('end put ppg in memory, use:', time.time() - last_time)
    last_time = time.time()

    print('start cluster...')
    # 需要快速的聚类                        #all_l=[en_ppg1,en_ppg2,...,cn_ppg1,cn_ppg2,...]
    all_class = cluster_kmeans(all_ppg_l, K_small)   #all_class=[en_label,en_label,...,cn_label,cn_label,...]
    print('end cluster..., k-means use:', time.time() - last_time)
    last_time = time.time()
    
    #... a[100], a[0].1, 2, 3,...  
    class_cn_ppgs = list()                      #建立一个列表class_cn_ppgs，列表中包含K个空列表class_cn_ppg = [[],[],[],...]
    class_cn_ppgs_value = list()
    class_cn_ppgs_value_kdtree = list()
    for i in range(K_small):
        l = list()
        class_cn_ppgs.append(l)    #append()在列表后面添加元素
        l_value = list()
        class_cn_ppgs_value.append(l_value)
        

    # 构造类的信息, 筛选出每个类里都有哪些中文的ppg; 并且平均每个类有100个中文ppg
    en_ppg_l_len = len(en_ppg_l)
    for i in range(len(cn_ppg_l)):
        idx = i + en_ppg_l_len
        now_class = all_class[idx]                #now_class = cn_label  可能是0-1999
        class_cn_ppgs[now_class].append(i)        #class_cn_ppg = [[2,8,19,...],[3,48,79,...],[4,5,36,...],...] 2000个类，每个类中含有cn_l中对应帧ppg的序列号
        class_cn_ppgs_value[now_class].append(cn_ppg_l[i])
        
    print('prepare for class infomation use:', time.time() - last_time)
    print('start construct kdtree')
    all_last_time = time.time()
    have_cnt = 0
    for i in tqdm(range(K_small)):
        l = len(class_cn_ppgs[i])
        if l > 0:
            have_cnt += 1
            print('cluster', i, 'len', l, 'start construct kd-tree')
            last_time = time.time()
            
            class_cn_ppgs_value[i] = np.asarray(class_cn_ppgs_value[i])
            class_cn_ppgs_value_kdtree.append(KDTree(class_cn_ppgs_value[i], leaf_size=40) )

            print('end cluster', i, 'kd-tree use:', time.time() - last_time)
    print('have class:', have_cnt)
    print('end construct all kdtrees, tot use:', time.time() - all_last_time)

    

    # 开始寻找en对应的类内所有中文ppg离他最近的
    print('start get cloest map array for all en ppg')
    last_time = time.time()
    en_final_cn_idx = np.zeros((en_ppg_l_len)) # a[1000000] np.zeros()返回来一个给定形状和类型的用0填充的数组；
    for i in tqdm(range(en_ppg_l_len)):                   #遍历英文ppg列表，
        now_class = all_class[i]                   #now_class = en_label  可能是0-1999 

        # 暴力寻找
        # ans1, ans_id1 = bruce_find_closest(i, now_class, en_ppg_l, cn_ppg_l, class_cn_ppgs)

        # k-d tree寻找
        ans, ans_id = kdtree_find_closest(i, en_ppg_l, class_cn_ppgs_value_kdtree[now_class], class_cn_ppgs[now_class])
        # assert np.absolute(ans1 - ans) < eps and ans_id1 == ans_id
        en_final_cn_idx[i] = ans_id
        
    np.save(en_final_cn_idx_path, en_final_cn_idx)
    print('end write map array, all use:', time.time() - last_time)


    # 开始findA的部分
    print('start findA')
    last_time = time.time()

    en_file_list = en_text2list(file=en_raw_list_path)
    now = 0
    for f in tqdm(en_file_list):
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=en_raw_ppg_path, linears_dir=en_raw_linear_dir)
        e_ppg_id = [] # 英文从零开始
        for i in range(wav_ppgs.shape[0]):
            e_ppg_id.append(now)
            now += 1
        print('en id from 0:', e_ppg_id[:10])
        c_ppg_id_projected = ppg_project(e_ppg_id, en_final_cn_idx) # 从中文的零开始

        # 找到ppg并存储
        c_ppgs_projected = list()
        for i in c_ppg_id_projected:
            c_ppgs_projected.append(cn_ppg_l[i])
        c_ppgs_projected = np.asarray(c_ppgs_projected)
        save_ppg_name_projected = f + '_cn_ppg_projected.npy'
        np.save(os.path.join(projected_wav_dir, save_ppg_name_projected), c_ppgs_projected)

        # 找到linear并存储
        c_lineas_projected = list()
        for i in c_ppg_id_projected:
            c_lineas_projected.append(cn_linear_l[i])
        c_lineas_projected = np.asarray(c_lineas_projected)
        save_linear_name_projected = f + '_cn_linear_projected.npy'
        np.save(os.path.join(projected_wav_dir, save_linear_name_projected), c_lineas_projected)

        # 计算音频wav并存储
        save_wav_name_projected = f + '_cn_wav_projected.wav'
        write_wav(os.path.join(projected_wav_dir, save_wav_name_projected), normalized_db_spec2wav(c_lineas_projected))
        

        #----------接下来是original的ppg，linear，wav存储，用来对比-----------

        # 找到ppg并存储
        save_ppg_name_original = f + '_en_ppg_original.npy'
        np.save(os.path.join(projected_wav_dir, save_ppg_name_original), wav_ppgs)

        # 找到linear并存储
        save_linear_name_original = f + '_en_linear_original.npy'
        np.save(os.path.join(projected_wav_dir, save_linear_name_original), linears)

        # 计算音频wav并存储-original
        save_wav_name_original = f + '_en_wav_original.wav'
        write_wav(os.path.join(projected_wav_dir, save_wav_name_original), normalized_db_spec2wav(linears))


    print('end findA, use:', time.time() - last_time)
    print('program use:', time.time() - program_time)


if __name__ == '__main__':
    main()