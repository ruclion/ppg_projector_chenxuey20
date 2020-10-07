import os
import numpy as np
from tqdm import tqdm
import time
from sklearn.cluster import KMeans

cn_raw_list_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_small.txt'
cn_raw_ppg_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/ppg_from_generate_batch'
cn_raw_linear_dir ='/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/spec_5ms_by_audio_2'

en_raw_list_path = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/meta_small.txt'
en_raw_ppg_path = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/ppg_from_generate_batch'
en_raw_linear_dir = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/spec_5ms_by_audio_2'

en_final_cn_log_path = '/datapool/home/hujk17/chenxueyuan/en_final_cn_log_small'
if os.path.exists(en_final_cn_log_path) is False:
    os.makedirs(en_final_cn_log_path)
en_final_cn_idx_path = os.path.join(en_final_cn_log_path, 'en_final_cn_idx_small.npy')

# f = open(en_final_cn_idx, 'w')
# f......
# np.save(en_final_cn_idx, ...)


Linear_DIM = 201
PPG_DIM = 345                                              #每一帧ppg的维度
# !!!!!!
K_small = 15     #类
K = 20000     #类

def en_text2list(file):                                       #封装读出每一句英文ppg文件名的函数，输入文本，得到每一句ppg文件名序列的列表
    en_file_list = []
    with open(file, 'r') as f:
        for line in f.readlines():
            # !!!!!!!!!!!!!!!!
            en_file_list.append(line.strip().split('|')[0])
    print('en:', en_file_list)
    return en_file_list


# 000001	那些#1庄稼#1田园#2在#1果果#1眼里#2感觉#1太亲切了#4
#	na4 xie1 zhuang1 jia5 tian2 yuan2 

def cn_text2list(file):                                #封装读出每一句中文ppg文件名的函数，输入文本，得到每一句ppg文件名序列的列表
    cn_file_list = []
    with open(file, 'r') as f:
        a = [i.strip() for i in f.readlines()]
        print(a[0])
        print(a[1])
        i = 0
        while i < len(a):
            fname = a[i][:6]
            cn_file_list.append(fname)
            i += 2
    print('cn:', cn_file_list)
    return cn_file_list




def get_single_data_pair(fname, ppgs_dir, linears_dir):           #输入每一句的文件名，ppg的地址，线性谱的地址，得到每一句的ppg和linear
    assert os.path.isdir(ppgs_dir) and os.path.isdir(linears_dir)

    # mfcc_f = os.path.join(os.path.join(os.path.join(mfcc_dir, fname.split('-')[0]),fname.split('-')[1]),fname+'.npy')#fname+'.npy')
    ppg_f = os.path.join(ppgs_dir, fname+'.npy')#os.path.join(ppg_dir, fname+'.npy')
    linear_f = os.path.join(linears_dir, fname+'.npy')#os.path.join(ppg_dir, fname+'.npy')
    ppg = np.load(ppg_f)
    linear = np.load(linear_f)
    # ppg = onehot(ppg, depth=PPG_DIM)
    assert ppg.shape[0] == linear.shape[0],fname+' 维度不相等'
    assert ppg.shape[1] == PPG_DIM and linear.shape[1] == Linear_DIM
    return ppg, linear



def for_loop_en():                                         #得到每一帧的英文ppg列表
    en_file_list = en_text2list(file=en_raw_list_path)
    en_ppgs_ls = []
    for f in en_file_list:
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=en_raw_ppg_path, linears_dir=en_raw_linear_dir)
        # 需要确认下
        # en_ppgs_ls.extend(list(wav_ppgs))
        # 或者
        for i in range(wav_ppgs.shape[0]):
            # ppg[i]
            en_ppgs_ls.append(wav_ppgs[i])
            # find_jin(ppg[i])
    # shuffule
    # wav_id, frame_id
    return en_ppgs_ls


def for_loop_cn():                                         #得到每一帧的中文ppg列表
    cn_file_list = cn_text2list(file=cn_raw_list_path)
    cn_ppgs_ls = []
    for f in cn_file_list:
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=cn_raw_ppg_path, linears_dir=cn_raw_linear_dir)
        # 需要确认下
        # en_ppgs_ls.extend(list(wav_ppgs))en_raw_data_path
        # 或者
        for i in range(wav_ppgs.shape[0]):
            # ppg[i]
            cn_ppgs_ls.append(wav_ppgs[i])
            # find_jin(ppg[i])
    # shuffule
    # wav_id, frame_id
    return cn_ppgs_ls


def dist(ppg_e, ppg_c):                                 #分别输入一帧中英文ppg，返回二者距离
    # # array, 345 dim
    # assert ppg_c.shape[0] == 345
    # ans = 0
    # for i in range(345):
    #     ans += (ppg_e[i] - ppg_c[i]) ** 2
    # ans = ans ** 0.5
    # print(ppg_e.shape)
    ans = np.linalg.norm(ppg_e - ppg_c)
    return ans

    
    
def cluster_kmeans(all, K):                               #聚类，输入是所有帧的ppg列表all,K类,输出是每一类的ppg列表
    # a = [a1, a2, ...], y = [label, label,...]
    #kmeans = KMeans(n_clusters=K, random_state=0).fit(all)
    class_as_index = KMeans(n_clusters=K, random_state=0).fit_predict(all)
    #class_as_index = k_means(a, K)
    return class_as_index


'''
def hjk_main1():                                         #
    # in cn & in en
    en_l = []
    cn_l = []

    en_final_cn_idx = np.zeros((len(en_l))) # a[1000000]
    for i, e in enumerate(en_l):
        ans = 1e100
        ans_id = -1
        ans_id_etc = -1
        for j, c in enumerate(cn_l):
            if dist(e, c) < ans:
                ans = dist(e, c)
                ans_id = c
                ans_id_etc = c
        en_final_cn_idx[i] = ans_id
    np.save('en_final_cn_idx', en_final_cn_idx)
'''


def main():
   
    en_l = for_loop_en()     #英文每一帧ppg的列表 en_l = [en_ppg1,en_ppg2,...]
    cn_l = for_loop_cn()     #中文每一帧ppg的列表 cn_l = [cn_ppg1,cn_ppg2,...]
    all_l = en_l + cn_l          #中英文混合后的ppg的列表
    
    # 需要快速的聚类                        #all_l=[en_ppg1,en_ppg2,...,cn_ppg1,cn_ppg2,...]
    all_class = cluster_kmeans(all_l, K_small)   #all_class=[en_label,en_label,...,cn_label,cn_label,...]

    #... a[100], a[0].1, 2, 3,...  
    class_cn_ppgs = list()                      #建立一个列表class_cn_ppgs，列表中包含K个空列表class_cn_ppg = [[],[],[],...]
    for i in range(K_small):
        l = list()
        class_cn_ppgs.append(l)    #append()在列表后面添加元素

    # int a[10][10];
    # a[0][0] = 888
    # a[0][1] = 999
    # a[0][2] = -1
    # a[1][0] = 777
    
    # 构造类的信息, 筛选出每个类里都有哪些中文的ppg; 并且平均每个类有100个中文ppg
    for i in range(len(cn_l)):
        idx = i + len(en_l)
        now_class = all_class[idx]                #now_class = cn_label  可能是0-1999
        class_cn_ppgs[now_class].append(i)        #class_cn_ppg = [[2,8,19,...],[3,48,79,...],[4,5,36,...],...] 2000个类，每个类中含有cn_l中对应帧ppg的序列号
    # 看下哪些类中没有中文的PPG
    have_cnt = 0
    for i in range(K_small):
        l = len(class_cn_ppgs[i])
        if l > 0:
            have_cnt += 1
            print('-----:', i)
    print('have class:', have_cnt)

    # 开始寻找en对应的类内所有中文ppg离他最近的
    en_final_cn_idx = np.zeros((len(en_l))) # a[1000000] np.zeros()返回来一个给定形状和类型的用0填充的数组；
    for i in tqdm(range(len(en_l))):                   #遍历英文ppg列表，
        now_class = all_class[i]                   #now_class = en_label  可能是0-1999 
        ans = 1e100     #科学计数法  1*10的100次方
        ans_id = -1     
        # ans_id_etc = -1
        for j in class_cn_ppgs[now_class]:      #class_cn_ppgs[now_class] = [2,8,19,...]或[3,48,79,...]或[4,5,36,...]或... eg,[2,8,19,...]
            e = en_l[i]                         #e = en_l[0], en_l[1], en_l[2], ...
            c = cn_l[j]                         #c = cn_l[2], cn_l[8], ...
            if dist(e, c) < ans:
                ans = dist(e, c)
                ans_id = j                      #cn_id  距离每一个en_ppg最近的cn_ppg的帧id
                # ans_id_etc = c                  #cn_l[j] 
        # 已经找到最接近的了，记下来
        en_final_cn_idx[i] = ans_id
        
    np.save(en_final_cn_idx_path, en_final_cn_idx)


if __name__ == '__main__':
    main()