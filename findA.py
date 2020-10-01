import os
import numpy as np
from tqdm import tqdm
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


cn_raw_list_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_small.txt'
cn_raw_ppg_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/ppg_from_generate_batch'
cn_raw_linear_dir ='/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/spec_5ms_by_audio_2'

en_raw_list_path = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/meta_small.txt'
en_raw_ppg_path = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/ppg_from_generate_batch'
en_raw_linear_dir = '/datapool/home/hujk17/chenxueyuan/LJSpeech-1.1/spec_5ms_by_audio_2'

en_final_cn_log_path = '/datapool/home/hujk17/chenxueyuan/en_final_cn_log'
en_final_cn_idx_path = os.path.join(en_final_cn_log_path, 'en_final_cn_idx.npy')


# 写
projected_wav_dir = 'projected_wavs_16000'
if os.path.exists(projected_wav_dir) is False:
    os.makedirs(projected_wav_dir)



Linear_DIM = 201
PPG_DIM = 345                                              #每一帧ppg的维度



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
    en_linears_ls = []
    for f in en_file_list:
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=en_raw_ppg_path, linears_dir=en_raw_linear_dir)
        # 需要确认下
        # en_ppgs_ls.extend(list(wav_ppgs))
        # 或者
        for i in range(wav_ppgs.shape[0]):
            # ppg[i]
            en_ppgs_ls.append(wav_ppgs[i])
            en_linears_ls.append(linears[i])
            # find_jin(ppg[i])
    # shuffule
    # wav_id, frame_id
    return en_ppgs_ls, en_linears_ls


def for_loop_cn():                                         #得到每一帧的中文ppg列表
    cn_file_list = cn_text2list(file=cn_raw_list_path)
    cn_ppgs_ls = []
    cn_linears_ls = []
    for f in cn_file_list:
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=cn_raw_ppg_path, linears_dir=cn_raw_linear_dir)
        # 需要确认下
        # en_ppgs_ls.extend(list(wav_ppgs))en_raw_data_path
        # 或者
        for i in range(wav_ppgs.shape[0]):
            # ppg[i]
            cn_ppgs_ls.append(wav_ppgs[i])
            cn_linears_ls.append(linears[i])
            # find_jin(ppg[i])
    # shuffule
    # wav_id, frame_id
    return cn_ppgs_ls, cn_linears_ls


# id list
def ppg_project(e_ppg_id, project_array):
    ans = list()
    for i in e_ppg_id:
        j = int(project_array[i] + eps)
        ans.append(j)
    print('cn num from 0:', ans[:10])
    return ans


def main():
    # 
    en_ppg_l, en_linear_l = for_loop_en()     #英文每一帧ppg的列表 en_l = [en_ppg1,en_ppg2,...]
    cn_ppg_l, cn_linear_l = for_loop_cn()     #中文每一帧ppg的列表 cn_l = [cn_ppg1,cn_ppg2,...]
    all_ppg_l = en_ppg_l + cn_ppg_l          #中英文混合后的ppg的列表

    #
    en_final_cn_idx = np.load(en_final_cn_idx_path)

    #
    en_file_list = en_text2list(file=en_raw_list_path)
    # en_ppgs_ls = []
    now = 0
    for f in tqdm(en_file_list):
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=en_raw_ppg_path, linears_dir=en_raw_linear_dir)
        e_ppg_id = [] # 英文从零开始
        for i in range(wav_ppgs.shape[0]):
            e_ppg_id.append(now)
            now += 1
        print('en id from 0:', e_ppg_id[:10])
        c_ppg_id_projected = ppg_project(e_ppg_id, en_final_cn_idx) # 从中文的零开始

        # 找到linear
        c_lineas_projected = list()
        for i in c_ppg_id_projected:
            c_lineas_projected.append(cn_linear_l[i])
        c_lineas_projected = np.asarray(c_lineas_projected)
        save_linear_name = f + '_cn_linear_projected.wav'
        write_wav(os.path.join(projected_wav_dir, save_linear_name), normalized_db_spec2wav(c_lineas_projected))
        save_linear_original_name = f + '_en_linear_original.wav'
        write_wav(os.path.join(projected_wav_dir, save_linear_original_name), normalized_db_spec2wav(linears))
        

        # break


        
    
    

    
        
    


if __name__ == '__main__':
    main()