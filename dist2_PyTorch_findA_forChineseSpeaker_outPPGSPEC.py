import os
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
# from sklearn.cluster import KMeans
# from sklearn.neighbors import KDTree
from audio_10ms import hparams as audio_hparams
from audio_10ms import load_wav, wav2unnormalized_mfcc, wav2normalized_db_mel, wav2normalized_db_spec
from audio_10ms import write_wav, normalized_db_mel2wav, normalized_db_spec2wav
eps = 1e-6

# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 160,
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


PPG_DIM = 347 + 218
MEL_DIM = 80
SPEC_DIM = 201

en_all_cnt = 10
# en_all_cnt = 300000
# cn_all_cnt = 1
cn_all_cnt = 2000

# in
en_raw_list_path = '/datapool/home/hujk17/ppg_decode_spec_10ms_sch_Multi/meta_good_merge_ljspeech_v2_small_1_15.txt'
cn_raw_list_path = '/datapool/home/hujk17/ppg_decode_spec_10ms_sch_Multi/meta_good_merge_databaker_v2.txt'

English_PPG_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG/ppg_generate_10ms_by_audio_hjk2'
English_PPG_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG/ppg_generate_10ms_by_audio_hjk2'
Mandarin_PPG_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/LJSpeech-1.1-Mandarin-PPG/ppg_generate_10ms_by_audio_hjk2'
Mandarin_PPG_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/DataBaker-Mandarin-PPG/ppg_generate_10ms_by_audio_hjk2'
MEL_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG/mel_10ms_by_audio_hjk2'
MEL_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG/mel_10ms_by_audio_hjk2'
SPEC_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG/spec_10ms_by_audio_hjk2'
SPEC_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG/spec_10ms_by_audio_hjk2'

# out
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
en_final_cn_log_path = './en_final_cn_log_dist2_PyTorch_findA_forChineseSpeaker_outPPGSPE_' + STARTED_DATESTRING
if os.path.exists(en_final_cn_log_path) is False:
    os.makedirs(en_final_cn_log_path)
en_final_cn_idx_path = os.path.join(en_final_cn_log_path, 'en_final_cn_map.npy')


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


def get_single_data_pair(fname, speaker_id, english_ppg_dir, mandarin_ppg_dir, mel_dir, spec_dir):
    english_ppg_f = os.path.join(english_ppg_dir, fname+'.npy')
    mandarin_ppg_f = os.path.join(mandarin_ppg_dir, fname+'.npy')
    mel_f = os.path.join(mel_dir, fname+'.npy')
    spec_f = os.path.join(spec_dir, fname+'.npy')

    english_ppg = np.load(english_ppg_f)
    mandarin_ppg = np.load(mandarin_ppg_f)
    bilingual_ppg = np.concatenate((english_ppg, mandarin_ppg),axis=-1)
    # print('bilingual shape:', bilingual_ppg.shape)
    mel = np.load(mel_f)
    spec = np.load(spec_f)
    assert mel.shape[0] == bilingual_ppg.shape[0] and mel.shape[0] == spec.shape[0], fname + ' 维度不相等'
    assert mel.shape[1] == MEL_DIM and bilingual_ppg.shape[1] == PPG_DIM and spec.shape[1] == SPEC_DIM, fname + ' 特征维度不正确'
    return bilingual_ppg, mel, spec


def for_loop_en(): 
    en_file_list = en_text2list(file=en_raw_list_path)
    en_ppgs_ls = []
    en_mels_ls = []
    en_specs_ls = []
    for f in tqdm(en_file_list):
        ppgs, mels, specs = get_single_data_pair(f, 0, English_PPG_LJSpeech_DIR, Mandarin_PPG_LJSpeech_DIR, MEL_LJSpeech_DIR, SPEC_LJSpeech_DIR)
        en_ppgs_ls.extend(list(ppgs))
        en_mels_ls.extend(list(mels))
        en_specs_ls.extend(list(specs))
    return en_ppgs_ls, en_mels_ls, en_specs_ls


def for_loop_cn():
    cn_file_list = cn_text2list(file=cn_raw_list_path)
    cn_ppgs_ls = []
    cn_mels_ls = []
    cn_specs_ls = []
    for f in tqdm(cn_file_list):
        ppgs, mels, specs = get_single_data_pair(f, 1, English_PPG_DataBaker_DIR, Mandarin_PPG_DataBaker_DIR, MEL_DataBaker_DIR, SPEC_DataBaker_DIR)
        cn_ppgs_ls.extend(list(ppgs))
        cn_mels_ls.extend(list(mels))
        cn_specs_ls.extend(list(specs))
    return cn_ppgs_ls, cn_mels_ls, cn_specs_ls


def entropy_torch(p, logp, q, logq):
    ans = p * (logp - logq)
    ans = torch.sum(ans, dim=-1)
    print('ans shape:', ans, ans.shape)
    return ans
    

def bruce_find_closest_dist2KL_byPyTorch(en_vec_torch, cn_array_torch):
    # print('en_vec_torch:', en_vec_torch.shape)
    # print('cn_array_torch:', cn_array_torch.shape)
    # en_vec_torch = en_vec_torch.unsqueeze(0).repeat((cn_array_torch.shape[0], 1))
    a = en_vec_torch.unsqueeze(0)
    b = cn_array_torch
    loga = torch.log(a)
    logb = torch.log(b)
    ans_vec = entropy_torch(a, loga, b, logb) + entropy_torch(b, logb, a, loga)
    
    # ans_vec = F.kl_div(torch.log(en_vec_torch), cn_array_torch, reduction='none') +F.kl_div(torch.log(cn_array_torch), en_vec_torch, reduction='none')
    # print('ans shape:', ans_vec.shape)
    ans, ans_id = torch.min(ans_vec, 0)
    ans = ans.cpu().numpy()
    ans_id = ans_id.cpu().numpy()
    # print('numpy ans:', ans, ans.shape)
    # print('numpy ans_id:', ans_id, ans_id.shape)

    return ans, ans_id



# id list
def ppg_project(e_ppg_id, project_array):
    ans = list()
    for i in e_ppg_id:
        j = int(project_array[i] + eps)
        ans.append(j)
    print('cn num from 0:', ans[:10])
    return ans


def main():
    # 计时
    print('start program')
    program_time = time.time()
    last_time = time.time()

    # 读PPG
    en_ppg_l, en_mel_l, en_spec_l = for_loop_en()     #英文每一帧ppg的列表 en_l = [en_ppg1,en_ppg2,...]
    cn_ppg_l, cn_mel_l, cn_spec_l = for_loop_cn()     #中文每一帧ppg的列表 cn_l = [cn_ppg1,cn_ppg2,...]
    en_ppg_l_len = len(en_ppg_l)
    print('end put ppg in memory, use:', time.time() - last_time)
    last_time = time.time()

    # 放PPG-numpy
    en_ppg_l = np.asarray(en_ppg_l)
    cn_ppg_l = np.asarray(cn_ppg_l)
    print('-----change to np-array, use:', time.time() - last_time)
    last_time = time.time()


    with torch.no_grad():
        # 放PPG-PyTorch-GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print('---------before:', type(en_ppg_l[0][0]), en_ppg_l[0][0])
        en_ppg_l = torch.from_numpy(en_ppg_l).float().to(device)
        cn_ppg_l = torch.from_numpy(cn_ppg_l).float().to(device)
        # print('---------after:', type(en_ppg_l[0][0]), en_ppg_l[0][0])
        print('end change to GPU, use:', time.time() - last_time)
        last_time = time.time()

        # en-PPG找最近cn-PPG
        print('start get cloest map array for all en ppg by PyTorch')
        last_time = time.time()
        en_final_cn_idx = np.zeros((en_ppg_l_len))
        for i in tqdm(range(en_ppg_l_len)): 
            iterator_time = time.time()
            _ans, ans_id = bruce_find_closest_dist2KL_byPyTorch(en_ppg_l[i], cn_ppg_l)
            print('once PPG once time use:', time.time() - iterator_time)
            en_final_cn_idx[i] = ans_id
        cn_ppg_l = cn_ppg_l.cpu().numpy()
        torch.cuda.empty_cache()
    np.save(en_final_cn_idx_path, en_final_cn_idx)
    print('end using PyTorch to get map array, all use:', time.time() - last_time)
    last_time = time.time()


    # 开始findA的部分
    print('start findA')
    en_file_list = en_text2list(file=en_raw_list_path)
    now = 0
    for f in tqdm(en_file_list):
        ppgs, mels, specs = get_single_data_pair(f, 0, English_PPG_LJSpeech_DIR, Mandarin_PPG_LJSpeech_DIR, MEL_LJSpeech_DIR, SPEC_LJSpeech_DIR)
        e_ppg_id = [] # 英文从零开始
        for i in range(ppgs.shape[0]):
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
        np.save(os.path.join(en_final_cn_log_path, save_ppg_name_projected), c_ppgs_projected)

        # 找到mel并存储
        c_mels_projected = list()
        for i in c_ppg_id_projected:
            c_mels_projected.append(cn_mel_l[i])
        c_mels_projected = np.asarray(c_mels_projected)
        save_mel_name_projected = f + '_cn_mel_projected.npy'
        np.save(os.path.join(en_final_cn_log_path, save_mel_name_projected), c_mels_projected)

        # 找到spec并存储
        c_specs_projected = list()
        for i in c_ppg_id_projected:
            c_specs_projected.append(cn_spec_l[i])
        c_specs_projected = np.asarray(c_specs_projected)
        save_spec_name_projected = f + '_cn_spec_projected.npy'
        np.save(os.path.join(en_final_cn_log_path, save_spec_name_projected), c_specs_projected)

        # 计算音频wav并存储
        save_wav_name_projected = f + '_cn_wav_projected.wav'
        write_wav(os.path.join(en_final_cn_log_path, save_wav_name_projected), normalized_db_spec2wav(c_specs_projected))
        

        #----------接下来是original的ppg，spec，wav存储，用来对比-----------

        # 找到ppg并存储
        save_ppg_name_original = f + '_en_ppg_original.npy'
        np.save(os.path.join(en_final_cn_log_path, save_ppg_name_original), ppgs)

        # 找到mel并存储
        save_mel_name_original = f + '_en_mel_original.npy'
        np.save(os.path.join(en_final_cn_log_path, save_mel_name_original), mels)

        # 找到spec并存储
        save_spec_name_original = f + '_en_spec_original.npy'
        np.save(os.path.join(en_final_cn_log_path, save_spec_name_original), specs)

        # 计算音频wav并存储-original
        save_wav_name_original = f + '_en_wav_original.wav'
        write_wav(os.path.join(en_final_cn_log_path, save_wav_name_original), normalized_db_spec2wav(specs))


    print('end findA, use:', time.time() - last_time)
    print('program use:', time.time() - program_time)


if __name__ == '__main__':
    main()