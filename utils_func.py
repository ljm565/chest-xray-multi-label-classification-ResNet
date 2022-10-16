import torch
import pandas as pd
import os
import string
import sys
import numpy as np
import shutil
from tqdm import tqdm



def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')



def check_data(base_path):
    try:
        assert os.path.isfile(base_path+'/data/mimic-cxr-2.0.0-metadata.csv') and os.path.isfile(base_path+'/data/mimic-cxr-2.0.0-negbio.csv')
    except:
        print('\n'+'*'*36)
        print("Error: Check the meta, negbio data")
        print('*'*36+'\n')
        sys.exit()

    

def make_dataframe(meta_path, negbio_path):
    meta = pd.read_csv(meta_path, dtype=str, usecols=['dicom_id', 'subject_id', 'study_id', 'ViewPosition'])
    negbio = pd.read_csv(negbio_path, dtype=str).iloc[:, 1:]
    return meta, negbio



def make_img_data(meta):
    # make col info to list
    dicom_id = meta['dicom_id'].tolist()
    subject_id = meta['subject_id'].tolist()
    study_id = meta['study_id'].tolist()
    ViewPosition = meta['ViewPosition'].tolist()

    # sanity check
    assert len(dicom_id) == len(subject_id) == len(study_id) == len(ViewPosition)

    # select one study_id per subject_id with an AP image
    data = {}
    for sub, stu, pos in zip(subject_id, study_id, ViewPosition):
        if pos == 'AP':
            try:
                data[sub].append(stu)
                tmp = sorted(data[sub])
                data[sub] = [tmp[0]]
            except KeyError:
                data[sub] = [stu]

    # select study_id per one AP image
    for sub, stu, dicom, pos in zip(subject_id, study_id, dicom_id, ViewPosition):
        if pos == 'AP' and data[sub][0] == stu:
            data[sub].append(dicom)
            if len(data[sub]) > 2:
                tmp = sorted(data[sub][1:])
                data[sub] = [data[sub][0], tmp[0]]

    return data



def make_label(negbio):
    # get label (string)
    labels = negbio.iloc[:, 1:].to_numpy()

    # convert nan, -1 to 0
    labels = labels.astype(np.float32)
    labels = np.where(labels==-1, 0, labels)
    labels = np.where(np.isnan(labels), 0, labels)
    labels = labels.tolist()

    # make label dict
    labels = {stu: label for stu, label in zip(negbio['study_id'].tolist(), labels)}

    return labels



def merge_data(img_dict, label_dict):
    dataset = {k: [v[0], v[1], label_dict[v[0]]] for k, v in img_dict.items()}
    return dataset



def pwd_processing(pwd):
    new_pwd = ''
    for c in pwd:
        if c in string.punctuation:
            new_pwd += '\\' + c
        else:
            new_pwd += c
    return new_pwd



def download_img(base_path, dataset, physio_id, pwd):
    assert os.path.isdir(base_path + '/data')
    pwd = pwd_processing(pwd)

    img_folder_dict = {'p10': 'p10/{', 'p11': 'p11/{', 'p12': 'p12/{', 'p13': 'p13/{', 'p14': 'p14/{',
                        'p15': 'p15/{', 'p16': 'p16/{', 'p17': 'p17/{', 'p18': 'p18/{', 'p19': 'p19/{'}
    base_exec = 'wget -r -N -c -np --user ' + physio_id + ' --password=' + pwd + ' https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/'

    for sub in dataset.keys():
        stu, img = dataset[sub][:2]
        img_folder_dict['p'+sub[:2]] += 'p' + sub + '/s' + stu + '/' + img + '.jpg,'
    
    cmd = ''
    for k in img_folder_dict.keys():
        cmd += base_exec + img_folder_dict[k][:-1] + '}'
        cmd += ' && '
    
    with open(base_path + '/cmd.txt', 'w') as f:
        f.write(cmd[:-4])
       


def split_data(base_path, dataset):
    os.makedirs(base_path+'/data/train', exist_ok=True)
    os.makedirs(base_path+'/data/test', exist_ok=True)

    for sub in tqdm(dataset.keys(), desc='splitting...'):
        digit = sub[:2]
        stu, img = dataset[sub][:2]
        open_path = base_path + '/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p' + digit + '/' + 'p' + sub + '/s' + stu + '/' + img + '.jpg'
        save_path = base_path + '/data/test/' + img + '.jpg' if stu[-1] in ['8', '9'] else base_path + '/data/train/' + img + '.jpg'
        shutil.move(open_path, save_path)



def get_coef(dataset):
    total_num, total_one = 0, []
    for sub in dataset.keys():
        stu, _, label = dataset[sub]
        if not stu[-1] in ['8', '9']:
            total_num += 1
            total_one.append(label)
    total_one = np.sum(np.array(total_one), axis=0)
    total_zero = total_num - total_one
    return torch.tensor(total_zero / total_one)