import copy
import json
import math
import os
import cv2
import base64
import os
from tqdm import tqdm
from multiprocessing import Pool, Lock
import random
from PIL import Image
from io import BytesIO
from moviepy.editor import VideoFileClip


import itertools
def get_file_names_v2(folder_path):
    train = []
    val = []
    test = []
    for file_name in os.listdir(os.path.join(folder_path, 'train')):
        if file_name == 'train_total':
            continue
        for file_name_2 in os.listdir(os.path.join(folder_path, 'train', file_name)):
            train.append(file_name_2)
    for file_name in os.listdir(os.path.join(folder_path, 'val')):
        for file_name_2 in os.listdir(os.path.join(folder_path, 'val', file_name)):
            val.append(file_name_2)
    for file_name in os.listdir(os.path.join(folder_path, 'test')):
        for file_name_2 in os.listdir(os.path.join(folder_path, 'test', file_name)):
            test.append(file_name_2)
    return train, val, test, train + val + test

def calculate_frame_count_C(start_time_str, end_time_str, frame_rate=30):
    start_minute = int(start_time_str[:-2])
    start_second = int(start_time_str[-2:])
    end_minute = int(end_time_str[:-2])
    end_second = int(end_time_str[-2:])

    start_time = start_minute * 60 + start_second
    end_time = end_minute * 60 + end_second
    video_duration = end_time - start_time
    frame_count = video_duration * frame_rate
    return int(frame_count)


def calculate_frame_count(filename):
    start_time_str = filename.split("_")[3]
    end_time_str = filename.split("_")[4].split(".")[0]

    start_time = int(start_time_str)
    end_time = int(end_time_str)

    time_difference = end_time - start_time
    return time_difference


def video2images(video_id,frames,frame_num = 8):
    H_8_ls = [0.0, 0.3103, 0.4506, 0.5, 0.6747, 0.8498, 0.9403, 0.9999]
    H_16_ls = [0.0, 0.0739, 0.1479, 0.2219, 0.2959, 0.3103, 0.3804, 0.4506, 0.5, 0.5873, 0.6747, 0.7621, 0.8498, 0.8702, 0.9403, 0.9999]
    C_8_ls = [0.0, 0.1304, 0.2439, 0.4993, 0.7546, 0.7886, 0.936, 0.9999]
    C_16_ls = [0.0, 0.0388, 0.0777, 0.1165, 0.1304, 0.2041, 0.2439, 0.3716, 0.4993, 0.627, 0.7546, 0.7886, 0.8623, 0.936, 0.9611, 0.9999]
    M_8_ls = [0.0, 0.3215, 0.4276, 0.4700, 0.5577, 0.6895, 0.7901, 0.9999]
    M_16_ls = [0.0, 0.1062, 0.2125, 0.3187, 0.3215, 0.3745, 0.4276, 0.4700, 0.4936, 0.5577, 0.6218, 0.6895, 0.7371, 0.7901, 0.8937, 0.9999]
    H_128_ls = [0.0, 0.0023, 0.0047, 0.0071, 0.0095, 0.0118, 0.0142, 0.0166, 0.019, 0.0214, 0.0237, 0.0261, 0.0285, 0.0309, 0.0332, 0.0356, 0.038, 0.0404, 0.0428, 0.0451, 0.0475, 0.0499, 0.0504, 0.0595, 0.0687, 0.0779, 0.0871, 0.0963, 0.1054, 0.1146, 0.1238, 0.133, 0.1422, 0.1513, 0.1605, 0.1697, 0.1789, 0.1881, 0.1973, 0.2064, 0.2156, 0.2248, 0.234, 0.2432, 0.2523, 0.2615, 0.2707, 0.2799, 0.2891, 0.2982, 0.3074, 0.5, 0.5, 0.5109, 0.5109, 0.5218, 0.5218, 0.5327, 0.5327, 0.5436, 0.5436, 0.5546, 0.5546, 0.5655, 0.5655, 0.5764, 0.5764, 0.5873, 0.5873, 0.5982, 0.5982, 0.6092, 0.6092, 0.6201, 0.6201, 0.631, 0.631, 0.6419, 0.6419, 0.6529, 0.6529, 0.6638, 0.6638, 0.6747, 0.6747, 0.6856, 0.6856, 0.6965, 0.6965, 0.7075, 0.7075, 0.7184, 0.7184, 0.7293, 0.7293, 0.7402, 0.7402, 0.7512, 0.7512, 0.7621, 0.7621, 0.773, 0.773, 0.7839, 0.7839, 0.7948, 0.7948, 0.8058, 0.8058, 0.8167, 0.8167, 0.8276, 0.8276, 0.8385, 0.8385, 0.9467, 0.9559, 0.9651, 0.9762, 0.9785, 0.9809, 0.9833, 0.9857, 0.9881, 0.9904, 0.9928, 0.9952, 0.9976]
    C_128_ls = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.0321, 0.0341, 0.0361, 0.0381, 0.0401, 0.0421, 0.0441, 0.0461, 0.0481, 0.0501, 0.0521, 0.0541, 0.0561, 0.0581, 0.0601, 0.0621, 0.0642, 0.0682, 0.0723, 0.0763, 0.0804, 0.0845, 0.0885, 0.0926, 0.0967, 0.1007, 0.1048, 0.1089, 0.1129, 0.117, 0.1211, 0.1251, 0.1292, 0.2439, 0.2439, 0.2598, 0.2598, 0.2758, 0.2758, 0.2917, 0.2917, 0.3077, 0.3077, 0.3237, 0.3237, 0.3396, 0.3396, 0.3556, 0.3556, 0.3716, 0.3716, 0.3875, 0.3875, 0.4035, 0.4035, 0.4194, 0.4194, 0.4354, 0.4354, 0.4514, 0.4514, 0.4673, 0.4673, 0.4833, 0.4833, 0.4993, 0.4993, 0.5152, 0.5152, 0.5312, 0.5312, 0.5471, 0.5471, 0.5631, 0.5631, 0.5791, 0.5791, 0.595, 0.595, 0.611, 0.611, 0.627, 0.627, 0.6429, 0.6429, 0.6589, 0.6589, 0.6748, 0.6748, 0.6908, 0.6908, 0.7068, 0.7068, 0.7227, 0.7227, 0.7387, 0.7387, 0.939, 0.943, 0.9471, 0.9512, 0.9552, 0.9593, 0.9634, 0.9674, 0.9715, 0.9756, 0.9796, 0.9837, 0.9878, 0.9918, 0.9959]
    M_128_ls = [0.0, 0.0093, 0.0186, 0.0279, 0.0373, 0.0466, 0.0559, 0.0652, 0.0746, 0.0839, 0.0932, 0.1026, 0.1119, 0.1212, 0.1305, 0.1399, 0.1492, 0.1585, 0.1679, 0.1772, 0.1865, 0.1935, 0.2007, 0.208, 0.2153, 0.2226, 0.2298, 0.2371, 0.2444, 0.2517, 0.2589, 0.2662, 0.2735, 0.2808, 0.288, 0.2953, 0.3026, 0.3099, 0.3171, 0.4277, 0.4277, 0.4375, 0.4375, 0.4455, 0.4455, 0.4535, 0.4535, 0.4615, 0.4615, 0.4695, 0.4695, 0.4775, 0.4775, 0.4856, 0.4856, 0.4936, 0.4936, 0.5016, 0.5016, 0.5096, 0.5096, 0.5176, 0.5176, 0.5256, 0.5256, 0.5337, 0.5337, 0.5417, 0.5417, 0.5497, 0.5497, 0.5577, 0.5577, 0.5657, 0.5657, 0.5737, 0.5737, 0.5817, 0.5817, 0.5898, 0.5898, 0.5978, 0.5978, 0.6058, 0.6058, 0.6138, 0.6138, 0.6218, 0.6218, 0.6298, 0.6298, 0.6379, 0.6379, 0.6459, 0.6459, 0.6539, 0.6539, 0.6619, 0.6619, 0.6699, 0.6699, 0.6779, 0.6779, 0.7931, 0.8004, 0.8077, 0.8149, 0.8222, 0.8295, 0.8368, 0.844, 0.8513, 0.8586, 0.8659, 0.8731, 0.8804, 0.8877, 0.8973, 0.9067, 0.916, 0.9253, 0.9347, 0.944, 0.9533, 0.9626, 0.972, 0.9813, 0.9906]
    # frames = frames-2 有修改 frames - 2 ----> frames - 1
    frames = frames-1
    if frame_num == 8:
        if video_id.startswith('H'):
            return [math.floor(frames * x) for x in H_8_ls]
        elif video_id.startswith('C'):
            return [math.floor(frames * x) for x in C_8_ls]
        elif video_id.startswith('M'):
            return [math.floor(frames * x) for x in M_8_ls]
    elif frame_num == 16:
        if video_id.startswith('H'):
            return [math.floor(frames * x) for x in H_16_ls]
        elif video_id.startswith('C'):
            return [math.floor(frames * x) for x in C_16_ls]
        elif video_id.startswith('M'):
            return [math.floor(frames * x) for x in M_16_ls]
    elif frame_num == 128:
        if video_id.startswith('H'):
            return [int(round(pos * frames)) for pos in H_128_ls]
        elif video_id.startswith('C'):
            return [int(round(pos * frames)) for pos in C_128_ls]
        elif video_id.startswith('M'):
            return [int(round(pos * frames))  for pos in M_128_ls]


def get_relate_type_dic(anno):
    relate_video_dic = {'H_A': [], 'H_H': [], 'H_T': [], 'C_K': [], 'M_Z': [], 'M_C': []}
    for i in anno:
        if i['visual_input'][0:3] == 'H_A':
            relate_video_dic['H_A'].append(i['ID'])
        elif i['visual_input'][0:3] == 'H_H':
            relate_video_dic['H_H'].append(i['ID'])
        elif i['visual_input'][0:3] == 'H_T':
            relate_video_dic['H_T'].append(i['ID'])
        elif i['visual_input'][0:3] == 'C_K':
            relate_video_dic['C_K'].append(i['ID'])
        elif i['visual_input'][0:3] == 'M_Z':
            relate_video_dic['M_Z'].append(i['ID'])
        elif i['visual_input'][0:3] == 'M_C':
            relate_video_dic['M_C'].append(i['ID'])
    return relate_video_dic


def get_relate_task_dic(anno):
    relate_task_dic = {'H2': [], 'H3': [], 'H4': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'M2': [], 'M3': []}
    for i in anno:
        if i['task'] == 'H2':
            relate_task_dic['H2'].append(i['ID'])
        elif i['task'] == 'H3':
            relate_task_dic['H3'].append(i['ID'])
        elif i['task'] == 'H4':
            relate_task_dic['H4'].append(i['ID'])
        elif i['task'] == 'C2':
            relate_task_dic['C2'].append(i['ID'])
        elif i['task'] == 'C3':
            relate_task_dic['C3'].append(i['ID'])
        elif i['task'] == 'C4':
            relate_task_dic['C4'].append(i['ID'])
        elif i['task'] == 'C5':
            relate_task_dic['C5'].append(i['ID'])
        elif i['task'] == 'M2':
            relate_task_dic['M2'].append(i['ID'])
        elif i['task'] == 'M3':
            relate_task_dic['M3'].append(i['ID'])
    return relate_task_dic


def get_relate_video_dic(anno):
    relate_video_dic = {}
    for i in anno:
        relate_video_dic[i['visual_input']] = []
    for i in anno:
        relate_video_dic[i['visual_input']].append(i['ID'])
    return relate_video_dic


def get_relate_dic(anno):
    relate_dic = {}
    relate_video_dic = get_relate_video_dic(anno)
    relate_task_dic = get_relate_task_dic(anno)
    relate_type_dic = get_relate_type_dic(anno)
    for i in anno:
        relate_dic[i['ID']] = []

    for i in anno:
        # 同视频其他task任选4个
        this_video = i['visual_input']
        this_video_ids = random.sample(relate_video_dic[this_video][5:], 1) ## 排除H1的部分进行筛选 H1 固定占用 0-1-2-3-4 位置
        # 同类的task任选4个
        this_task = i['task']
        if this_task == 'H1' or this_task == 'C1' or this_task == 'M1':
            continue ## 当前的所有任务不考虑H1
        
        this_task_ids = random.sample(relate_task_dic[this_task], 1)
        # 同类的视频任选4个
        this_type = i['visual_input'][0:3]
        this_type_ids = random.sample(relate_type_dic[this_type], 1)
        relate_dic[i['ID']] = this_video_ids + this_task_ids + this_type_ids
    return relate_dic


def get_video_frames_dic(dir):
    video_frames_dic = {}
    for filename in os.listdir(dir):
        for filename_2 in tqdm(os.listdir(os.path.join(dir, filename))):
            cap = cv2.VideoCapture(os.path.join(dir, filename, filename_2))

            # 检查视频是否成功打开
            if not cap.isOpened():
                print("无法打开视频文件")
                return

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frames_dic[filename_2] = frame_count
            # 关闭视频文件
            cap.release()
    return video_frames_dic


def get_image_ids_dic(total, all_video_path, num_frames=128):
    def get_lens(path):
        clip = VideoFileClip(path)
        # 计算视频的时长，单位为s
        len1 = int(clip.reader.duration)
        clip.close()
        return len1
    image_ids_dic = {}
    for i in tqdm(total):
        if i not in image_ids_dic:
            video_frames_dic = get_lens(all_video_path + i) ## 每秒读取一帧，即获取视频长度
            pre_ls =  video2images(i,video_frames_dic,num_frames)

            frame_numbers = [max(min(frame, video_frames_dic - 1), 0) for frame in pre_ls]

            frame_numbers.sort()
            for j in range(1, len(frame_numbers)):
                if frame_numbers[j] < frame_numbers[j-1]:
                    frame_numbers[j] = min(frame_numbers[j-1] + 1, video_frames_dic - 1)
            image_ids_dic[i] = frame_numbers

    return image_ids_dic


# def get_lens(path):
#     clip = VideoFileClip(path)
#     # 计算视频的时长，单位为s
#     len1 = int(clip.reader.duration)
#     clip.close()
#     return len1

# def process_video(i, num_frames=128):
#     video_frames_dic = get_lens('/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/all_videos/' + i) ## 每秒读取一帧，即获取视频长度
#     pre_ls = video2images(i,video_frames_dic,num_frames)

#     frame_numbers = [max(min(frame, video_frames_dic - 1), 0) for frame in pre_ls]

#     frame_numbers.sort()
#     for j in range(1, len(frame_numbers)):
#         if frame_numbers[j] < frame_numbers[j-1]:
#             frame_numbers[j] = min(frame_numbers[j-1] + 1, video_frames_dic - 1)
#     return i, frame_numbers

# def get_image_ids_dic(total,num_frames=128):
#     image_ids_dic = {}

#     with Pool(processes=64) as pool: # 利用计算机的4个核心
#         for i, frame_numbers in tqdm(pool.imap_unordered(process_video, total), total=len(total)):
#             image_ids_dic[i] = frame_numbers
#     return image_ids_dic

def get_video_group(train):
    with open(r'F:\Projects\VQA_data\FunQA\otter_json\video_group.json', 'r') as f: ## 子任务与视频归类
        video_group_dic = json.load(f)
    video_in_group_dict = {}
    for key, values in video_in_group_dict.items():
        for value in values:
            video_in_group_dict[value] = key
    return video_group_dic,video_in_group_dict
def get_QA_dic(anno):
    instruction_answer_id_dic = {}
    for i in anno:
        instruction_answer_id_dic[i['visual_input']] = []
    for i in anno:
        instruction_answer_id_dic[i['visual_input']].append([i['instruction'], i['output'], i['ID'], i['task']])
        # instruction_answer_id_dic[i['visual_input']].append([i['instruction'], i['output'], i['task']])
    return instruction_answer_id_dic


import base64
import multiprocessing

def get_img_json(dir, videos, des, image_ids_dic):
    print(str(os.getpid()) + "Start")
    out_dic = {}
    err_id = []
    with open(des, 'w', encoding='utf8') as f:
        f.write('{\n')
        idx = 0
        for filename in tqdm(videos):
            # 打开视频文件
            video = cv2.VideoCapture(os.path.join(dir, filename))
            frame_ls = image_ids_dic[filename]
            # 逐帧读取视频
            frame_id = 0
            output_num = 0
            while video.isOpened():
                # 读取一帧图像
                ret, frame = video.read()
                if not ret:
                    break
                if frame_id in frame_ls:
                    # Resize the image to 224x224
                    frame = cv2.resize(frame, (224, 224))

                    # 将图像转换为 Base64 格式
                    _, buffer = cv2.imencode('.jpg', frame)
                    base64_image = base64.b64encode(buffer)
                    # 将 Base64 编码转换为字符串
                    id = 'FunQA_' + filename.split('.mp4')[0] + '_f_' + str(frame_id)
                    if idx == 0:
                        json_string = f'  "{id}": "{base64_image.decode()}"'
                        f.write(json_string)
                        output_num += 1
                        idx = 1
                    else:
                        json_string = f',\n  "{id}": "{base64_image.decode()}"'
                        f.write(json_string)
                        output_num += 1
                        idx = 1
                frame_id += 1
            # 释放视频对象和资源
            # if output_num != 128:
            #     print(filename)
            #     err_id.append(filename)
            video.release()

        f.write('\n}')
    print(err_id)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_video_path', type=str, default='/data/chengshuang/Otter/data/train')
    parser.add_argument('--annotation_json_path', type=str, default='/data/chengshuang/Otter/data/annotation_with_ID/')
    parser.add_argument('--output_name', type=str, default='FunQAf128_instructions_train.json')
    parser.add_argument('--all_video_path', type=str, default='/data/chengshuang/Otter/data/all_videos/')
    args = parser.parse_args()
    train_annotation_json_path = os.path.join(args.annotation_json_path, 'funqa_train.json')
    with open(train_annotation_json_path) as f:
        train = json.load(f)
    # with open(r'/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/annotation_with_ID/funqa_test.json') as f:
    #     val = json.load(f)
    # with open(r'/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/annotation_with_ID/funqa_test.json') as f:
    #     test = json.load(f)
    # total = train + val + test
    total = train
    total1 = copy.copy(total)
    # total = sorted(list(set([i['visual_input'] for i in total])))
    total = sorted([i['visual_input'] for i in total])
    anno = total1
    
    # total = total [:200]
    # anno = total1 [:200] ## 小数据集
    all_video_path = args.all_video_path
    image_ids_dic = get_image_ids_dic(total,all_video_path,128)
    instruction_answer_id_task_dic = get_QA_dic(anno)
    relate_dic = get_relate_dic(anno)
    
    # video_group_dic, video_in_group_dic = get_video_group(train)

    # base_path = r'/home/hnu2/.ss/Otter/dawan'
    # train_total_dir = os.path.join(base_path, r'FunQA_v1_total\train_total')
    # json_dir = os.path.join(base_path, r'otter_json\FunQAf128')

    # processes = []
    # for i in range(20):
    #     id_str = f'FunQAf128_{i:02d}'
    #     json_file = os.path.join(json_dir, f'{id_str}.json')
    #     process = multiprocessing.Process(target=get_img_json,
    #                                       args=(train_total_dir, video_group_dic[id_str], json_file, image_ids_dic))
    #     processes.append(process)
    #
    # # Start all processes
    # for process in processes:
    #     process.start()
    #
    # # Wait for all processes to finish
    # for process in processes:
    #     process.join()
#%%

# video_in_group_dic['H_H_260_3450_3595.mp4']

    #%%

    # 创建空的 JSON 数据结构
    json_data = {
        "meta": {
            "version": "1.0",
            "time": "2023-8-22 12:00:00",
            "author": "FunQA"
        },
        "data": {}
    }



    def get_file_names_v2(folder_path):
        train = []
        for file_name in os.listdir(folder_path):
            train.append(file_name)
        return train    
    create_path = os.path.join(args.train_video_path, 'train_creative')
    humor_path = os.path.join(args.train_video_path, 'train_humor')
    magic_path = os.path.join(args.train_video_path, 'train_magic')
    
    train_creative = get_file_names_v2(create_path)
    train_humor = get_file_names_v2(humor_path)
    train_magic = get_file_names_v2(magic_path)
    train = train_creative + train_humor + train_magic
        
    
    # train = ['H_A_1_0500_0635.mp4', 'H_A_1_1796_2102.mp4', 'H_A_1_3310_3422.mp4'] ## 当前只针对train_humor来构建数据集
    # image_ids_dic = {'H_A_1_0500_0635.mp4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63], 'H_A_1_1796_2102.mp4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33], 'H_A_1_3310_3422.mp4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]}


    train = sorted(train) ## code 中 所有的sorted 只是为了调试方便
    for filename in tqdm(train):
        # 提供 instruction、answer、image_ids 和 relate 的值
        instruction_answer_id_task_ls = instruction_answer_id_task_dic[filename]

        image_ids_list = image_ids_dic[filename]
        image_ids_list_out = []
        # out_name = video_in_group_dic[filename]
        # 对应的图片id
        for i in range(len(image_ids_list)):
            formatted_string = "{:0>4}".format(str(image_ids_list[i]))
            image_ids_list_out.append('FunQA_' + "IMG_" + filename.split('.mp4')[0] + '_' + formatted_string) ## 128帧图像的抽取
        # 相关的QAid 需要有f-
        for i in instruction_answer_id_task_ls:
            # question_id = out_name + "_QA_" + i[2]
            question_id = 'Fun' + "QA_" + i[2]
            relate_list = relate_dic[i[2]]
            relate_list_out = []

            for j in range(len(relate_list)):
                # relate_list_out.append(out_name+ '_QA_' + relate_list[j])
                relate_list_out.append('Fun'+ 'QA_' + relate_list[j])
            # 生成数据字典
            data_dict = {
                "instruction": i[0],
                "answer": i[1],
                "image_ids": image_ids_list_out,
                "rel_ins_ids": relate_list_out
            }


            # 将数据添加到 JSON 数据结构中
            json_data["data"][question_id] = data_dict

    # 将 JSON 数据写入文件
    output_path = os.path.join('./output', args.output_name)
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)