import os
import shutil
import argparse

def move_files_to_one_directory(directory_path, all_videos_dir):
    if not os.path.exists(all_videos_dir):
        os.makedirs(all_videos_dir)
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.mp4'):
            file_path = os.path.join(directory_path, file_name)
            print('cp', file_path, all_videos_dir)
            shutil.copy(file_path, all_videos_dir)


            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq')
    parser.add_argument('--all_videos_path', type=str, default='/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/all_videos')
    args = parser.parse_args()
    # 把train和test路径下的所有mp4文件移到all_videos文件夹下
    directory_path = ['train', 'test']
    kind_type = ['creative', 'humor', 'magic']
    for kind in kind_type:
        for directory in directory_path:
            raw_data_path = args.raw_data_path
            path = os.path.join(raw_data_path, f'{directory}/{directory}_{kind}')
            print(path)
            all_videos_dir = args.all_videos_path
            move_files_to_one_directory(path, all_videos_dir)