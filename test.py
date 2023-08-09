import os
import shutil


def clear_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            shutil.rmtree(folder_path)


# 调用函数清除指定文件夹内的所有文件夹
folder_to_clear = "runs"
clear_folders(folder_to_clear)


#%tensorboard --logdir C:\Users\hrdaweed\Desktop\RL-based-Transferable-EMS\RL-based-Transferable-EMS\runs