# -*- coding: utf-8 -*-
"""
将数据集划分为训练集，验证集，测试集
"""

import splitfolders

# 1.确定原图像数据集路径
dataset_dir = "/DATA/lzw/data/10C/"  ##原始数据集路径
# 2.确定数据集划分后保存的路径
split_dir = "/DATA/lzw/data/10C_split/"  ##划分后保存路径



splitfolders.ratio(input=dataset_dir,
                   output=split_dir,
                   seed=1337, ratio=(0.7, 0.2, 0.1)) # 划分为训练集\验证集\测试集
