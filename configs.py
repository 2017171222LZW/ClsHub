import numpy as np
import torch

# 普氏硬度
P_rigidity = {
    '花岗岩': [6, 7],
    '橄榄岩': [6.5, 7],
    '砾岩':   [0, 0], # 胶结而成的岩石
    '大理岩': [2.5, 5],
    '玄武岩': [5, 7],
    '泥岩':   [4, 6]
}
P_rigidity_en = {
    'huagan': [12., 14.],
    'ganlan': [6.5, 7.],
    'liyan':   [0., 0.], # 胶结而成的岩石
    'dali': [10., 12.],
    'xuanwu': [20., 25.],
    'niyan':   [4., 6.]
}
P_max_v = 25.

M_rigidity = {
    '花岗岩': [6., 7],
    '橄榄岩': [6.5, 7.],
    '砾岩':   [0., 0.], # 胶结而成的岩石
    '大理岩': [6.5, 7.5],
    '玄武岩': [5, 7],
    '泥岩':   [2, 5]
}
M_rigidity_en = {
    'huagan': [6, 7],
    'ganlan': [6.5, 7],
    'liyan':   [0, 0], # 胶结而成的岩石
    'dali': [6.5, 7.5],
    'xuanwu': [5, 7],
    'niyan':   [2, 5]
}
M_max_v = 7.5

def digsfunc(classes, target):
    classes = np.array(classes)
    digs = torch.zeros(target.shape[0], 1)
    target_names = classes[target.detach().int().cpu()]
    if target.shape[0] == 1:
        target_names = [target_names]
    for i in range(target.shape[0]):
        section = M_rigidity_en[target_names[i]]
        digs[i, 0] = (section[0] + torch.rand(1)*(section[1] - section[0])) / M_max_v
    return digs

if __name__ == '__main__':
    data = torch.tensor([3, 1, 2, 0, 5, 1, 2])
    classes = ['dali', 'ganlan', 'huagan', 'liyan', 'niyan', 'xuanwu']
    res = digsfunc(classes, data)
    print(res.shape)