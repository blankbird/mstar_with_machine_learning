import numpy as np
import scipy.misc as im  # 使用 imresize imread 函数，已经不能使用，故修改如下
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

#替换im功能
from PIL import Image
import imageio

def get_mstar_data(stage, width=128, height=128, crop_size=128, aug=False):
    data_dir = "MSTAR-10/train/" if stage == "train" else "MSTAR-10/test/" if stage == "test" else None
    '''上句等价于：
    if stage == "train":
        data_dir = "MSTAR-10/train/"
    else:
        data_dir = "MSTAR-10/test/" if stage == "test" else None
    '''

    print("------ " + stage + " ------")
    sub_dir = ["2S1", "BMP2", "BRDM_2", "BTR60", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU_23_4"]
    X = []
    y = []

    for i in range(len(sub_dir)):
        tmp_dir = data_dir + sub_dir[i] + "/"
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpeg")]
        '''等价于
        img_idx = []
        for x in os.listdir(tmp_dir):
            if x.endswith(".jpeg"):
                img_idx.append(x)  # 添加图片到img.idx里
        '''

        print(sub_dir[i], len(img_idx))
        y += [i] * len(img_idx)  # 不太懂,标记的样本标志吧，但是式子不懂(懂了。。)
        for j in range(len(img_idx)):   # 遍历同一类里的图片
            #img = im.imresize(im.imread((tmp_dir + img_idx[j])), [height, width])
            ###上一句修改如下
            myimg = imageio.imread((tmp_dir + img_idx[j]))
            img=np.array(Image.fromarray(myimg).resize([height, width]))  # 把每个图片改成[height, width]大小
            ###
            img = img[(height - crop_size) // 2 : height - (height - crop_size) // 2, \
                  (width - crop_size) // 2: width - (width - crop_size) // 2] # img[32:96,32:96] [16:48,16:48]
            # img = img[16:112, 16:112]   # crop
            X.append(img)  # 处理的图片逐一加入 X 中

    return np.asarray(X), np.asarray(y)

def data_shuffle(X, y, seed=0):   #打乱顺序
    data = np.hstack([X, y[:, np.newaxis]])  # np.hstack():在水平方向上平铺 np.newaxis:插入新维度
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]

def one_hot(y_train, y_test):
    one_hot_trans = OneHotEncoder().fit(y_train[:, np.newaxis])
    return one_hot_trans.transform(y_train[:, np.newaxis]).toarray(), one_hot_trans.transform(y_test[:, np.newaxis]).toarray()

def mean_wise(X):
    return (X.T - np.mean(X, axis=1)).T  # 减去每个图片的均值

def pca(X_train, X_test, n):
    pca_trans = PCA(n_components=n).fit(X_train)
    return pca_trans.transform(X_train), pca_trans.transform(X_test)

