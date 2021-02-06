import numpy as np
import data
import model
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print("loading ... ")
X_train, y_train = data.get_mstar_data("train", 128, 128, 96)  # X_train未改格式:001.txt
X_test, y_test = data.get_mstar_data("test", 128, 128, 96)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # 后加
#plt.imshow(X_train[0],cmap=plt.cm.gray)
#plt.show()
X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1] * X_train.shape[2]])  # 修改格式后X_train:002.txt,其实把96*96的图片展成一行
X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1] * X_test.shape[2]])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


file3 = open(r'E:\python\Code\mstar_with_machine_learning\\' + '002.txt', 'w', encoding='UTF-8')
for i in range(len(X_train)):
    file3.write(str(X_train[i]) + '\n')

'''
print(X_train[:64,:64])
print(X_train[:64,:64].shape)  # 取行和列有问题
plt.imshow(X_train[:64,:64],cmap=plt.cm.gray)
plt.show()
'''

# print("shuffling ... ")
X_train, y_train = data.data_shuffle(X_train, y_train)
X_test, y_test = data.data_shuffle(X_test, y_test)

print("preprocessing ...")
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = data.mean_wise(X_train)
X_test = data.mean_wise(X_test)
X_train, X_test = data.pca(X_train, X_test, 80)
# y_train, y_test = data.one_hot(y_train, y_test)

print("training ...")
# classifier = model.train(X_train, y_train, model.dt("entropy", 0.8)) # 70.68%
# classifier = model.train(X_train, y_train, model.rf(1000, "sqrt")) # 96.49%
# classifier = model.train(X_train, y_train, model.gbdt(1000, "sqrt")) # 95.17%
# classifier = model.train(X_train, y_train, model.logit(1.0))    # 90.14%
# classifier = model.train(X_train, y_train, model.mlp(1000, "logistic"))   # 93.36%
# classifier = model.train(X_train, y_train, model.svm(1.0, "rbf"))     # 96.82%
# classifier = model.train(X_train, y_train, model.knn(10, "uniform"))   # 95.34%
classifier = model.train(X_train, y_train, model.bayes())   # 82.30%

print("testing ...")
print(model.acc(X_train, y_train, classifier))
print(model.acc(X_test, y_test, classifier))