import numpy as np
from sklearn.model_selection import train_test_split
#a = np.random.random((2,4,2))
f = np.load('eeg_32_60_32_1123_hamming_5s_arousal.npz')
x = f['x_train']
print(x.shape)
y = np.genfromtxt('dataeeg/labels_0_class.dat', delimiter=' ')
y = np.array(y)
#print(a)
d = []
for item in x:
    a = item
    #print(a.shape)
    b= []
    i=0
    for item in a:
        b.append(item.T)
        i+=1
    b = np.array(b)
    b = b.T
    c = []
    for item in b:
        c.append(item.T)
    c= np.array(c)
    d.append(c)
    #print(c)
    #print(c.shape)
d= np.array(d)
#print(d.shape)
i =0
e = []
label = []
for item in d:
    for item_ in item:
        e.append(item_.reshape(1,32,32))
        label.append(y[i])
    i+=1
e = np.array(e)
label = np.array(label)

#np.savez('eeg_pwd_new.npz',x_train = e,y_train = label)
train_x = []
train_y = []
test_x = []
test_y = []
print(e.shape)
print(y.shape)
for k in range(32):
    x = []
    y = []
    x_ = []
    y_ = []
    i=j=0
    for item in label[k*400:(k+1)*400]:
        if item==0:
            i+=1
        else:
            j+=1
    min_ij = min(i,j)
    i = j = 0
    for len_t in range(400):
        if label[k*400+len_t] ==0 and i <min_ij:
            i+=1
            x.append(e[400*k+len_t])
            y.append(label[400*k+len_t])
        if label[k * 400 + len_t] == 1 and j< min_ij:
            j+=1
            x_.append(e[400 * k + len_t])
            y_.append(label[400 * k + len_t])

    X_train_, X_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.2, random_state=57)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(x_, y_, test_size=0.2, random_state=4)
    for item in X_train_:
        train_x.append(item)
    for item in y_train_:
        train_y.append(item)
    for item in X_test_:
        test_x.append(item)
    for item in y_test_:
        test_y.append(item)
    for item in X_train_1:
        train_x.append(item)
    for item in y_train_1:
        train_y.append(item)
    for item in X_test_1:
        test_x.append(item)
    for item in y_test_1:
        test_y.append(item)
    print(i,j)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
i = j = 0
for item in train_y:
    if item == 0:
        i += 1
    else:
        j += 1
print(i,j)
print(test_y)

np.savez('eeg1201_r57_b_hamming_5s_2_c_arousal.npz', X_train_=train_x, y_train_=train_y, X_test=test_x, y_test=test_y)


