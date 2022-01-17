import pickle
import  numpy as np
from scipy import stats
nLabel, nTrial, nUser, nChannel, nTime = 4, 40, 32, 40, 8064
no_of_users = 32

def transform(x):
    x = x
    mean = np.mean(x, axis=0)
    median = np.median(x)
    max = np.max(x)
    min = np.min(x)
    sigma = np.std(x, axis=0)
    var = np.var(x)
    range = max-min
    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    rebulitData = [mean,median,max,min,sigma,var,range,skew,kurtosis]
    return rebulitData
#x = [1,3.5,5,7]
#y = []
#for i in transform(x):
#    y.append(i)
#for i in transform(x):
#    y.append(i)
#y.append(transform(x))
#y.append(transform(x))
#print(y)
#print(transform(x))
channel = [4,3,19,20,21,1,17,16,11,29,12,30,5,6,17,19,22,23,1,9,10,28,27,13,31]

channel_1 = []
for item in channel:
    channel_1.append(item-1)

channel1 = [3,1,17,20]
channel2 = [7,2,19,25]
data = []
print("Program started" + "\n")
fout_data = open("data/feature_0129_every32.dat", 'w')
fout_labels0 = open("data/feature_0129_every32_label_raw.dat", 'w')
#fout_labels1 = open("data/new_labels_1.dat", 'w')
#fout_labels2 = open("data/new_labels_2.dat", 'w')
#fout_labels3 = open("data/new_labels_3.dat", 'w')
for i in range(32):  # nUser #4, 40, 32, 40, 8064 4 labels, 40 sample for each user, 32 such user, 40 electrode, 8064*40 features
    if (i % 1 == 0):
        if i < 10:
            name = '%0*d' % (2, i + 1)
        else:
            name = i + 1
    fname = "data/s" + str(name) + ".dat"
    f = open(fname, 'rb')  # Read the file in Binary mode
    x = pickle.load(f, encoding='latin1')
    print(fname)
    #for tr in range(nTrial):
    for tr in range(nTrial):
        one_line = []
        if (tr % 1 == 0):
            for ch in range(32):
                save = []
                if ch !=90:
                    for dat in range(nTime):
                        if dat>383:
                            save.append(x['data'][tr][ch][dat])
                one_line.append(save)
        data.append(one_line)
array = np.array(data)
print(array.shape)

np.savez('eeg_raw_1114.npz',x_train = array)

