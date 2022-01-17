import numpy as np
from scipy import signal

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

f = np.load('eeg_raw_1114.npz')
x = f['x_train']
#x = torch.tensor(x)
y = np.genfromtxt('dataeeg/label_1_class.dat', delimiter=' ')
print(y)

y = np.array(y)
label = []
#y = torch.tensor(y)
for i in range(1280):
    x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
print(x[0].shape)
save_l = []
test_max_number =np.zeros(32)
for i in range(1280):
    save_ = []
    #for j in channel:
    for j in range(32):
        save_one_t = []
        batch_ = []
        batch_1 = []
        g = 0
        for t in range(7680):
            batch_.append(x[i][j][t])
            batch_1.append(x[i][j][t])
            if (t+1)%768 ==0 and t!=0:
                #print(len(batch_))
                g+=1
                freqs, psd = signal.welch(batch_, fs=1.0, window='hamming', nperseg=63, noverlap=None, nfft=None, detrend='constant',return_onesided=True, scaling='density', axis=- 1, average='mean')
                #print(np.clip(psd*300,a_min = 0,a_max = 1))
                batch_ = []
                save_one_t.append(psd)
                #print(psd * 3000)
        #draw_quantization_(batch_1)
        #break
        save_.append(save_one_t)
    #break
    save_l.append(save_)

    if i%64 ==0:
        print(str(i/12.8)+"%")

save_l = np.array(save_l)
print(save_l.shape)
print(test_max_number)
np.savez('eeg_32_60_32_1123_hamming_5s_arousal.npz',x_train = save_l)






x = np.linspace(0,31,32)
print(x)

