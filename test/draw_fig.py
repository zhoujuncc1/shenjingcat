import numpy as np
from matplotlib import pyplot
def draw_quantization():
    x = np.linspace(0,1,1000)
    T = 100
    x_q = np.floor(x*T)/T
    pyplot.plot(x_q)
    pyplot.show()

draw_quantization()