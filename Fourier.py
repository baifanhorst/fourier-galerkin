import numpy as np
import scipy.fft as spfft



def Interpolation(x, f):
    # f = np.array([f0, f1, ..., fN-1]), function values
    # x: in [0,2pi]
    N = len(f)
    
    if not (N%2 == 0):
        print('The length of the input must be even')
        return None
    
    Nhalf = N//2
    
    # FFT, David's version
    f_fft = spfft.fft(f) / N
    
    # Interpolation
    I = 0
    
    # k = 0, ..., N/2-1
    for k in range(Nhalf):
        I += f_fft[k] * np.exp(1j * k * x)
    
    # k = N/2, -N/2    
    # Note that ftilde_{N/2} = ftilde_{-N/2}
    k = Nhalf
    I += f_fft[k] * np.exp(1j * k * x) / 2
    I += f_fft[k] * np.exp(1j * (-k) * x) / 2
    
    # k = -N/2+1, ..., -1
    for k in range(-Nhalf+1, 0):
        I += f_fft[k+N] * np.exp(1j * k * x)
        
    return I
    
    
    
    
    