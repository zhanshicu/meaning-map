import numpy as np
from scipy import fftpack

def antonio_gaussian(img, fc):
    """
    Gaussian low pass filter (with circular boundary conditions).

    Parameters:
    img (numpy.ndarray): Input image
    fc (float): Cut off frequency (-6dB)

    Returns:
    tuple: (BF, gf)
        BF (numpy.ndarray): Output image
        gf (numpy.ndarray): Gaussian filter
    """
    sn, sm = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    n = max(sn, sm)
    n = n + (n % 2)
    n = 2 ** np.ceil(np.log2(n)).astype(int)

    img2 = zero_pad_image(img, n)
    sn2, sm2 = img2.shape[:2]
    n2 = max(sn2, sm2)

    fx, fy = np.meshgrid(np.arange(n) - n/2, np.arange(n) - n/2)

    s = round(n2/n) * fc / np.sqrt(np.log(2))

    gf = np.exp(-(fx**2 + fy**2) / (s**2))

    gf2 = np.zeros((n2, n2, c))
    gf2[n2//2-n//2:n2//2+n//2, n2//2-n//2:n2//2+n//2] = gf[:, :, np.newaxis]
    gf2 = fftpack.fftshift(gf2)

    BF = np.zeros((n2, n2, c))
    for i in range(c):
        img_channel = img2[:,:,i] if c > 1 else img2
        BF[:,:,i] = np.real(fftpack.ifft2(fftpack.fft2(img_channel) * gf2[:,:,i]))

    BF = BF[n:n+sn, n:n+sm]
    
    if c == 1:
        BF = BF[:,:,0]

    return BF, gf

def zero_pad_image(I, p):
    """
    Zero pad the input image.

    Parameters:
    I (numpy.ndarray): Input image
    p (int): Padding size

    Returns:
    numpy.ndarray: Padded image
    """
    h, w = I.shape[:2]
    c = I.shape[2] if I.ndim == 3 else 1
    
    Ipad = np.zeros((2*p, 2*p, c)) if c > 1 else np.zeros((2*p, 2*p))
    
    if c > 1:
        Ipad[p:p+h, p:p+w, :] = I
    else:
        Ipad[p:p+h, p:p+w] = I
    
    return Ipad