import numpy as np


def DetectDirect(A, type, k, T):
    if type == 1:
        # 45 degree diagonal direction
        t1 = abs(A[2,0]-A[0,2])
        t2 = abs(A[4,0]-A[2,2])+abs(A[2,2]-A[0,4])
        t3 = abs(A[6,0]-A[4,2])+abs(A[4,2]-A[2,4])+abs(A[2,4]-A[0,6])
        t4 = abs(A[6,2]-A[4,4])+abs(A[4,4]-A[2,6])
        t5 = abs(A[6,4]-A[4,6])
        d1 = t1+t2+t3+t4+t5

        # 135 degree diagonal direction
        t1 = abs(A[0,4]-A[2,6])
        t2 = abs(A[0,2]-A[2,4])+abs(A[2,4]-A[4,6])
        t3 = abs(A[0,0]-A[2,2])+abs(A[2,2]-A[4,4])+abs(A[4,4]-A[6,6])
        t4 = abs(A[2,0]-A[4,2])+abs(A[4,2]-A[6,4])
        t5 = abs(A[4,0]-A[6,2])
        d2 = t1+t2+t3+t4+t5
    else:
        # horizontal direction
        t1 = abs(A[0,1]-A[0,3])+abs(A[2,1]-A[2,3])+abs(A[4,1]-A[4,3])
        t2 = abs(A[1,0]-A[1,2])+abs(A[1,2]-A[1,4])
        t3 = abs(A[3,0]-A[3,2])+abs(A[3,2]-A[3,4])
        d1 = t1+t2+t3

        # vertical direction
        t1 = abs(A[1,0]-A[3,0])+abs(A[1,2]-A[3,2])+abs(A[1,4]-A[3,4])
        t2 = abs(A[0,1]-A[2,1])+abs(A[2,1]-A[4,1])
        t3 = abs(A[0,3]-A[2,3])+abs(A[2,3]-A[4,3])
        d2 = t1+t2+t3
    # Compute the weight vector
    w = np.array([1/(1+d1**k), 1/(1+d2**k)])
    # Compute the directional index
    n = 3
    if (1+d1)/(1+d2) > T:
        n = 1
    elif (1+d2)/(1+d1) > T:
        n = 2
    return w, n

def PixelValue(A, mode, w, n, f):
    if mode == 1:
        v1 = np.diag(np.fliplr(A))[::2]
        v2 = np.diag(A)[::2]
    else:
        v1 =  A[3,::2]
        v2 =  A[::2,3]
    if n == 1:
        p = np.dot(v2, f)
    elif n == 2:
        p = np.dot(v1, f)
    else:
        p1 = np.dot(v1, f)
        p2 = np.dot(v2, f)
        p = (w[0]*p1+w[1]*p2)/(w[0]+w[1])
    return p

def PadLeftTop(img_pad, H, W):
    img = img_pad[3:-3,3:-3]
    # Pad the first/last three col and row
    img_pad[3:H+3,1]=img[:,0]
    img_pad[H+3::2,3:W+3]=img[H-2:H-1,:]
    img_pad[3:H+3,W+3::2]=img[:,W-2:W-1]
    img_pad[1,3:W+3]=img[0,:]
    # Pad the missing nine points
    img_pad[1,1]=img[0,0]
    img_pad[H+3::2,1]=img[H-2,0]
    img_pad[H+3::2,W+3::2]=img[H-2,W-2]
    img_pad[1,W+3::2]=img[0,W-2]
    return img_pad

def PadRightBottom(img_pad, H, W):
    img = img_pad[3:-3,3:-3]
    # Pad the first/last three col and row
    img_pad[3:H+3,0:3:2]=img[:,1:2]
    img_pad[H+4::2,3:W+3]=img[H-1:H,:]
    img_pad[3:H+3,W+4::2]=img[:,W-1:W]
    img_pad[0:3:2,3:W+3]=img[1,:]
    # Pad the missing nine points
    img_pad[0:3:2,0:3:2]=img[1,1]
    img_pad[H+4,0:3:2]=img[H-1,1]
    img_pad[H+4,W+4]=img[H-1,W-1]
    img_pad[0:3:2,W+4]=img[0,W-1]
    return img_pad

def _DCC(I, k, T):
    m, n = I.shape
    nRow = 2*m
    nCol = 2*n
    A = np.zeros([nRow+6, nCol+6])
    A[0+3:-1-3:2, 0+3:-1-3:2] = I
    A = PadLeftTop(A, nRow, nCol)
    f = np.array([-1, 9, 9, -1])/16
    for i in range(4,nRow+3,2):
        for j in range(4,nCol+3,2):
            [w,n] = DetectDirect(A[i-3:i+4,j-3:j+4],1,k,T)
            A[i,j] = PixelValue(A[i-3:i+4,j-3:j+4],1,w,n,f)
    A = PadRightBottom(A, nRow, nCol)
    for i in range(3,nRow+3,2):
        for j in range(4,nCol+3,2):
            [w,n] = DetectDirect(A[i-2:i+3,j-2:j+3],2,k,T)
            A[i,j] = PixelValue(A[i-3:i+4,j-3:j+4],2,w,n,f)
    for i in range(4,nRow+3,2):
        for j in range(3,nCol+3,2):
            [w,n] = DetectDirect(A[i-2:i+3,j-2:j+3],3,k,T)
            A[i,j] = PixelValue(A[i-3:i+4,j-3:j+4],3,w,n,f)
    return A[3:-3,3:-3]


'''
img: Shape[H,W,C], Value Range[0-1]
level: super resolution level
Return: super resolution img who shape is the same with input
'''
def DCC(img, level):
    # hyper parameters
    k, T = 5, 1.15
    sr_img = img
    # get the high resolution image channel by channel
    for channel in range(img.shape[-1]):
        sr_img_simple = img[:,:,channel]
        for _ in range(level):
            sr_img_simple  = _DCC(sr_img_simple, k, T)
        sr_img[:,:,channel] = sr_img_simple
    return sr_img
