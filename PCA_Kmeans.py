import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from numpy import genfromtxt

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def save_gr_img(gray,name):
    gray_image = im.fromarray(gray)
    gray_image = gray_image.convert("L")
    gray_image.save(name+".jpg")
    return

def vid2Darray(name):
    cap = cv2.VideoCapture(name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3),dtype=int)
    buf_gray = np.empty((frameCount, frameHeight, frameWidth))
    buf_gf = np.empty((frameCount, frameHeight*frameWidth))
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    for i in np.arange(frameCount):
        buf_gray[i] = rgb2gray(buf[i])
        buf_gf[i] = buf_gray[i].flatten()

    cap.release()
    return buf,buf_gray, buf_gf

def vid2Darray2(name):
    cap = cv2.VideoCapture(name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameCount2 = frameCount//3+1
    buf1=[]
    
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        third1 = fc%3
        third2 = fc//3
        ret, array = cap.read()
        if third1 == 0:
            buf1.append(array)
        fc += 1
    buf = np.array(buf1,dtype=int)
    fC,fH,fW,rgb = buf.shape
    buf_gray = np.empty((fC,fH,fW))
    buf_gf = np.empty((fC,fH*fW))
    for i in np.arange(fC):
        buf_gray[i] = rgb2gray(buf[i])
        buf_gf[i] = buf_gray[i].flatten()

    cap.release()
    return buf,buf_gray, buf_gf


def res_vid(array,n):
    shape = array.shape
    f = shape[0]
    nx = int(shape[2]/n)
    ny = int(shape[1]/n)
    resvid = np.empty((f, ny, nx, 3),dtype=int)
    resvid_gray = np.empty((f, ny, nx))
    resvid_gf = np.empty((f, ny*nx))
    array1 = array.astype('float32')
    for i in np.arange(f):
        for j in np.arange(3):
            resvid[i,:,:,j] = cv2.resize(array1[i,:,:,j], dsize=(nx, ny), interpolation=cv2.INTER_CUBIC)
    
    for i in np.arange(f):
        resvid_gray[i] = rgb2gray(resvid[i])
        resvid_gf[i] = resvid_gray[i].flatten()
    return resvid, resvid_gray, resvid_gf

def vid2d_to_4d(vid2d,resvid_gray):
    f,nrow,ncol = resvid_gray.shape
    Output = np.empty((f, nrow, ncol,3))
    for i in np.arange(f):
        Output[i,:,:,0] = vid2d[i].reshape(nrow,ncol)
        Output[i,:,:,1] = vid2d[i].reshape(nrow,ncol)
        Output[i,:,:,2] = vid2d[i].reshape(nrow,ncol)
    Output = Output.astype(np.uint8)
    return Output
    
def save_vid(vid4d,name):
    f,nrow,ncol,l = vid4d.shape
    c = max(nrow,ncol)
    frameSize =(c,c)
    out = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'XVID'), 30, frameSize)

    for i in np.arange(f):
        img1 = np.zeros((c,c,3))#, dtype=np.uint8)
        img1[0:nrow,0:ncol] += vid4d[i]
        img1 = img1.astype(np.uint8)
        out.write(img1)
    out.release()
    cv2.destroyAllWindows()
    return

def k_means(data,k,b="random"):
    n = len(data)
    count = -1
    S_new = []
    initial = [[] for i in range(k)]
    if b == "random":
        b = np.random.choice(range(len(data)),k,replace=False)
        b = np.sort(b)
    
    for i in np.arange(k):
        if type(b[i])==np.ndarray:
                initial[i] = list(b[i])
        else:
            initial[i] = data[b[i]]
    
    while(True):
        S = [[] for i in range(k)]
        index = [[] for i in range(k)]
        vect = [[] for i in range(k)]
        dist_square = [0 for i in range(k)]
        tds = 0
        S_old = S_new
        count = count+1
    
        for i in np.arange(n):
        
            for j in np.arange(k):
                vect[j] = data[i]-initial[j]
                dist_square[j] = (np.dot(vect[j],vect[j]))
        
            min_value = min(dist_square)
            tds = tds + min_value
            min_index = dist_square.index(min_value)
            if type(data[i])==np.ndarray:
                a = list(data[i])
            else:
                a = data[i]
            S[min_index].append(a)
            index[min_index].append(i)
        for i in np.arange(k):
            initial[i] = np.sum(S[i],axis=0)/len(S[i])
    
        S_new = S
        if S_new==S_old:
            break
    print(count,"iterations")
    print(tds,"total squared distance")
    return S,index,tds

def PCA(A):
    M = np.mean(A,axis=0)
    X = A - M
    XtX = np.dot(X.transpose(),X)
    w,v = np.linalg.eigh(XtX)
    eVal = np.flipud(w)
    eVec = np.fliplr(v)
    return eVal,eVec,M

def var(eVal,n):
    totvar = np.sum(eVal)
    var = eVal[n-1]/totvar*100
    return var

def pcd(A,eVec,n):
    M = np.mean(A,axis=0)
    Mm = np.zeros(A.shape)+M
    if n == 0:
        return Mm
    X = A - M
    t = np.dot(X,eVec[:,n-1])
    pcd = t.reshape(-1,1)*eVec[:,n-1]
    return pcd

def count(datatag,index,n):
    array = np.zeros(n)
    for i in np.arange(n):
        for j in np.arange(len(index)):
            if datatag[index[j]]==i:
                array[i] += 1
    return array

def confusion(answer,index):
    n = len(index)
    matrix = np.zeros((n,n))
    for i in np.arange(n):
        matrix[i,:] = count(answer,index[i],n)
    return matrix

def var_of_n(n,eVal):
    var = np.sum(eVal[0:n])/np.sum(eVal)*100
    return var

def comp_to_img(n,M,eVec,size):
    if n==0:
        comp = M.reshape(size)
    else:
        comp = eVec[:,n-1].reshape(size)
    return comp

def load(n,M,eVec,rvid_gf):   
    if n == 0:
        t = np.ones(rvid_gf.shape[0])
    else:
        v = eVec[:,n-1]
        t = np.dot(rvid_gf-M,v)
    f = np.arange(rvid_gf.shape[0])+1
    return t,f

def comp_load(n,M,eVec,rvid_gf,rvid_gray,save=0):
    n_frame = rvid_gray.shape[0]
    size = rvid_gray.shape[1:3]
    cmp = comp_to_img(n,M,eVec,size)
    t,f = load(n,M,eVec,rvid_gf)
    
    fig = plt.figure(figsize=(10, 7), dpi=150)
    plt.subplot(121)
    plt.imshow(cmp)
    plt.colorbar()
    plt.subplot(122)
    plt.plot(f,t)
    if save == 1:
        plt.savefig("component"+str(n)+".jpg")
    return

def PCA_data(n,M,eVec,rvid_gf):
    f = rvid_gf.shape[0]
    PCA_video = np.zeros((f,n))
    
    for i in np.arange(n):
        PCA_video[:,i] = load(i+1,M,eVec,rvid_gf)[0]
    return PCA_video

def total_distance(data,save=0):
    x = np.arange(10)
    y = np.zeros(10)
    
    for i in x:
        S,index,y[i] = k_means(data,i+1)

    plt.plot(x,y)
    if save == 1:
        plt.savefig("tds.jpg")
    return

def k_graph(index,k,save=0):
    for i in np.arange(k):
        plt.plot(index[i],np.ones(len(index[i]))*(i+1),".")
    if save == 1:
        plt.savefig("k_graph.jpg")
    return

def kmean_image(index,rvid_gf,rvid_gray,save=0):
    f,n = rvid_gf.shape
    f,row,col = rvid_gray.shape
    M = np.zeros(n)
    
    for i in index:
        M += rvid_gf[i]
    M = M/len(index)
    plt.imshow(M.reshape(row,col))
    plt.colorbar()
    if save == 1:
        plt.savefig("mean_image.jpg")
    return M

def comp_vid(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"sample_vid.avi")
    return

def comp_vid1(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA1.avi")
    return

def comp_vid2(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA2.avi")
    return

def comp_vid3(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA3.avi")
    return

def comp_vid4(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA4.avi")
    return

def comp_vid5(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA5.avi")
    return

def comp_vid6(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA6.avi")
    return

def comp_vid01(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA01.avi")
    return

def comp_vid02(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA02.avi")
    return

def comp_vid03(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA03.avi")
    return

def comp_vid04(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA04.avi")
    return

def comp_vid05(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA05.avi")
    return

def comp_vid06(comp_n,eVec,rvid_gf,rvid_gray):
    PCD = np.zeros((rvid_gf.shape))
    for i in comp_n:
        PCD +=pcd(rvid_gf,eVec,i)
    sample_vid = vid2d_to_4d(PCD,rvid_gray)
    save_vid(sample_vid,"0209ReSe2_PCA06.avi")
    return

def center_image(Array):
    X = np.absolute(Array)
    return np.amax(X)