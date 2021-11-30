import numpy as np, cv2
import matplotlib.pyplot as plt   
from multiprocessing import Pool

def DT_filter_IC(im, out, sigma_s, sigma_r, max_iter, gradient=False):
    dcx=np.zeros((im.shape[0], im.shape[1]-1))
    dcy=np.zeros((im.shape[0]-1, im.shape[1]))
    ratio=sigma_s/sigma_r
    if not gradient:
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]-im[:, ii]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]-im[ii, :]), 1)
    else: 
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]), 1)
            
    for iteration in range(max_iter): 
        sigma_H=sigma_s*(3**0.5)*(2**(max_iter-iteration-1))/((4**max_iter-1)**0.5)
        a=np.exp(-(2**0.5)/sigma_H)
        V=np.power(a, dcx)
        for ii in range(1, im.shape[1]): 
            out[:, ii]=out[:, ii]+V[:, ii-1, np.newaxis]*(out[:, ii-1]-out[:, ii])
        for ii in range(im.shape[1]-2, -1, -1): 
            out[:, ii]=out[:, ii]+V[:, ii, np.newaxis]*(out[:, ii+1]-out[:, ii])
        
        V=np.power(a, dcy)
        for ii in range(1, im.shape[0]): 
            out[ii, :]=out[ii, :]+V[ii-1, :, np.newaxis]*(out[ii-1, :]-out[ii, :])
        for ii in range(im.shape[0]-2, -1, -1): 
            out[ii, :]=out[ii, :]+V[ii, :, np.newaxis]*(out[ii+1, :]-out[ii, :])
    return out

def disp_centroid(probV, disp_range):
    di_range=np.arange(disp_range[0], disp_range[1]+1)
    di_range=di_range[np.newaxis, np.newaxis, :]
    expect=np.sum(probV*di_range, axis=2)/np.sum(probV, axis=2)
    return expect 

def norm_im(img): 
    nim=(img-np.min(img))/(np.max(img)-np.min(img))*255
    nim=cv2.applyColorMap(nim.astype('uint8'), cv2.COLORMAP_JET)
    return nim

def DT_filter_with_mask(im, out, trimap, mask, sigma_s, sigma_r, max_iter):
    dcx=np.zeros((im.shape[0], im.shape[1]-1))
    dcy=np.zeros((im.shape[0]-1, im.shape[1]))
    ratio=sigma_s/sigma_r
    
    for ii in range(im.shape[1]-1):
        dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]), 1)
    for ii in range(im.shape[0]-1): 
        dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]), 1)
    trimap_o=trimap.copy()    
    for iteration in range(max_iter): 
        sigma_H=sigma_s*(3**0.5)*(2**(max_iter-iteration-1))/((4**max_iter-1)**0.5)
        a=np.exp(-(2**0.5)/sigma_H)
        V=np.power(a, dcx)
        for ii in range(1, im.shape[1]): 
            sel=(~trimap[:, ii])*trimap[:, ii-1]*(mask[:, ii]>=mask[:, ii-1])*(V[:, ii-1]>0.2)
            out[:, ii]=out[:, ii]+(out[:, ii-1]-out[:, ii])*sel[..., np.newaxis]
            # out[:, ii]/=np.sum(out[:, ii])
            trimap[:, ii]|=sel
        # updated=np.sum(out, axis=2)>0.5
        # out[updated, :]/=np.sum(out, axis=2, keepdims=True)[updated, :]
        # out[~updated]=1e-9


        for ii in range(im.shape[1]-2, -1, -1): 
            sel=trimap[:, ii+1]*(mask[:, ii]>=mask[:, ii+1])
            out[:, ii]=out[:, ii]+V[:, ii, np.newaxis]*(out[:, ii+1]-out[:, ii])*sel[..., np.newaxis]
            trimap[:, ii]|=(sel*V[:, ii])>0.1
        V=np.power(a, dcy)
        for ii in range(1, im.shape[0]): 
            out[ii, :]=out[ii, :]+V[ii-1, :, np.newaxis]*(out[ii-1, :]-out[ii, :])
        for ii in range(im.shape[0]-2, -1, -1): 
            out[ii, :]=out[ii, :]+V[ii, :, np.newaxis]*(out[ii+1, :]-out[ii, :])
    return out


def DT_filter_with_conf2(im, out, conf, sigma_s, sigma_r, max_iter, gradient=False):
    dcx=np.zeros((im.shape[0], im.shape[1]-1))
    dcy=np.zeros((im.shape[0]-1, im.shape[1]))
    ratio=sigma_s/sigma_r
    if not gradient:
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]-im[:, ii]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]-im[ii, :]), 1)
    else: 
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]), 1)
            
    for iteration in range(max_iter): 
        sigma_H=sigma_s*(3**0.5)*(2**(max_iter-iteration-1))/((4**max_iter-1)**0.5)
        a=np.exp(-(2**0.5)/sigma_H)
        V=np.power(a, dcx)
        for ii in range(1, im.shape[1]): 
            confVec=conf[:, ii-1]/(conf[:, ii-1]+conf[:,ii]+1e-6)
            out[:, ii]=out[:, ii]+V[:, ii-1, np.newaxis]*(out[:, ii-1]-out[:, ii])*confVec[..., np.newaxis]
            conf[:, ii]=conf*confVec#conf[:, ii]+V[:, ii-1]*(conf[:, ii-1]-conf[:, ii])
        for ii in range(im.shape[1]-2, -1, -1): 
            confVec=conf[:, ii+1]/(conf[:, ii+1]+conf[:,ii]+1e-6)
            out[:, ii]=out[:, ii]+V[:, ii, np.newaxis]*(out[:, ii+1]-out[:, ii])*confVec[..., np.newaxis]
            conf[:, ii]=conf[:, ii]+V[:, ii]*(conf[:, ii+1]-conf[:, ii])
        
        V=np.power(a, dcy)
        for ii in range(1, im.shape[0]): 
            confVec=conf[ii-1, :]/(conf[ii-1, :]+conf[ii, :]+1e-6)
            out[ii, :]=out[ii, :]+V[ii-1, :, np.newaxis]*(out[ii-1, :]-out[ii, :])*confVec[..., np.newaxis]
            conf[ii, :]=conf[ii, :]+V[ii-1, :]*(conf[ii-1, :]-conf[ii, :])
        for ii in range(im.shape[0]-2, -1, -1): 
            confVec=conf[ii+1, :]/(conf[ii+1, :]+conf[ii, :]+1e-6)
            out[ii, :]=out[ii, :]+V[ii, :, np.newaxis]*(out[ii+1, :]-out[ii, :])
            conf[ii, :]=conf[ii, :]+V[ii, :]*(conf[ii+1, :]-conf[ii, :])
    return out, conf

def DT_filter_with_conf(im, out, conf, sigma_s, sigma_r, max_iter, gradient=False):
    dcx=np.zeros((im.shape[0], im.shape[1]-1))
    dcy=np.zeros((im.shape[0]-1, im.shape[1]))
    ratio=sigma_s/sigma_r
    if not gradient:
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]-im[:, ii]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]-im[ii, :]), 1)
    else: 
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]), 1)
            
    for iteration in range(max_iter): 
        sigma_H=sigma_s*(3**0.5)*(2**(max_iter-iteration-1))/((4**max_iter-1)**0.5)
        a=np.exp(-(2**0.5)/sigma_H)
        V=np.power(a, dcx)
        for ii in range(1, im.shape[1]): 
            confVec=conf[:, ii-1]*V[:, ii-1]
            confVec=confVec/(confVec+conf[:, ii])
            out[:, ii]=out[:, ii]+(out[:, ii-1]-out[:, ii])*confVec[..., np.newaxis]
            if ii == 230: 
                print('yes')
            conf[:, ii]=np.maximum(conf[:, ii]+(conf[:, ii-1]-conf[:, ii])*V[:, ii-1], conf[:, ii])
        for ii in range(im.shape[1]-2, -1, -1): 
            confVec=conf[:, ii+1]*V[:, ii]
            confVec=confVec/(confVec+conf[:, ii])
            out[:, ii]=out[:, ii]+(out[:, ii+1]-out[:, ii])*confVec[..., np.newaxis]
            conf[:, ii]=np.maximum(conf[:, ii]+V[:, ii]*(conf[:, ii+1]-conf[:, ii]), conf[:, ii])
        
        V=np.power(a, dcy)
        for ii in range(1, im.shape[0]): 
            confVec=conf[ii-1,:]*V[ii-1, :]
            confVec=confVec/(confVec+conf[ii, :])
            out[ii, :]=out[ii, :]+(out[ii-1, :]-out[ii, :])*confVec[..., np.newaxis]
            conf[ii, :]=np.maximum(conf[ii, :]+(conf[ii-1, :]-conf[ii, :])*V[ii-1, :], conf[ii, :])
        for ii in range(im.shape[0]-2, -1, -1): 
            confVec=conf[ii+1, :]*V[ii, :]
            confVec=confVec/(confVec+conf[ii, :])
            out[ii, :]=out[ii, :]+(out[ii+1, :]-out[ii, :])*confVec[..., np.newaxis]
            conf[ii, :]=np.maximum(conf[ii, :]+V[ii, :]*(conf[ii+1, :]-conf[ii, :]), conf[ii, :])
    return out, conf

def DT_filter_NC(im, out, sigma_s, sigma_r, max_iter, gradient=False):
    dcx=np.zeros((im.shape[0], im.shape[1]-1))
    dcy=np.zeros((im.shape[0]-1, im.shape[1]))
    ratio=sigma_s/sigma_r
    height, width, channel=im.shape
    if not gradient:
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]-im[:, ii]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]-im[ii, :]), 1)
    else: 
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]), 1)
    
    ctx=np.concatenate([np.zeros((height, 1)), np.cumsum(dcx, axis=1)], axis=1)
    cty=np.concatenate([np.zeros((1, width)),np.cumsum(dcy, axis=0)])

    # int_x, int_y=np.cumsum(out, axis=1), np.cumsum(out, axis=0)
    # int_x=np.concatenate([np.zeros((height, 1, channel)), int_x], axis=1)
    # int_y=np.concatenate([np.zeros((1, width, channel)), int_y], axis=0)
    for iteration in range(max_iter): 
        int_x =np.cumsum(out, axis=1)
        int_x=np.concatenate([np.zeros((height, 1, channel)), int_x], axis=1)
        r=sigma_s*3*(2**(max_iter-iteration-1))/((4**max_iter-1)**0.5)
        left, right=np.ones_like(ctx)*width, np.zeros_like(ctx)
        up, down=np.ones_like(cty)*height, np.zeros_like(cty)
        for ii in range(height):
            left_idx,right_idx=interval_search(ctx[ii, :], left[ii, :], right[ii, :], r)
            out[ii, :, :]=(int_x[ii, right_idx.astype(int)+1]-int_x[ii, left_idx.astype(int)])/((right_idx-left_idx+1)[..., np.newaxis])
        int_y = np.cumsum(out, axis=0)
        int_y=np.concatenate([np.zeros((1, width, channel)), int_y], axis=0)
        for ii in range(width):
            up_idx,down_idx=interval_search(cty[:,ii], up[:, ii], down[:, ii], r)
            # if ii==105:
            #     print('done')
            out[:, ii, :]=(int_y[down_idx.astype(int)+1, ii]-int_y[up_idx.astype(int), ii])/((down_idx-up_idx+1)[..., np.newaxis])
        print('Done')

    return out

def DT_filter_NC_parallel(im, out, sigma_s, sigma_r, max_iter, gradient=False):
    dcx=np.zeros((im.shape[0], im.shape[1]-1))
    dcy=np.zeros((im.shape[0]-1, im.shape[1]))
    ratio=sigma_s/sigma_r
    height, width, channel=out.shape
    if not gradient:
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]-im[:, ii]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]-im[ii, :]), 1)
    else: 
        for ii in range(im.shape[1]-1):
            dcx[:, ii]=1+ratio*np.sum(np.abs(im[:, ii+1]), 1)
        for ii in range(im.shape[0]-1): 
            dcy[ii, :]=1+ratio*np.sum(np.abs(im[ii+1,:]), 1)
    
    ctx=np.concatenate([np.zeros((height, 1)), np.cumsum(dcx, axis=1)], axis=1)
    cty=np.concatenate([np.zeros((1, width)),np.cumsum(dcy, axis=0)])
    # pool=Pool(8)
    for iteration in range(max_iter): 
        r=sigma_s*3*(2**(max_iter-iteration-1))/((4**max_iter-1)**0.5)
        int_x =np.cumsum(out, axis=1)
        int_x=np.concatenate([np.zeros((height, 1, channel)), int_x], axis=1)
        # res=pool.starmap(interval_search_P, zip(list(ctx), list(int_x), list(np.ones(height)*r)))
        res=[]
        for ii in range(height):
            res.append(interval_search_P(ctx[ii, :], int_x[ii, :], r))
        out=np.array(res)
        
        int_y = np.cumsum(out, axis=0)
        int_y=np.concatenate([np.zeros((1, width, channel)), int_y], axis=0)
        # res=pool.starmap(interval_search_P, zip(list(cty.T), list(int_y.transpose(1, 0, 2)), list(np.ones(width)*r)))
        res=[]
        for ii in range(width):
            res.append(interval_search_P(cty[:,ii], int_y[:, ii], r))
        out=np.array(res).transpose(1, 0, 2)
        print('Done')

    return out


def interval_search_P(arr, integral, r):
    width=len(arr)
    left=np.zeros_like(arr)
    right=np.zeros_like(arr)
    
    r_idx=int(binary_search(arr[:min(int(r+1), width)], r))
    right[0]=r_idx
    left[:r_idx+1]=0
    
    for ii in range(1, len(arr)): 
        start=max(r_idx, ii)
        r_idx=iter_search(arr[start:min(int(ii+r+1), width)], arr[ii]+r)+start
        right[ii]=r_idx
        left[start:int(r_idx)+1]=ii
        if r_idx>=width-1: 
            break
    for idx in range(ii, len(arr)): 
        right[idx]=width-1
    right=right.astype('int')
    left=left.astype('int')
    count=(right+1-left)
    res=(integral[right+1]-integral[left])/count[..., np.newaxis]
    return res





def interval_search_P_conf(arr, integral, integral_conf, r):
    width=len(arr)
    left=np.zeros_like(arr)
    right=np.zeros_like(arr)
    integral_conf=integral_conf
    
    r_idx=int(binary_search(arr[:min(int(r+1), width)], r))
    right[0]=r_idx
    left[:r_idx+1]=0
    
    for ii in range(1, len(arr)): 
        start=max(r_idx, ii)
        r_idx=iter_search(arr[start:min(int(start+r+1), width)], arr[ii]+r)+start
        right[ii]=r_idx
        left[start:int(r_idx)+1]=ii
        if r_idx>=width-1: 
            break
    for idx in range(ii, len(arr)): 
        right[idx]=width-1
    right=right.astype('int')
    left=left.astype('int')
    count=(right+1-left)
    res=(integral[right+1]-integral[left])/((integral_conf[right+1]-integral_conf[left])+1e-6)
    new_conf=(integral_conf[right+1, 0]-integral_conf[left, 0])/(right+1-left)
    res=np.concatenate([res, new_conf[..., np.newaxis]], axis=1)
    return res


def interval_search(arr, left, right, r):
    width=len(arr)
    
    r_idx=binary_search(arr[:min(int(r+1), width)], r)
    right[0]=r_idx
    left[0:r_idx+1]=0 
    for ii in range(1, len(arr)): 
        # if ii==518: 
        #     print('done')
        #     # pass
        start=max(int(right[ii-1]), ii)
        if right[ii-1]>=width-1: 
            # print(start, width)
            
            right[ii:]=right[ii-1]
            break
        
        r_idx=binary_search(arr[start:min(int(start+r+1), width)], arr[ii]+r)+start
        right[ii]=r_idx
        left[start:int(r_idx)+1]=np.minimum(ii, left[start:int(r_idx)+1])
        # print(start, r_idx)
    return left,right

def binary_search(arr, val): 
    l, r=0,len(arr)-1
    if len(arr)<=2:
        if arr[r]<val:
            return r
        else: 
            return 0

    while l<r-1: 
        m=l+(r+1-l)//2
        # print(l,r,m)
        if (arr[m]<=val and arr[m+1]>val):
            return m
        elif arr[m]>val: 
            r=m
        else: 
            l=m
    return l if l==0 else r

def iter_search(arr, val): 
    if arr[0]>val: 
        return 0
    for ii in range(1, len(arr)): 
        if arr[ii-1]<=val and arr[ii]>val: 
            return ii-1
    return len(arr)-1



def interval_search_P_mask(arr, integral, mask, integral_t, r):
    width=len(arr)
    left=np.zeros_like(arr)
    right=np.zeros_like(arr)
    
    r_idx=int(binary_search(arr[:min(int(r+1), width)], r))
    right[0]=r_idx
    left[:r_idx+1]=0
    
    for ii in range(1, len(arr)): 
        start=max(r_idx, ii)
        r_idx=iter_search(arr[start:min(int(start+r+1), width)], arr[ii]+r)+start
        right[ii]=r_idx
        left[start:int(r_idx)+1]=ii
        if r_idx>=width-1: 
            break
    for idx in range(ii, len(arr)): 
        right[idx]=width-1
    right=right.astype('int')
    left=left.astype('int')
    mask_end=mask_ends(mask)
    while len(mask_end):
        start=mask_end.pop(0)
        end=mask_end.pop(0)
        left[start:end+1]=np.maximum(left[start:end+1], start)
        right[start:end+1]=np.minimum(right[start:end+1], end)

    count=(right+1-left)
    res=(integral[right+1]-integral[left])
    tri=(integral_t[right+1]-integral_t[left])
    trimap=tri>0
    res[trimap]/=tri[trimap, np.newaxis]
    return res, trimap

def mask_ends(mask_column):
    if not mask_column.any() or mask_column.all(): 
        return []
    else: 
        start_end=[]
        state=1
        ii=0
        while ii<len(mask_column):
            if mask_column[ii] != state:
                if state == 1: 
                    current_start=ii
                    state=0 
                elif state == 0: 
                    start_end.append(current_start)
                    start_end.append(ii-1)  
                    state=1
            ii+=1
        if state==0: 
            start_end.append(current_start)
            start_end.append(ii-1)  
        
    return start_end



if __name__=='__main__': 
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--filename", type=str, default = 'imgs/statue.png',help="Image Name")
    ap.add_argument("-s", "--sigmaS", type = float, default = 40, help="Spatial Sigma")
    ap.add_argument("-r", "--sigmaR", type = float, default = 0.77, help="Color Sigma")
    ap.add_argument("-i", "--iteration", type = int, default = 3, help="# of iterations")
    ap.add_argument("-o", "--outputname", type=str, default = 'imgs/output.png',help="Output Name")
    ap.add_argument("-t", "--ftype", type = str, default = 'IC', help="Type of filter", \
        choices=['IC', 'NC'])
    args = ap.parse_args()
    sigmaS = args.sigmaS
    sigmaR = args.sigmaR
    iteration = args.iteration
    ftype = args.ftype
    outputname = args.outputname
    im1 = cv2.imread(args.filename).astype('float32')/255
    
    if ftype == 'IC':
        result = DT_filter_IC(im1, im1.copy(),sigmaS, sigmaR, iteration)
    if ftype == 'NC':
        result = DT_filter_NC_parallel(im1, im1.copy(),sigmaS, sigmaR, iteration)
    result = np.clip(result*255, 0, 255).astype('uint8')
    cv2.imwrite(outputname, result)
    # plt.imshow(result[...,::-1])
    # plt.show()