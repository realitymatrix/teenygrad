import numpy as np
E2M1 = np.array([0,.5,1,1.5,2,3,4,6, -0.0,-.5,-1,-1.5,-2,-3,-4,-6], np.float32)
def e4m3(b):
    b=b.astype(np.uint32); s=np.where(b>>7&1,-1.,1.).astype(np.float32)
    e=(b>>3&0xF).astype(np.int32); m=(b&7).astype(np.float32)
    sub=(m/8.)*2.0**-6
    nor=(1.+m/8.)*np.exp2((e-7).astype(np.float32))
    v=np.where(e==0,sub,nor).astype(np.float32)
    v=np.where((e==0xF)&(m==7), np.nan, v)
    return s*v
raw=np.fromfile("w13_rows.bin",np.uint8).reshape(8,2048)
lo,hi=raw&0xF, raw>>4
q=np.empty((8,4096),np.uint8); q[:,0::2]=lo; q[:,1::2]=hi
vals=E2M1[q]
sc=e4m3(np.fromfile("w13_scale.bin",np.uint8).reshape(8,256))
s2=np.fromfile("w13_scale2.bin",np.float32)[0]
w=vals*np.repeat((sc*s2).astype(np.float32),16,axis=1)
np.save("ref_w.npy", w)
print("scale2[0] =",s2, " scale2 range", np.fromfile("w13_scale2.bin",np.float32).min(), np.fromfile("w13_scale2.bin",np.float32).max())
print("e4m3 block scales row0[:6]:", np.round(sc[0,:6],5))
print("dequant w[0,:8]:", np.round(w[0,:8],6))
print("shape",w.shape,"absmax %.5f"%np.abs(w).max(),"std %.5f"%w.std(),"zeros %.1f%%"%(100*(w==0).mean()))
print("nibble histogram:", np.bincount(q.ravel(),minlength=16))
