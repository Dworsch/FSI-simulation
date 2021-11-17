import numpy as np

fsill = 0.0156
fimg = 1
d = 0.3048
wo1 = 7e-7
Si2BC = 1
So1BC = 0.0156
lamb = 7.8e-7

def minBlur(So1):
    # Catch case where infinites cancel each other
    if So1 == fsill:
        mTot = 64
        wi2 = wo1*mTot
        Si2 = 1
        dcam = 1-Si2
        zr = (np.pi/lamb)*wi2**2
        
        minBlur = wi2*np.sqrt(1+(dcam/zr)**2)
        
    else: 
        Si1 = -(fsill*So1)/(fsill-So1)
        So2 = d-Si1
        Si2 = -(fimg*So2)/(fimg-So2)    

        mTot = (Si1*Si2)/(So1*So2)
        wi2 = np.abs(wo1*mTot)
        dcam = 1-Si2
        zr = (np.pi/lamb)*wi2**2
        
        minBlur = wi2*np.sqrt(1+(dcam/zr)**2)
        
    return minBlur
