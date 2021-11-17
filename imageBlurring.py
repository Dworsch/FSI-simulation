import numpy as np
import random
import blurringFunctions as bf
import MarksConstants as mc
import IPython
from misc import round_sig_str as round_sig_str
import cameraSimulation as cm

numZPoints = 50
zTotalSize = 5e-6

def reportProgress(num, total):
    print( round_sig_str(num/total*100) + '%                     ',  end='\r' )
    IPython.display.clear_output(wait=True)

### this function creates a Gaussian distribution which appropriately scales the axial dimension of the 2D wave function.
    
def zGaussian(t, z):
    sigma_x_0 = np.sqrt(mc.hbar/(mc.Rb87_M*2*np.pi*35e3))
    sigma_v = np.sqrt(mc.hbar*np.pi*35e3/mc.Rb87_M)
    sigma  = np.sqrt(sigma_v**2*(t*1e-6)**2 + sigma_x_0**2)
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(z)**2/(2*sigma**2))


def getBlurredImage(imageArray):
    print(len(imageArray))
    
    timeSteps = np.arange(0, 99, 1)
    
    array3D = np.zeros((numZPoints,100,100))
    dzPos = np.linspace(-zTotalSize/2, zTotalSize/2, num=numZPoints)
    blurredwvftnList = []
    
    for i in timeSteps:
        
        xySlice = imageArray[i]
        waveFunctionSlices = []
        for j, zSlice in enumerate(array3D):
            zSlice = xySlice*zGaussian(2*i, dzPos[j])
            waveFunctionSlices.append(zSlice)
        waveFunctionSlices = np.array(waveFunctionSlices)

        binnedwvftn = cm.softwareBinning([10,10], waveFunctionSlices)

        blurredwvftn = bf.XYZ_Blur(binnedwvftn)*2e9
        blurredwvftnList.append(blurredwvftn)
        reportProgress(i, len(timeSteps))
    
    return blurredwvftnList

