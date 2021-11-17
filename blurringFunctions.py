import numpy as np
from misc import round_sig_str as round_sig_str
import IPython

fsill = 0.0156
fimg = 1
d = 0.3048
wo1 = 7e-7
Si2BC = 1
So1BC = 0.0156
lamb = 7.8e-7
size = 50

def reportProgress(num, total):
    print( round_sig_str(num/total*100) + '%                     ',  end='\r' )
    IPython.display.clear_output(wait=True)

def minBlur(So1):
    """Returns a waist
    """
    
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



def XYZ_Blur(input3D, zTotalSize = 5e-6, zDepBlur = True, xyDepBlur = True, fovSize = 20e-6):
    """
    zTotalSize is the z direction length in meters.
    """
    fsill = 0.0156
    all2DSlices = [input3D[i] for i in range(0,size)]
    blurrySlicesGausTest = []
    dz = np.linspace(-zTotalSize/2, zTotalSize/2, size)
   
    for j in range(0,len(all2DSlices)):
        ### Divide by 2 to convert from waist to sigma
        if zDepBlur:
            zBlurredAmountSigma = minBlur(dz[j]+fsill)/2
        else:
            zBlurredAmountSigma = minBlur(fsill)/2
        reportProgress(j, len(all2DSlices))
        
        xyzBlurrySlice = gaussianFilter(all2DSlices[j], zBlurredAmountSigma, 20e-6, xyDepBlur, fovSize)
        blurrySlicesGausTest.append(xyzBlurrySlice)


    blurrySlicesSum = sum(blurrySlicesGausTest)
    
    return blurrySlicesSum

def gaussianFilter(objectRepr, minBlurSigma, objectXYTotalSize, xyDepBlur = True, fovSize = 20e-6):
    """
    fovSize is in real units, xyTotalSize in meters to scale image.
    xyTotalSize is the size of the object array/representation.
    minBlur input should be in real units.
    Warning! Confusing. Here, we try to match geometric optics convention. 
    It is tempting to call objectRepr "image", and imageRepr "magnified image"
        ObjectRepr is object representation of the atoms.
        imageRepr is the geometric optics result at the camera.
    """
    center = len(objectRepr)//2
    magnification = 64
    imageTotalSize = objectXYTotalSize*magnification
    mPerImagePixScale = imageTotalSize/(len(objectRepr))
    mPerObjectPixScale = objectXYTotalSize/(len(objectRepr))
  
    def getFilterSigma(fx,fy,xp0,yp0, minBlurParam,fieldOfView):
        """
        Expects minBlurParam, fieldOfView to be in real units.
        Returns in real units.
        """
        return (mPerObjectPixScale**2)*(1/fieldOfView)*((fx-xp0)**2+(fy-yp0)**2) + minBlurParam
       
    def gaussian(intensity, x0, y0, sigma):
        x_values = np.arange(0, len(objectRepr[0]), 1)
        X, Y = np.meshgrid(x_values, x_values)
        return ((intensity)/(2*np.pi*(sigma**2))*np.exp(-(((X-x0)**2)+(Y-y0)**2)/(2*sigma**2)))
     
    imageRepr = np.zeros(objectRepr.shape)
    #print(minBlurSigma/mPerImagePixScale)
    for rowi, row in enumerate(objectRepr):
        for coli in range(0, len(row), 1):
            if xyDepBlur:
                filterSigmaPix = getFilterSigma(rowi, coli, center, center, minBlurSigma, fovSize)/mPerImagePixScale
            else: 
                filterSigmaPix = minBlurSigma/mPerImagePixScale
                
            intensityDist = objectRepr[rowi, coli]
            imageRepr += gaussian(intensityDist, coli, rowi, filterSigmaPix)
            

        
    return imageRepr