import numpy as np
from misc import round_sig_str as round_sig_str
import IPython

fsill = 0.0156
fimg = 1
d = 0.3048
#wo1 = 7e-7
#wo1 = 6e-7 # From calculation of expected diffraction size.
Si2BC = 1
So1BC = 0.0156
lamb = 7.8e-7
minBlurx = 6.056e-7 #8.3575e-7
minBlury = 6.056e-7 #7.1165e-7

def reportProgress(num, total):
    print( round_sig_str(num/total*100) + '%                     ',  end='\r' )
    IPython.display.clear_output(wait=True)

def minBlur(So1, wo1):
    """Returns a waist
    """
     
    #Update these values from paper (as waists)
    if So1 == fsill:
        minBlurVal = wo1*64
    
    else: 
        Si1 = -(fsill*So1)/(fsill-So1)
        So2 = d-Si1
        Si2 = -(fimg*So2)/(fimg-So2)    

        mTot = (Si1*Si2)/(So1*So2)
        wi2 = np.abs(wo1*mTot)
        dcam = 1-Si2
        zr = (np.pi/lamb)*wi2**2
        
        minBlurVal = wi2*np.sqrt(1+(dcam/zr)**2)  
        #print(mTot, wi2, dcam, zr)
      
    return minBlurVal



def XYZ_Blur(input3D, zTotalSize = 5e-6, numPix = 64, zDepBlur = True, xyDepBlur = True, objectXYTotalSize = 20e-6, fovSize = 20e-6):
    """
    zTotalSize is the z direction length in meters.
    """
    fsill = 0.0156
    all2DSlices = [input3D[:,:,i] for i in range(0,numPix)]
    blurrySlicesGausTest = []
    z_atomPlane = np.linspace(-zTotalSize/2, zTotalSize/2, numPix)
   
   
    for j in range(0,len(all2DSlices)):
        ### Divide by 2 to convert from waist to sigma
        if zDepBlur:
            zxBlurredAmountSigma = 0.5*minBlur(z_atomPlane[j]+fsill, minBlurx)
            zyBlurredAmountSigma = 0.5*minBlur(z_atomPlane[j]+fsill, minBlury)
            #print(zxBlurredAmountSigma)
        else:
            zxBlurredAmountSigma = 0.5*minBlur(fsill, minBlurx)
            zyBlurredAmountSigma = 0.5*minBlur(fsill, minBlury)
           
        reportProgress(j, len(all2DSlices))
            
        xyzBlurrySlice = gaussianFilter(all2DSlices[j], zxBlurredAmountSigma, zyBlurredAmountSigma, objectXYTotalSize, xyDepBlur, fovSize)
                    
        blurrySlicesGausTest.append(xyzBlurrySlice)

    blurrySlicesSum = sum(blurrySlicesGausTest)
    
    return blurrySlicesSum

def gaussianFilter(objectRepr, minBlurSigmax, minBlurSigmay, objectXYTotalSize, xyDepBlur = True, fovSize = 20e-6):
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
  
    def getFilterSigma(fx,fy,xp0,yp0,minBlurParam,fieldOfView):
        """
        Expects minBlurParam, fieldOfView to be in real units.
        Returns in real units.
        """
        fieldOfViewinPix = fieldOfView/mPerObjectPixScale
        fieldofViewBlurinPix = (1/fieldOfViewinPix)*((fx-xp0)**2+(fy-yp0)**2)
        mfieldofViewBlurCamera = mPerImagePixScale*fieldofViewBlurinPix
        return mfieldofViewBlurCamera + minBlurParam 
       
    def gaussian(intensity, x0, y0, sigmax, sigmay):
        x_values = np.arange(0, len(objectRepr[0]), 1)
        X, Y = np.meshgrid(x_values, x_values)
        return (intensity)/(2*np.pi*(np.sqrt(sigmax**2+sigmay**2)))*np.exp(-((X-x0)**2/(2*sigmax**2)+(Y-y0)**2/(2*sigmay**2)))
    
     
    imageRepr = np.zeros(objectRepr.shape)
    #print(minBlurSigma/mPerImagePixScale)
    for rowi, row in enumerate(objectRepr):
        for coli in range(0, len(row), 1):
            if xyDepBlur:
                filterSigmaPixx = getFilterSigma(rowi, coli, center, center, minBlurSigmax, fovSize)/mPerImagePixScale
                filterSigmaPixy = getFilterSigma(rowi, coli, center, center, minBlurSigmay, fovSize)/mPerImagePixScale
            else: 
                filterSigmaPixx = minBlurSigmax/mPerImagePixScale
                filterSigmaPixy = minBlurSigmay/mPerImagePixScale
                
            atomAmp = objectRepr[rowi, coli]
            #if atomAmp != 0:
             #   print('filtsigs', filterSigmaPixx, filterSigmaPixy, 'minblurs', minBlurSigmax, minBlurSigmay)
              #  print(mPerImagePixScale)
            imageRepr += gaussian(atomAmp, coli, rowi, filterSigmaPixx, filterSigmaPixy)
            

        
    return imageRepr