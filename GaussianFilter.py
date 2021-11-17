import numpy as np

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