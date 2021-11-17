import numpy as np

def gaussianFilter(image, size, p0, minBlur, fovSize):
    
    def getFilterSigma(fx,fy,xp0, yp0,minBlurParam,fovSizeParam):
        return (1/fovSize**2)*((fx-xp0)**2+(fy-yp0)**2)+minBlur
    
    def gaussian(intensity, x0, y0, sigma):
        x_values = np.arange(0, size, 1)
        X, Y = np.meshgrid(x_values, x_values)
        return ((intensity)/(2*np.pi*(sigma**2))*np.exp(-(((X-x0)**2)+(Y-y0)**2)/(2*sigma**2)))
     
    result = np.zeros((size,size))
    
    for rowi, row in enumerate(image):
        for coli in range(0, len(row), 1):
            filterSigma = getFilterSigma(rowi, coli, p0, p0, minBlur, fovSize)
            intensityDist = image[rowi, coli]
            result += gaussian(intensityDist, coli, rowi, filterSigma)
        
    return result