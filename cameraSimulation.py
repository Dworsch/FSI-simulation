import numpy as np
import random

QE = 0.9
darkElectronChance = 0.0003
seed = random.randint(1,70)
rs = np.random.RandomState(seed)
x = np.arange(1000)

def gain(x): 
    return np.exp(-0.0137615*x)

def readOutNoise(x):
    return np.exp(-(x-88.4)**2/52.84)

gain_pdf=gain(x)

def softwareBinning(binningParams, rawData):
    if binningParams is not None:
        sb = binningParams
        if len(np.array(rawData).shape) == 3: 
            if not ((rawData.shape[1]/sb[0]).is_integer()): 
                raise ValueError('Vertical size ' + str(rawData.shape[1]) +  ' not divisible by binning parameter ' + str(sb[0]))
            if not ((rawData.shape[2]/sb[1]).is_integer()):
                raise ValueError('Horizontal size ' + str(rawData.shape[2]) +  ' not divisible by binning parameter ' + str(sb[1]))
            rawData = rawData.reshape(rawData.shape[0], rawData.shape[1]//sb[0], sb[0], rawData.shape[2]//sb[1], sb[1]).sum(4).sum(2)
        elif len(np.array(rawData).shape) == 2:
            if not ((rawData.shape[0]/sb[0]).is_integer()): 
                raise ValueError('Vertical size ' + str(rawData.shape[0]) +  ' not divisible by binning parameter ' + str(sb[0]))
            if not ((rawData.shape[1]/sb[1]).is_integer()):
                raise ValueError('Horizontal size ' + str(rawData.shape[1]) +  ' not divisible by binning parameter ' + str(sb[1]))
            rawData = rawData.reshape(rawData.shape[0]//sb[0], sb[0], rawData.shape[1]//sb[1], sb[1]).sum(3).sum(1)
        else:
            raise ValueError('Raw data must either 2 or 3 dimensions')            
    return rawData

def getSimulatedImage(blurredImage, size, binningParams, rawData):
    
    NoisyBlurryImage = np.zeros((size,size))
    randomNoiseImage = np.zeros((size,size))

    # must fix: can't consider photons from different locations (indicating the atom is in different locations)
    #This loop applies Poisson-distributed shotnoise.        
    for rowi, row in enumerate(NoisyBlurryImage):
        for coli, col in enumerate(row):
            #sample from blurred wave function
            #value at blurred image is mean intensity, value returned by rs.poisson is shotnoise
            NoisyBlurryImage[rowi, coli] += rs.poisson(blurredImage[rowi, coli])
            
    #Calculate number of electrons accounting for QE, dark electrons, EM gain noise, and readout noise.
    for rowi, row in enumerate(NoisyBlurryImage):
        for coli, col in enumerate(row):
            numPhotons = NoisyBlurryImage[rowi, coli]
            numElectrons = numPhotons*QE
            #random chance to find a "Dark electron"           
            if np.random.random() < darkElectronChance:
                numElectrons += 1    
                
            NoisyBlurryImage[rowi, coli] += numElectrons
                
 
        #EM gain amplification: using 1 distribution, sampling from it as many times as there are electrons on a pixel.
            if numElectrons == 0:
                electronsGained = 0
            elif numElectrons == 1:
                ElectronsOut = np.random.choice(x, p=gain_pdf/sum(gain_pdf))
                electronsGained = ElectronsOut
            else:
                ElectronsOut = np.random.choice(x, size=int(numElectrons), p=gain_pdf/sum(gain_pdf))
                electronsGained = sum(ElectronsOut)
            NoisyBlurryImage[rowi, coli] += electronsGained
            
 #camera binning in 2D
    print(NoisyBlurryImage.shape)
    binnedNoisyBlurryImage = softwareBinning(binningParams, NoisyBlurryImage)   
    for rowi, row in enumerate(binnedNoisyBlurryImage):
        for coli, col in enumerate(row):   
            
            readOutNoise = np.random.randint(0, 20)
            #counts = readOutNoise

            binnedNoisyBlurryImage[rowi, coli] += readOutNoise
            
    return binnedNoisyBlurryImage
    