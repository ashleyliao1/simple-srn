class srn:
    def __init__(self, inSize, outSize, hidSize, LR, activationType):
        self.I = random.random((1,inSize)) # input
        self.H = random.random((1,hidSize)) # hidden
        self.O = random.random((1,outSize)) # output
        self.C = random.random((1,hidSize)) # context
        self.error = []
        self.wIH = 2*random.random((inSize,hidSize))-1 # weights from I to H
        self.wHO = 2*random.random((hidSize,outSize))-1 # etc.
        self.wCH = 2*random.random((hidSize,hidSize))-1
        self.LR = LR # learning rate
        self.biasH = np.zeros((1,hidSize)) # bias nodes
        self.biasO = np.zeros((1,outSize))    
        self.activationType = activationType # tanh or sigmoid?
    def activate(self, vectIn):
        inact = np.matrix(vectIn)*self.wIH + np.matrix(self.C)*self.wCH + self.biasH
        self.H = 1 / (1 + np.exp(-inact))        
        inact = np.matrix(self.H)*self.wHO + self.biasO
        if self.activationType=='tanh':
            self.O = np.tanh(inact) # tanh version 
        else:
            self.O = 1 / (1 + np.exp(-inact)) # sigmoid      
        self.C = self.H
        return(self.O)
    def iterate(self, vectIn, nIterations): # note: assumes len(self.I) = len(self.O)
        outputs = np.empty((nIterations,4),float)*0
        for i in range(nIterations):
            inact = np.matrix(vectIn)*self.wIH + np.matrix(self.C)*self.wCH + self.biasH            
            self.H = 1 / (1 + np.exp(-inact))        
            inact = np.matrix(self.H)*self.wHO + self.biasO
            if self.activationType=='tanh':
                self.O = np.tanh(inact) # tanh version 
            else:
                self.O = 1 / (1 + np.exp(-inact)) # sigmoid   
            outputs[i,:] = self.O
            vectIn = self.O
            self.C = self.H
        return(outputs)
    def iterateSymbolic(self, vectIn, nIterations): # note: assumes len(self.I) = len(self.O)
        outputs = np.empty((1,nIterations),float)*0
        for i in range(nIterations):
            inact = np.matrix(vectIn)*self.wIH + np.matrix(self.C)*self.wCH + self.biasH            
            self.H = 1 / (1 + np.exp(-inact))        
            inact = np.matrix(self.H)*self.wHO + self.biasO
            if self.activationType=='tanh':
                self.O = np.tanh(inact) # tanh version 
            else:
                self.O = 1 / (1 + np.exp(-inact)) # sigmoid   
            sample = get_activation_sample(self.O)
            ix = random.choice(sample)
            outputs[0,i] = ix
            np.multiply(range(np.shape(self.O)[1]),0)
            vectIn[ix] = 1
            self.C = self.H
        return(outputs)
    
def get_activation_sample(vect):
    counts = np.matrix(np.round(np.multiply(vect,100)))
    ixSample = []
    for i in range(np.shape(vect)[1]):
        ixSample.append([i]*counts[0,i])
    return(np.concatenate(ixSample))

def pattern_update(vectIn,vectOut,network,train):
    ##########################################################################################
    # calculate activations
    ##########################################################################################       
    inact = np.matrix(vectIn)*network.wIH + np.matrix(network.C)*network.wCH + network.biasH
    network.H = 1 / (1 + np.exp(-inact))        
    inact = np.matrix(network.H)*network.wHO + network.biasO
    if network.activationType=='tanh':
        network.O = np.tanh(inact) # tanh version 
        # network.O = 2 * (1 / (1 + np.exp(-inact)) - .5) # quasi-tanh version             
    else:
        network.O = 1 / (1 + np.exp(-inact)) # sigmoid            
    ##########################################################################################
    # get error and save error of some kind
    ##########################################################################################      
    error = np.subtract(vectOut,network.O)
    if network.activationType=='tanh':        
        network.error.append(-spatial.distance.cosine(vectOut,network.O)+1) # cosine
    else:
        network.error.append(np.mean(np.power(error,2))) # rmse
    ##########################################################################################
    # get deltas from error
    ##########################################################################################
    if network.activationType=='tanh':        
        dO = np.multiply(.5 * (1 - np.power(network.O,2)), error) # tanh f'
    else:
        dO = np.multiply(np.multiply(network.O, np.subtract(1,network.O)), error) # sigmoid       
    dwHO = network.LR * np.transpose(network.H) * dO
    sumTerm = dO * np.transpose(np.matrix(network.wHO))
    dH = np.multiply(np.multiply(network.H, np.subtract(1,network.H)), sumTerm)
    dwIH = network.LR * np.transpose(np.matrix(vectIn)) * dH
    dwCH = network.LR * np.transpose(np.matrix(network.C)) * dH
    ##########################################################################################
    # apply deltas
    ##########################################################################################       
    if train==1:
        network.wHO = network.wHO + dwHO
        network.wIH = network.wIH + dwIH
        network.wCH = network.wCH + dwCH                          
        network.biasH = network.biasH + network.LR * dH
        network.biasO = network.biasO + network.LR * dO
        network.C = network.H # remember context    
    # here you go!
    return(network)

def train_srn(vectsIn,vectsOut,network,nIterations):    
    wordIndex = 0            
    for i in range(nIterations):        
        pattern_update(vectsIn[wordIndex],vectsOut[wordIndex],network,train=1)        
        wordIndex = wordIndex + 1 # loop through input if out of info
        if wordIndex>=len(vectsIn):
            wordIndex = 0        
    return(network)

def test_srn(vectsIn,vectsOut,network,nIterations):    
    wordIndex = 0            
    for i in range(nIterations):        
        pattern_update(vectsIn[wordIndex],vectsOut[wordIndex],network,train=0)        
        wordIndex = wordIndex + 1 # loop through input if out of info
        if wordIndex>=len(vectsIn):
            wordIndex = 0        
    return(network)


