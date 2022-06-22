#Import files
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import time

#Data processing functions that can be used to pass into the trainNetwork function

def identityFun(input):
    #Function that returns the input again.
    #Used in the case where there's no pre-processing to be done for the trainNetwork function
    
    return input

def logAll(input):
    #Function that applies a log to all data points
    
    #Dev note: does output.shape[1] always correspond with the number of inputs in a data point?
    
    #Process targets by putting them on a log scale
    output = np.log(input)
    output = output.reshape((output.shape[0], output.shape[1]))
    
    return output
    
def logFirstCol(input):
    #Function that applies a log to the first column of the input
    
    #Dev note: does input.shape[1] always correspond with the number of inputs in a data point?
    
    #Process intensity by putting it on a log scale
    firstCol = input[:, 0:1]
    firstCol = np.log(firstCol)
    output = torch.cat((firstCol, input[:,1:input.shape[1]]), axis = 1)
    
    return output

#Useful functions for machine learning
def trainNetwork(model, loss_function, optimizer, numEpochs, dataloader, numInputs, processInputs = identityFun, processTargets = identityFun):
    #Function to train a neural network
    
    #Parameters:
    #- model: Pytorch neural network model we're training on
    #- loss_function: Loss function that is used for the neural network
    #- optimizer: Optimization algorithm used for neural network. Must already be initialized to the model parameters.
    #- numEpochs: Number of epochs we're training the neural network for
    #- dataloader: DataLoader object that feeds the batches of data. (Assumes that one data point has both the inputs and outputs in it)
    #- numInputs: The number of inputs that the data has
    # - processInputs: Function that performs any needed pre-processing for the inputs before we feed them into the network
    # - processTargets: Function that performs any needed pre-processing for the target values (the actual outputs)
    
    
    
    #Function always assumes that intens column is first column
    #In the future, perhaps move the intensity processing to a different function?
    
    for epoch in range(numEpochs):
    
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(dataloader, 0):

            # Get and prepare input
            totalColumns = data.size()[1]
            numOutputs = totalColumns - numInputs

            preProcessedInputs = data[:, 0:numInputs] #This line doesn't really do anything, delete later?
            preProcessedTargets = data[:, numInputs:(numInputs+numOutputs)]
            
            #Process the inputs and targets
            inputs = processInputs(preProcessedInputs)
            targets = processTargets(preProcessedTargets)
            
            #Move the below code over to target/input processing
##             targets = data[:, numInputs:(numInputs+numOutputs)]

##             #Process intensity by putting it on a log scale
##             intens = data[:, 0:1]
##             intens = np.log(intens)
##             inputs = torch.cat((intens, data[:,1:numInputs]), axis = 1)

            #Comment the next two lines out if not using GPU
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')

            #Normalize inputs
            inputs, targets = inputs.float(), targets.float()
            #targets = targets.reshape((targets.shape[0], numOutputs)) #Check to see if I can put this in target process fun.

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            #inputs = inputs #Probably unnecessary, remove later to verify
            outputs = model(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                     (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')
    
def getModelError(model, epochList, loss_function, trainDataset, testDataset, numInputs, learningRate = 1e-3, batchSize = 32, processInputs = identityFun, processTargets = identityFun):
    
    #Function that trains a model over a set of epochs and saves the model for each epoch set specified.
    #Also returns the MSE error and average percent errors for Max Energy, Total Energy, and Avg Energy for each model.
    #Also returns the time spent on each epoch
    
    #Parameters:
    #- model: PyTorch neural network model we're using
    #- epochList: List of epochs we're tesing over
    #- loss_function: Loss function that we're using
    #- trainDataset: Dataset containing our training data
    #- testDataset: Dataset containing our testing data
    #- numInputs: The number of inputs into our model
    #- learningRate: Learning rate of our optimizer.
    # - processInputs: Function that performs any needed pre-processing for the inputs before we feed them into the network
    # - processTargets: Function that performs any needed pre-processing for the target values (the actual outputs)
    
    mseErrorList = []
    avgErrorList = []
    mseTrainList = []
    avgTrainList = []
    timeList = []
    
    #print("Epochs to test:", epochList)
    
    
    for numEpochs in epochList:
        
        #Reset model parameters
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        print("Training with", numEpochs, "epochs.")
        
        #Define optimizer 
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
        
        #Create dataloader for training set
        dataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        numPoints = trainDataset.shape[0]
        
        #Start clock
        startTime = time.time()
        
        #First train the network
        trainNetwork(model, loss_function, optimizer, numEpochs, dataloader, numInputs = numInputs, processInputs = processInputs, processTargets = processTargets)
        
        #End clock
        endTime = time.time()
        timeSpent = endTime - startTime #In seconds
        
        #Next test the network
        model.eval()
        
        #Create dataloader for testing set
        numTestPoints = testDataset.shape[0] #Number of points in the test dataset
        testDataloader = DataLoader(testDataset, batch_size=math.floor(0.1*numTestPoints), shuffle=True)
        iterDataLoader = iter(testDataloader)
        testData = next(iterDataLoader)
        
        totalColumns = testData.size()[1]
        numOutputs = totalColumns - numInputs
        
        #Get preprocessed inputs and targets
        preProcessedInputs = testData[:, 0:numInputs]
        preProcessedTargets = testData[:, numInputs:(numInputs+numOutputs)]
            
        #Process the inputs and targets
        inputs = processInputs(preProcessedInputs)
        target = processTargets(preProcessedTargets)
        
#         #Process the intens value so it is in a log scale
#         intens = testData[:, 0:1]
#         logIntens = np.log(intens)

#         #Create the final tensor of inputs we will feed into the model
#         inputs = torch.cat((logIntens, testData[:,1:numInputs]), axis = 1)
        
#         #Create the tensor of our actual values
#         target = testData[:, numInputs:(numInputs+numOutputs)]
#         target = np.log(target)

        #Push our input tensor to the GPU
        inputs = inputs.to('cuda')
        #target = target.to('cuda')

        inputs = inputs.float()
        #target = target.reshape((target.shape[0], 3))
        
 

        #Get the model predictions and apply the same processing to the targets
        output = model(inputs)
        

        
        #Initialize error lists
        #Index mappings:
        #0 = Max KE
        #1 = Total Energy
        #2 = Average Energy
        error = [0., 0., 0.]
        percentError = [0., 0., 0.]

        
        print("Calculate error for test")
        for index in range(3):
            error[index] = calc_MSE_Error(target, output, index)
            percentError[index] = calc_Avg_Percent_Error(target, output, index)
            
        #Append error values into our list
        mseErrorList.append(error)
        avgErrorList.append(percentError)
        timeList.append(timeSpent)
        
        #Also retrieve the training error
        
        dataloader = DataLoader(trainDataset, batch_size=math.floor(0.1 * numPoints), shuffle=True)
        iterDataLoader = iter(dataloader)
        trainData = next(iterDataLoader)
        
#         #Process the intens value so it is in a log scale
#         intens = trainData[:, 0:1]
#         logIntens = np.log(intens)

#         #Create the final tensor of inputs we will feed into the model
#         inputs = torch.cat((logIntens, trainData[:,1:numInputs]), axis = 1)

        #Get preprocessed inputs and targets
        preProcessedInputs = trainData[:, 0:numInputs]
        preProcessedTargets = trainData[:, numInputs:(numInputs+numOutputs)]
            
        #Process the inputs and targets
        inputs = processInputs(preProcessedInputs)
        trainTarget = processTargets(preProcessedTargets)

        #Push our input tensor to the GPU
        inputs = inputs.to('cuda')
        #target = target.to('cuda')

        inputs = inputs.float()
        #target = target.reshape((target.shape[0], 3))

        #Get the model predictions and apply a log-scale to our actual values
        #(Model predictions already have a log-scale applied to them)
        trainOutput = model(inputs)
        #trainTarget = np.log(trainData[:, numInputs:(numInputs + numOutputs)])
        
#         print(output)
#         print(target)


        print("Calculate error for train")
    
        trainError = [0., 0., 0.]
        trainPercentError = [0., 0., 0.]
        
        for index in range(3):
            trainError[index] = calc_MSE_Error(trainTarget, trainOutput, index)
            trainPercentError[index] = calc_Avg_Percent_Error(trainTarget, trainOutput, index)
            
        #Append error values
        mseTrainList.append(trainError)
        avgTrainList.append(trainPercentError)
        
        #Save each model
        # Specify a path
        PATH = "Models/" + str(numPoints) + "_points_" + str(numInputs) + "_inputs_" + str(numEpochs) + "_epochs_" +"_2_layers_" + 'lr' + str(learningRate) + "_WP_Constraints" + "_gaussian_noise_0.1" + ".pt"

        # Save
        torch.save(model.state_dict(), PATH)
        
    return mseErrorList, avgErrorList, mseTrainList, avgTrainList, timeList


#Useful functions for error calculations and plotting

def calc_MSE_Error(target, output, index):
    #Calculates the mean squared error between two tensor arrays for one output
    
    #target: Pytorch tensor array of actual values, assumed to be a 2-d tensor.
    #output: Pytorch tensor array of predicted values, assumed to be of same shape as target.
    #index: Index corresponding to either max energy, total energy, or avg energy
    
    result = np.square(np.subtract(np.exp(target[:, index].cpu().detach().numpy()), np.exp(output[:, index].cpu().detach().numpy())).mean())
                       
    return result

def calc_Avg_Percent_Error(target, output, index):
    #Calculates the average relative error between two tensor arrays for one output
    
    #target: Pytorch tensor array of actual values, assumed to be a 2-d tensor.
    #output: Pytorch tensor array of predicted values, assumed to be of same shape as target.
    #index: Index corresponding to either max energy, total energy, or avg energy
    
    difference = np.exp(target[:, index].cpu().detach().numpy()) - np.exp(output[:, index].cpu().detach().numpy())
    difference = np.abs(difference)
    error = np.divide(difference, np.exp(output[:, index].cpu().detach().numpy())) * 100
    
    result = error.mean()
    
    return result


def plot_absolute_error(target, output, logIntens, duration, thickness, spotSize, label):
    
    #%matplotlib inline
    import matplotlib.pyplot as plt
    
    #Parameters:
    #- target: Numpy array of actual values
    #- output: Numpy array of predicted values
    #- logIntens: Array of intensities with log-scale applied
    #- duration: Array of pulse durations
    #- thickness: Array of target thicknesses
    #- spotSize: Array of spot sizes
    #- label: Label for y-axis
    
    # Positive absolute error means the NN had over-estimated the error
    # Negative absolute error means the NN had under-estimated the error
    
    #Original code
#     difference = np.exp(output[:, index].cpu().detach().numpy()) - np.exp(target[:, index].cpu().detach().numpy())
#     absDifference = np.abs(difference)

    difference = output - target
    absDifference = np.abs(difference)

    fig=plt.figure(figsize=(8,8))
    plt.subplot(2, 2, 1)
    plt.scatter(logIntens, difference, color = 'blue')
    plt.xscale('log')
    #plt.ylim([0, -1])
    plt.xlabel(r'Intensity (W cm$^{-2}$)')
    plt.ylabel(label)

    plt.subplot(2, 2, 2)
    plt.scatter(thickness, difference, color = 'blue')
    plt.xlabel(r'Target Thickness ($\mu m$)')
    plt.ylabel(label)

    plt.subplot(2, 2, 3)
    plt.scatter(logIntens, absDifference, color = 'blue')
    plt.xscale('log')
    #plt.ylim([0, 1])
    plt.xlabel(r'Intensity (W cm$^{-2}$)')
    plt.ylabel(label + " (Magnitude)")

    plt.subplot(2, 2, 4)
    plt.scatter(thickness, absDifference, color = 'blue')
    plt.xlabel(r'Target Thickness ($\mu m$)')
    plt.ylabel(label + " (Magnitude)")

    plt.tight_layout()
    plt.show()
    
def plot_relative_error(target, output, logIntens, duration, thickness, spotSize, label):
    
    #Function plots the relative error
    #%matplotlib inline
    import matplotlib.pyplot as plt
    
    #Parameters:
    #- target: Numpy array of actual values
    #- output: Numpy array of predicted values
    #- logIntens: Array of intensities with log-scale applied
    #- duration: Array of pulse durations
    #- thickness: Array of target thicknesses
    #- spotSize: Array of spot sizes
    #- label: Label for y-axis
    
    difference = output - target
    difference = np.abs(difference)
    error = np.divide(difference, output) * 100

    fig=plt.figure(figsize=(6,4))
    plt.subplot(1, 2, 1)
    plt.scatter(logIntens,error, color = 'blue')
    plt.xscale('log')
    plt.xlabel(r'Intensity (W cm$^{-2}$)')
    plt.ylabel(label)

    plt.subplot(1, 2, 2)
    plt.scatter(thickness,error, color = 'blue')
    plt.xlabel(r'Target Thickness ($\mu m$)')
    plt.ylabel(label)
    
    plt.tight_layout()
    plt.show()
    
   