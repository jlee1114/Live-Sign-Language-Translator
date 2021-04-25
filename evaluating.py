from training import train_test_loaders
from training import Model
import numpy as np 

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable

#ONNX
import onnx
import onnxruntime as ort

def evaluate(outputs: Variable, labels: Variable) -> float:
    Y = labels.numpy()
    Y_h = np.argmax(outputs, axis=1)
    return float(np.sum(Y_h == Y))

def batch(model: Model, dataloader: torch.utils.data.DataLoader) -> float: 
    score = n = 0.0
    for batch in dataloader: 
        n += len(batch['image'])
        outputs = model(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs, batch['label'][:, 0])
    return score / n 

def results():
    train_load, test_load = train_test_loaders()
    model = Model().float().eval()
    saved_model = torch.load('checkpoint.pth') 
    model.load_state_dict(saved_model)

    #Print Pytorch Model list
    train_accuracy = batch(model, train_load) * 100.
    test_accuracy = batch(model, test_load) * 100. 
    print('-' * 10, 'PyTorch Model', '-'  * 10) 
    print('Training accuracy: %.1f' % train_accuracy)
    print('Validation accuracy: %.1f' % test_accuracy)


    '''
    Transfering the model to ONNX.
    Reference: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    '''
    #The '1' here is just a random number to initialize the model 
    batch_size = 1 
    file = 'ASL.onnx'
    x = torch.randn(batch_size, 1, 28, 28)
    torch.onnx.export(model, x, file, input_names = ['input'])

    #Check that the model works
    ort_session = ort.InferenceSession(file)
    model = lambda x: ort_session.run(None, {'input': x.data.numpy()})[0]
    #Load a model
    train_load, test_load = train_test_loaders(batch_size) 
    
    train_accuracy = batch(model, train_load) * 100.
    test_accuracy = batch(model, test_load) * 100.

    print('-' * 10, ' ONNX Model ', '-'  * 10) 
    print('Training accuracy: %.1f' % train_accuracy)
    print('Validation accuracy: %.1f' % test_accuracy)



if __name__ == '__main__':
    results()


'''
---------- PyTorch Model ----------
Training accuracy: 99.8
Validation accuracy: 96.9
----------  ONNX Model  ----------
Training accuracy: 99.8
Validation accuracy: 97.1
'''