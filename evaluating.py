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
    classes = ('A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    train_load, test_load = train_test_loaders()
    model = Model().float().eval()
    saved_model = torch.load('checkpoint.pth') 
    model.load_state_dict(saved_model)
    with torch.no_grad():
        for data in test_load:
            images = Variable(data['image'].float())
            labels = Variable(data['label'].long())
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
                
    print('-' * 8, 'Accuracy per class', '-'  * 8) 
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for letter {:5s} is: {:.1f} %".format(classname, accuracy))
    
    #  Print Pytorch Model results 
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

'''