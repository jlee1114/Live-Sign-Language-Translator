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
-------- Accuracy per class --------
Accuracy for letter A     is: 100.0 %
Accuracy for letter B     is: 99.1 %
Accuracy for letter C     is: 98.4 %
Accuracy for letter D     is: 99.2 %
Accuracy for letter E     is: 99.8 %
Accuracy for letter F     is: 99.6 %
Accuracy for letter G     is: 94.3 %
Accuracy for letter H     is: 95.4 %
Accuracy for letter I     is: 98.3 %
Accuracy for letter K     is: 98.5 %
Accuracy for letter L     is: 99.5 %
Accuracy for letter M     is: 94.9 %
Accuracy for letter N     is: 94.5 %
Accuracy for letter O     is: 98.8 %
Accuracy for letter P     is: 100.0 %
Accuracy for letter Q     is: 98.8 %
Accuracy for letter R     is: 97.2 %
Accuracy for letter S     is: 92.7 %
Accuracy for letter T     is: 77.0 %
Accuracy for letter U     is: 95.5 %
Accuracy for letter V     is: 97.1 %
Accuracy for letter W     is: 100.0 %
Accuracy for letter X     is: 98.9 %
Accuracy for letter Y     is: 95.8 %
---------- PyTorch Model ----------
Training accuracy: 99.8
Validation accuracy: 96.9
----------  ONNX Model  ----------
Training accuracy: 99.8
Validation accuracy: 96.9
'''