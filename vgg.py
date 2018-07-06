
from __future__ import print_function, division

from  data import perfect500k
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import datasets, models, transforms
import time

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
import os
from NCE import NEG_loss

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in range(1):
            scheduler.step()
            model.train(True)  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloader:
                # get the inputs
                inputs, labels = data
                print (labels)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                ##loss = criterion(outputs, labels)

                output = lemniscate(outputs, labels)
                loss = criterion(output, labels)

                # backward + optimize only if in training phase

                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects / dataset_sizes

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            """
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            """

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    P_PATH='/Users/EASON/Shaofu/PycharmProjects/vgg16_NCE/a.pkl'

    """
    # data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # your image data file
    data_dir = '/data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=4,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    """
    datas = perfect500k('/Users/EASON/Shaofu/orz_data/perfect1', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ]))
    dataset_sizes=len(datas)
    dataloader = Data.DataLoader(datas, batch_size=20,
                                 shuffle=False, num_workers=4)
    # use gpu or not
    use_gpu = torch.cuda.is_available()

    # get model and replace the original fc layer with your fc layer
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, dataset_sizes)

    #model_ft.load_state_dict(torch.load(P_PATH))
    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    #criterion = nn.CrossEntropyLoss()
    #criterion = NEG_loss(dataset_sizes, dataset_sizes)

    lemniscate = NCEAverage(dataset_sizes, dataset_sizes, 100, 0.07, 0.5)
    criterion = NCECriterion(dataset_sizes)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=60)
    torch.save(model_ft.state_dict(), P_PATH)

