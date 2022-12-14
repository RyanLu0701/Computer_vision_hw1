import numpy as np
import pandas as pd
import os
from pytz import timezone
import tqdm
from random import randint
import torch
import torch.nn as nn
from torchvision import datasets, models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def test(net, testLoader, criterion):
    net.eval()
    totalLoss = 0
    accuracy = 0
    count = 0
    with torch.no_grad():
        for data, label in testLoader:

            data = data.to(device)
            label = label.to(device, dtype=torch.long)
            data = data.float()

            output = net(data)


            Total_output = torch.argmax(output, dim=1)


            loss = criterion(output , label)


            accuracy += (Total_output == label).sum().item()

            totalLoss += loss.item()


    return totalLoss/len(testLoader.dataset) , accuracy

def train(net, trainLoader, testLoader, optimizer, criterion, epochs,scheduler,patience ):

    testAccuracy  = 0
    bestModel     = net
    trigger_times = 0
    last_loss     = 0

    for i in range(0,epochs+1):

        net.train()

        totalLoss    = 0
        accuracy     = 0
        count        = 0

        for idx, (data_x, label) in enumerate(trainLoader) :

            data_x = data_x.to(device)
            label  = label.to(device, dtype=torch.long)

            optimizer.zero_grad()

            data_x = data_x.float()
            output = net(data_x)

            loss          = criterion(output , label)
            Total_output  = torch.argmax(output, dim=1)

            accuracy   +=  (Total_output == label).sum().item()
            totalLoss  +=  loss.item()


            loss.backward()
            optimizer.step()




        current_val_loss , tmpAccuracy = test(net, testLoader, criterion)

        print(f"epoch : [{i+1}/{epochs}] TOTAl Train Loss: {totalLoss / len(trainLoader.dataset):.4f} , TOTAl Train Accuracy: {100*(accuracy / len(trainLoader.dataset)):.2f}  Val Loss: {current_val_loss / len(testLoader.dataset) } , Val Accuracy: {  100*(tmpAccuracy / len(testLoader.dataset))} ")

        scheduler.step(current_val_loss / len(testLoader.dataset))

        if current_val_loss > last_loss:

            last_loss = current_val_loss
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:

                print(f"Early stopping! at {epochs+1} :)) ")

                return net

        if (tmpAccuracy > testAccuracy):

            testAccuracy = tmpAccuracy

            bestModel = net

            print("Saving best model")

            torch.save(bestModel.state_dict(), "model/epoch"+str(i+1)+"_"+str(testAccuracy)+".pt")

            bset_model_path = "model/epoch"+str(i+1)+"_"+str(testAccuracy)+".pt"

    print("Complete training :)")

    return net,bset_model_path

def run_pred(model,path,test_loader):


    model.load_state_dict(torch.load(path))

    print("Testing")
    print('...')
    ###

    model.eval()
    predictions = []
    label = []

    print("start run predict...")

    model.eval()
    # prediction = []

    prediction2 = []
    with torch.no_grad():

        for test_samples in test_loader:

            test_samples = test_samples.to(device)
            test_outputs = model(test_samples)

            test_label2 = test_outputs.cpu().data.numpy()

            # for y in test_label:
            #     prediction.append(y)

            for x in test_label2:
                    prediction2.append(x)


    result_path = os.listdir('test_frame')
    output_name = []
    for i, file in enumerate(result_path):
        output_name.append(file.split("_")[0])


    result = pd.DataFrame()
    result['name'] = output_name
    result = pd.DataFrame()
    result['name'] = output_name
    result['label_all'] = prediction2
    name = result['name'].unique()

    with open('prediction_v2.csv', 'w') as f:
        f.write('name,label\n')
        for i in tqdm.tqdm(name):
            y = np.argmax(result[result['name'] == i]['label_all'].sum())
            f.write('{},{}\n' .format(i, y))

    print('...')
    print('Testing Complete')



