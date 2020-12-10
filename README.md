Datasets:
The city scapes data available in the link mentioned in the project report. Please find the link below as well.
https://www.cityscapes-dataset.com/downloads/
Download the LeftImg8bit_trainvaltest.zip file of size 11GB. The ground truth for the data is available is also available in the dataset under the name gtFine_trainvaltest.zip

After downloading the data. Copy the path of the training and ground truth data set and paste them in the program load_data in the utils folder of the submission. 
the dataset has three divisions in the leftImg8bit: train, val and test. copy the train path for the variable train_dir and val for val_dir. similarly paste the mask path for train_mask and val_mask. 
Further once the model is saved, a quick test run can be done using the program test.py. we need to manually copy and paste a random input image along with its path from the val directory in the program test.py. 

Training: 
Once the paths have been loaded in the load_data in utils. We can just run the train.py program. The program is configured to run for 100 epochs, but can be changed by changing the variable epoch in the train.py program. After each epoch a model is saved in the name fscnn_[epoch number].pth. Change the name of the model in the evaluate.py and the test.py programs to the corresponding 

Evaluation:
If the load_data is configured for the right path, we need to change the model name to the latest model that has been trained and saved the the current directory in the format fscnn_[epoch number].pth. 
Once the right model is loaded, we can run the program evaluate.py which prints the pixel accuracy and meanIoU of each image in the val dataset by the model.
 

 Please note that all the programs are configured to run on gpu as it is copmuationally taxing.