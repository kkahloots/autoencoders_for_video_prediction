# Autoencoders-for-Video-Prediction
This repo is for "Autoencoders for Video Prediction" paper 


## Models
### Full Connected Autoencoder (FC-AE) 
![FC_AE](./pictures/models/Dense_AE-1.png)

### Convolutional Autoencoder (C-AE) 
![C_AE](./pictures/models/CONV_AE-1.png)

### Full Connected Variational Autoencoder (FC-VAE) 
![FC_VAE](./pictures/models/Dense_VAE-1.png)

### Convolutional Variational Autoencoder (C-VAE) 
![C_VAE](./pictures/models/CONV_VAE-1.png)

### Convolutional Time-Distributed/LSTM Autoencoder (C-TD-AE/C-LSTM-AE) 
![C_TD_AE](./pictures/models/Time_CONV_AE-1.png)

### Convolutional Time-Distributed/LSTM Variational Autoencoder (C-TD-VAE/C-LSTM-VAE) 
![C_TD_VAE](./pictures/models/Time_CONV_VAE-1.png)


## Datasets
Atari Datasets [3] originally intended for RNN(Recurrent Neural Network)usage, consists of recorded game play of 5 old Atari console games. This Thesis uses all 5 of them on different models as well as on 2 reference works as baselinefor the models. Below are some details and samples from all 5 games.
* Ms. Pacman : 1172401 training images 381757 validation images (a)
* Video Pinball : 901479 training images 295328 validation images (b)
* Q*bert : 1124726 training images 360062 validation images (c)
* Montezuma’s Revenge : 1783796 training images578075 validation images (d)
* Space Invaders : 1331762 training images 434316 validation images (e)
