# Generative-Models-for-Video-Prediction
Utilize multiple model architectures to improve sharpness in video prediction

# Introduction
Unsupervised frame prediction of videos is a promising direction of study due to the abundance of available data. 
This project entails the problem of frame prediction. 
There exists a significant difference from pure image reconstructions since the models needs to be much more accurate 
to be able to predict future frames as well as learn the correct correlation of changes within the image sequences.
The project thus focuses on predicting frames directly in pixel space and tries to address some of the possible problems.
This project encapsulates various model and loss combinations to deal with degrading image sharpness over long term 
predictions. Many different solutions to this problem is explored utilizing GANs and LSTM based 
approaches. The proposed solution to the problem is to combine different model structure and losses to achieve greater 
results in pixel space. The project discusses the baseline models as well as the introduced combined ones in detail. The 4 
novel combined models are depicted as various Adversarial models. 

This project gives an overview of scientific advances covering future frame prediction and proposes a LSTM and Adversarial
based model which utilizes techniques from Computer Vision networks. The presented
architecture is based on the recurrent encoder-decoder framework with convolutional and dense
cells, which allows the preservation of spatio-temporal data correlations.
This project focuses on future frame predictions with mainly future quality in mind. The project includes references to past 
works experiments on proposed models using Atari data-sets. 
The most successful models described in this project are Convolutional LSTM Variational Autoencoder and Generative Inference
Adversarial Convolutional Autoencoder. During experiments both of these models performed comparably better
than the baseline model for this project "Multi-Scale GAN [1]", thus it can be said that the project was able to
reach similar results to the state of the art in this task. Their performance and some potential shortcomings are
discussed extensively in [Evaluation](#evaluation) section (below) of the project.


## Model Structure
Below proposed composite models and main base models from which 
proposed composite models are made of are described. In each header depicting the 
individual model there is a visual representation (Figure) with accurate 
layer names and dimensions. In all of the models described in the following headers, as an 
optimizer Rectified Adam [2] is used due to its performance benefits and fast 
convergence rate.
### Convolutional Autoencoder (C-AE) 
![CONV_AE](./PlotNeuralNetModels/pdf2png/CONV_AE/CONV_AE-1.png)

### Convolutional Variational Autoencoder (C-VAE)
![CONV_VAE](./PlotNeuralNetModels/pdf2png/CONV_VAE/CONV_VAE-1.png?raw=true "title")

###  Fully-Connected Autoencoder (FC-AE)
![Dense_AE](./PlotNeuralNetModels/pdf2png/Dense_AE/Dense_AE-1.png?raw=true "title")

###  Fully-Connected Variational Autoencoder (FC-VAE)
![Dense_VAE](./PlotNeuralNetModels/pdf2png/Dense_VAE/Dense_VAE-1.png?raw=true "title")

###  Convolutional Generative-Adversarial Autoencoder (C-GA-AE)
![CONV_GAAE](./PlotNeuralNetModels/pdf2png/CONV_GAAE/CONV_GAAE-1.png?raw=true "title")

###  Fully-Connected Generative-Adversarial Autoencoder (FC-GA-AE)
![Dense_GAAE](./PlotNeuralNetModels/pdf2png/Dense_GAAE/Dense_GAAE-1.png?raw=true "title")

###  Convolutional Generative-Inference-Adversarial Autoencoder (C-GIA-AE)
![CONV_GIAAE](./PlotNeuralNetModels/pdf2png/CONV_GIAAE/CONV_GIAAE-1.png?raw=true "title")

###  Fully-Connected Generative-Inference-Adversarial Autoencoder (FC-GIA-AE)
![Dense_GIAAE](./PlotNeuralNetModels/pdf2png/Dense_GIAAE/Dense_GIAAE-1.png?raw=true "title")

###  Convolutional Inference-Adversarial Autoencoder (C-IA-AE) 
![CONV_IAAE](./PlotNeuralNetModels/pdf2png/CONV_IAAE/CONV_IAAE-1.png?raw=true "title")

### Full-Connected Inference-Adversarial Autoencoder (FC-IA-AE)
![Dense_IAAE](./PlotNeuralNetModels/pdf2png/Dense_IAAE/Dense_IAAE-1.png?raw=true "title")

### Convolutional LSTM Autoencoder (C-LSTM-AE)
![CONV_LSTM_AE](./PlotNeuralNetModels/pdf2png/CONV_LSTM_AE/CONV_LSTM_AE-1.png?raw=true "title")

### Convolutional LSTM Variational Autoencoder (C-LSTM-VAE)
![CONV_LSTM_VAE](./PlotNeuralNetModels/pdf2png/CONV_LSTM_VAE/CONV_LSTM_VAE-1.png?raw=true "title")

### Convolutional Time-Distributed Autoencoder (C-TD-AE)
![Time_CONV_AE](./PlotNeuralNetModels/pdf2png/Time_CONV_AE/Time_CONV_AE-1.png?raw=true "title")

### Convolutional Time-Distributed Variational Autoencoder (C-TD-VAE)
![Time_CONV_VAE](./PlotNeuralNetModels/pdf2png/Time_CONV_VAE/Time_CONV_VAE-1.png?raw=true "title")



## Datasets
Atari Datasets [3] originally intended for RNN(Recurrent Neural Network)usage, consists of recorded game play of 5 old Atari console games. This Thesis uses all 5 of them on different models as well as on 2 reference works as baselinefor the models. Below are some details and samples from all 5 games.
* Ms. Pacman : 1172401 training images 381757 validation images (a)
* Video Pinball : 901479 training images 295328 validation images (b)
* Q*bert : 1124726 training images 360062 validation images (c)
* Montezuma’s Revenge : 1783796 training images578075 validation images (d)
* Space Invaders : 1331762 training images 434316 validation images (e)

![Pacman](./PlotNeuralNetModels/dataset_samples/Pacman.png?raw=true "title") ![Pinball](./PlotNeuralNetModels/dataset_samples/Pinball.png?raw=true "title") ![Qbert](./PlotNeuralNetModels/dataset_samples/Qbert.png?raw=true "title") ![Revenge](./PlotNeuralNetModels/dataset_samples/Revenge.png?raw=true "title") ![Spaceinvaders](./PlotNeuralNetModels/dataset_samples/Spaceinvaders.png?raw=true "title")

# Evaluation 
Below are some of the results from experiments on Atari datasets using the models described in this work.
For demonstration purposes, some notebooks from the proposed models are prepared, linked below beside the results.
Notebooks are currently hosted on Google Colaboratory for testing purposes. In order to retrain the best models on the datasets.
Please click the "Open in colab" icon and copy the notebook to run the code used in the notebooks are open source.
Using GPU env is suggested since some trainings took ~30 hours.

### Results on Pacman
[comment]: <> (![Pacman]&#40;./GIFS/original.gif?raw=true "title"&#41;)
#### Best Results 
- Pacman
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1TTYoZ7IwvLIRNXMiyDQpRWsL_qd2ij8c#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1DqcxfpD4ya6eT_7XO69fVM9Gy6rhikj_#offline=true&sandboxMode=true)
       
<table>
  <tr>
     <td>Ground Truth</td>
     <td>Multi Scale GAN [1]</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/Multi-Scale GAN.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvGIAAE.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvLSTMVAE.gif" width=160 height=210></td>
  </tr>
 </table>

#### Worst Results 
<table>
  <tr>
     <td>Ground Truth</td>
     <td>FC-AE</td>
     <td>FC-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseAE.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseVAE.gif" width=160 height=210></td>
  </tr>
 </table>

#### All models 

<table>
  <tr>
     <td>Ground Truth</td>
     <td>FC-AE</td>
     <td>FC-VAE</td>
     <td>C-AE</td>
     <td>C-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=80 height=105 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseVAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvVAE.gif" width=80 height=105 ></td>
  </tr>

  <tr>
     <td>Ground Truth</td>
     <td>C-LSTM-AE</td>
     <td>C-LSTM-VAE</td>
     <td>C-TD-AE</td>
     <td>C-TD-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=80 height=105 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvLSTMAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvLSTMVAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/TimeDistAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/TimeDistVAE.gif" width=80 height=105 ></td>
  </tr>

  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>FC-GIA-AE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=80 height=105 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvGIAAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseGIAAE.gif" width=80 height=105 ></td>
  </tr>
 </table>

### Results on Pinball
- Pinball
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1ZsFM18WAL_JgXPK3Gm0hKqY9E1k9rbTA#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1fGICc07-HJ4oQh9C2cjlwlKLpjPLlk8y#offline=true&sandboxMode=true)
     
<table>
  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/original_pinball.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvGIAAE_pinball.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvLSTMVAE_pinball.gif" width=160 height=210></td>
  </tr>
 </table>

### Results on Q*bert
- Q*bert
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1E0RzzvBlG5uh2G6ADJkNNamx-_sk9LG3#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1pxPQuvcPe5GeFtHdoIUEpSwLEnNkI_WN#offline=true&sandboxMode=true)
       

<table>
  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/original_qbert.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvGIAAE_qbert.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvLSTMVAE_qbert.gif" width=160 height=210></td>
  </tr>
 </table>

### Results on Montezuma’s Revenge

- Montezuma’s Revenge
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1jhqtOFqWL-4cbHLr4xnwTsT17C5sbsSg#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1KfmzpITMH8USwjObSMVhIQ-6cfectYi_#offline=true&sandboxMode=true)
       
<table>
  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/original_revenge.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvGIAAE_revenge.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvLSTMVAE_revenge.gif" width=160 height=210></td>
  </tr>
 </table>

### Results on Space Invaders

- Space Invaders
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1IM4toZY-3UcxanZMsoc8KcGKScE4AbDB#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1fYjIDsAOwo0mlfbtRD26lFcZ9RFK6K3j#offline=true&sandboxMode=true)

<table>
  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/original_spaceinvaders.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvGIAAE_spaceinvaders.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvLSTMVAE_spaceinvaders.gif" width=160 height=210></td>
  </tr>
 </table>


## Overall guide
### Installation instructions required packages
Requirements: **Tensorflow, Keras, Colorlog, Pillow, Tensorflow-probabilities, Jupyter-notebooks**

Google Colaboratory examples given above.

Downloading datasets for local use: 

    - Linux based systems:  Script_dir = 'data'+sep_local+'download_atari_datasets.sh'
                            Script call to download using dataset_name 
                            !/bin/bash $Script_dir -f $DATA_DOWN_PATH -d $dataset_name

    - Windows based systems:    No script created Manually downloadable from 
                                https://github.com/yobibyte/atarigrandchallenge
## References
[1] M. Mathieu, C. Couprie, and Y. LeCun., “Deep multi-scale video predictionbeyond mean square error.,”ICLR., Feb. 2016.

[2] L. Liyuan, J. Haoming, H. Pengcheng, C. Weizhu, L. Xiaodong, G. Jian-feng, and H. Jiawei, “On the variance of the adaptive learning rate andbeyond,”ICLR 2019, April. 2019.

[3] V. Kurin, S. Nowozin, K. Hofmann, L. Beyer, and B. Leibe, “The atarigrand challenge dataset.,”arXiv:1705.10998., 2017.
