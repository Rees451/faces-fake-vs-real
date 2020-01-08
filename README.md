# Fake vs Real Face Detection
## Project Intro/Objective
The purpose of this project was to develop a classifier to detect fake images of faces. 

### Methods Used
* Machine Learning
* Convolutional Neural Networks

### Technologies
* Python
* jupyter
* Tensorflow (incl. Tensorboard)
* Sklearn 

## Project Structure

1. Raw Data is kept [here](./data/raw) within this repo
2. Some python scripts are [here](./src)
3. Notebooks are kept [here](./notebooks)
4. A demo of the project constructured as a game is kept [here](./demo)

## Featured Notebooks/Analysis/Deliverables
* [EDA/ Data Visualisation](.notebooks/EDA.ipynb)
* [Training Notebook](notebooks/training.ipynb)
* [Evaluation Notebook](./notebooks/Evaluation.ipynb)

## Data

I found the dataset for this project on [kaggle](https://www.kaggle.com/ciplab/real-and-fake-face-detection) it consists of 2000 images of fake and real images (split roughly 50/50). The fake images are split into three categories (easy, medium and hard) and are also broken down by which part of the image had been photoshopped: left eye, right eye, mouth and nose or any combination of these 4.

Below is a selection of the dataset:

![image-20200108150438294](assets/image-20200108150438294.png)

## Game

In order to showcase my model and 'deploy' it I made a game where the player is presented with two picutres, one of which is photoshopped and one which is not. They then must guess which is the fake one. Once the player has made their selection the image is fed into the trained model which also makes a prediction. The aim is to try and get more correct than the algoirthm. To play the game clone my github repo and run the script `demo.py` 

## Contact
* If you want to contact me - reach out to me on [LinkedIn](www.linkedin.com/in/rees) or send me an email

