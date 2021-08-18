# **Neuromatch Academy - Deep Learning 2021 Project**

[Anukool Purohit](https://github.com/AnukoolPurohit), [Ahmed Alramly](https://github.com/ahmedramly), [Prakriti Nayak](https://github.com/PrakritiNayak)  


## _**Title**_
### **Abstract**

Brain Computer Interfaces (BCIs) have the ability to revolutionize rehabilitation of quadriplegic, paralyzed, or those who have otherwise lost the ability to perform fine motor activities. Yet, current BCI systems focus more on decoding gross movements from the neural activity and less on sequential movements like handwriting. In this study, we will attempt to build an algorithm to decode the neural activity implicit in handwriting planning in the motor cortex. 

Given that high dimensional neural activity can be represented by a lower number of latent factors, we hypothesize that a decoder that uses these latent representations can decode the signals with higher accuracy.

In this study, we will attempt to extract the latent dynamics of these neurons and use them to predict the alphabetic characters the subject attempted to write. We will use Latent Factor Analysis via Dynamical Systems (LFADS) to extract said latent dynamics and then use them to classify the characters. 

LFADS is a deep learning method that trains a Recurrent Neural Network (RNN) based nonlinear dynamical system to infer the underlying latent dynamics of the observed neural activity. We are benchmarking our model by applying two other well-established deep learning models, a Convolutional Neural Network (CNN) and an RNN, for significance.

We anticipate LFADS to infer perturbations to dynamics that correlate with planned handwriting; a decoder on these inferred dynamics should outperform the other methods in deciphering accuracy.
