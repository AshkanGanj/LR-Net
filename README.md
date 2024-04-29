# LR-Net
> ### LR-Net : A Block-based Convolutional Neural Network for Low-Resolution Image Classification
> _Ashkan Ganj • Mohsen Ebadpour • Mahdi Darvish • Hamid Bahador_
> 
><a href="https://doi.org/10.1007/s40998-023-00618-5">https://doi.org/10.1007/s40998-023-00618-5</a>
# How to run
Please download the Mnist-digit, Mnist-Fashion, and Mnist oracle datasets. 
and run for trianing
```bash
python dev/LR-Net.py
```

# Method
![image](https://user-images.githubusercontent.com/55941654/183844100-9f79ccd5-0b6e-45f8-8197-d18041ae6d28.png)

<p>As you can see from the provided picture, our architecture consists of 3 Multi-kernel blocks that stack on top of each other, including the steps that our models take to learn the features and details. After that, we add some Fully connected layers and the last layer with a sigmoid activation function for classifying the outputs. The proposed architecture uses the concept of inception and residual connections in MK blocks to provide robust performance for classifying images. As we know the inception modules were introduced as a way mapping is generally easier to tweak than the original one. In theory, fitting an identity mapping by a stack of nonlinear functions requires less effort than pushing the residual to zero if an identity mapping is optimal.</p>

# Results
<p> As shown in bellow image, our model easily outperformed 3 other state-of-art models in terms of accuracy in all 3 datasets, without experiencing overfitting or a gradient exploding problem. It is important to note that the introduced model is a simpler and easier model than the other two. We can see this by looking at the number of parameters in table II, and despite the lower number of parameters, it outperformed the other famous models due to the unique MK-blocks we introduced in this section III. By looking at the tables and charts one point that we can understand is that the model’s performance is directly impacted by the image’s detail and noise.</p>
<p> For example, model’s accuracy is lower on Oracle-MNIST dataset because the dataset contains a lot of noisy images, and also the inner-variance in the labels affects the performance of the model, but compared to other state-of-the-art models, the introduced method can achieve a higher and acceptable result with an accuracy  of 95.13. We were expecting these results because of our unique block-based architecture that we introduced. In each block we used a number of residual connections between the input and upper levels of each layer in order to maintain the features. This turns out to be very effective at increasing
the meaningful features of models, and also enabled us to overcome the gradient exploding problem that we have in deeper networks. Using residual connections in combination with different kernel sizes in each layer leads to achieving better results, as we can see in both the bellow picture. </p>

![image](https://user-images.githubusercontent.com/55941654/183844145-7a63c67a-f35b-4dda-a7e6-5cd4dbbe8b78.png)



# Citation
```
@article{ganj2023lr,
  title={LR-Net: A Block-based Convolutional Neural Network for Low-Resolution Image Classification},
  author={Ganj, Ashkan and Ebadpour, Mohsen and Darvish, Mahdi and Bahador, Hamid},
  journal={Iranian Journal of Science and Technology, Transactions of Electrical Engineering},
  volume={47},
  number={4},
  pages={1561--1568},
  year={2023},
  publisher={Springer International Publishing Cham}
```
