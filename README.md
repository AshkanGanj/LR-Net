# LR-Net
> ### LR-Net : A Block-based Convolutional Neural Network for Low-Resolution Image Classification
> _Ashkan Ganj • Mohsen Ebadpour • Mahdi Darvish • Hamid Bahador_
> 
><a href="https://arxiv.org/abs/2207.09531">https://arxiv.org/abs/2207.09531</a>
<p style="text-align: justify">Abstract - The success of CNN-based architecture on image classification in learning and extracting features made them so popular these days, but the task of image classification becomes more challenging when we apply state of art models to classify noisy and low-quality images. It is still difficult for models to extract meaningful features from this type of image due to its low-resolution and the lack of meaningful global features. Moreover, high-resolution images need more layers to train which means they take more time and computational power to train. Our method also addresses the problem of vanishing gradients as the layers become deeper in deep neural networks that we mentioned earlier. In order to address all these issues, we developed a novel image classification architecture, composed of blocks that are designed to learn both low level and global features from blurred and noisy low-resolution images. Our design of the blocks was heavily influenced by Residual Connections and Inception modules in order to increase performance and reduce parameter sizes. We also assess our work using the MNIST family datasets, with a particular emphasis on the Oracle-MNIST dataset, which is the most difficult to classify due to its low-quality and noisy images. We have performed in-depth tests that demonstrate the presented architecture is faster and more accurate than existing cutting-edge convolutional neural networks. Furthermore, due to the unique properties of our model, it can produce a better result with fewer parameters. </p>

# Method
![image](https://user-images.githubusercontent.com/55941654/183844100-9f79ccd5-0b6e-45f8-8197-d18041ae6d28.png)

<p>As you can see from the provided picture, our architecture consists of 3 Multi-kernel blocks that stack on top of each other, including the steps that our models take to learn the features and details. After that, we add some Fully connected layers and the last layer with a sigmoid activation function for classifying the outputs. The proposed architecture uses the concept of inception and residual connections in MK blocks to provide robust performance for classifying images. As we know the inception modules were introduced as a way mapping is generally easier to tweak than the original one. In theory, fitting an identity mapping by a stack of nonlinear functions requires less effort than pushing the residual to zero if an identity mapping is optimal.</p>

# Results
<p> As shown in bellow image, our model easily outperformed 3 other state-of-art models in terms of accuracy in all 3 datasets, without experiencing overfitting or a gradient exploding problem. It is important to note that the introduced model is a simpler and easier model than the other two. We can see this by looking at the number of parameters in table II, and despite the lower number of parameters, it outperformed the other famous models due to the unique MK-blocks we introduced in this section III. By looking at the tables and charts one point that we can understand is that the model’s performance is directly impacted by the image’s detail and noise.</p>
<p> For example, model’s accuracy is lower on Oracle-MNIST dataset because the dataset contains a lot of noisy images, and also the inner-variance in the labels affects the performance of the model, but compared to other state-of-the-art models, the introduced method can achieve a higher and acceptable result with an accuracy  of 95.13. We were expecting these results because of our unique block-based architecture that we introduced. In each block we used a number of residual connections between the input and upper levels of each layer in order to maintain the features. This turns out to be very effective at increasing
the meaningful features of models, and also enabled us to overcome the gradient exploding problem that we have in deeper networks. Using residual connections in combination with different kernel sizes in each layer leads to achieving better results, as we can see in both the bellow picture. </p>

![image](https://user-images.githubusercontent.com/55941654/183844145-7a63c67a-f35b-4dda-a7e6-5cd4dbbe8b78.png)

# Citation
    @misc{https://doi.org/10.48550/arxiv.2207.09531,
      doi = {10.48550/ARXIV.2207.09531},
      url = {https://arxiv.org/abs/2207.09531},
      author = {Ganj, Ashkan and Ebadpour, Mohsen and Darvish, Mahdi and Bahador, Hamid},
      keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.0},
      title = {A Block-based Convolutional Neural Network for Low-Resolution Image Classification},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution 4.0 International}
    }
