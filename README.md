# Block-Based-ImageClassification-Architecture
> ### A Block-based Convolutional Neural Network for Low-Resolution Image Classification
> _Ashkan Ganj • Mohsen Ebadpour • Mahdi Darvish • Hamid Bahador_
> 
><a href="https://arxiv.org/abs/2207.09531">https://arxiv.org/abs/2207.09531</a>
<p style="text-align: justify">Abstract - The success of CNN-based architecture on image classification in learning and extracting features made them so popular these days, but the task of image classification becomes more challenging when we apply state of art models to classify noisy and low-quality images. It is still difficult for models to extract meaningful features from this type of image due to its low resolution and the lack of global features. Moreover, high-resolution images need more layers to train which means they take more time and computational power to train. Our method also addresses the problem of vanishing gradients as the layers become deeper. In order to address all these issues, we developed a novel image classification architecture, composed of blocks that are designed to learn subtle details from blurred and noisy low-resolution images. Our design of the blocks was heavily influenced by Residual Connections and Inception modules in order to increase performance and reduce parameter sizes. Using the MNIST datasets, we have conducted extensive experiments that show that the introduced architecture is more accurate and faster than other state-of-the-art Convolutional neural networks. Also, as a result of the special characteristics of our model, it can achieve a better result with fewer parameters. </p>

# Method
![image](https://user-images.githubusercontent.com/55941654/183844100-9f79ccd5-0b6e-45f8-8197-d18041ae6d28.png)

<p>As you can see from the provided picture, our architecture consists of 3 Multi-kernel blocks that stack on top of each other, including the steps that our models take to learn the features and details. After that, we add some Fully connected layers and the last layer with a sigmoid activation function for classifying the outputs. The proposed architecture uses the concept of inception and residual connections in MK blocks to provide robust performance for classifying images. As we know the inception modules were introduced as a way mapping is generally easier to tweak than the original one. In theory, fitting an identity mapping by a stack of nonlinear functions requires less effort than pushing the residual to zero if an identity mapping is optimal.</p>

# Results
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
