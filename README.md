## Introduction 

This project was made by Guim Casadellà, Gerard Grau, Marc Herrero & Pol Resina for the Inditex TECH challenge at the 10th edition of [hackupc](https://hackupc.com/). We have developed a solution that is able to detect duplicate or similar images in a dataset, a clothing recommendation system based on the outfit provided by the user and an assistant system to manage out of stock problems of the clothes in the Inditex dataset.

## Problem setting & approach

### Problem setting

At Inditex, the world's largest fashion retailer, the objective of the challenge was to create an algorithm able to detect duplicate or similar images in a dataset. Even though the problem seems simple, it is quite complex due to the large amount of data and the variety of images, so our solution had to be scalable and robust. Even though we understood the complexity and the implications of the challenge, we decided to go a step further and create a solution that could be applied to a real-world problem. 

Suppose you are walking down the street and you see your favorite influented wearing a cool outfit. You would like to buy the same clothes, but you don't know where to find them or you might not have money enough to purchase these expensive clothes. Our solution is able to detect the clothes in the image and find similar clothes in the Inditex dataset. This way, you can find the clothes you like in a matter of seconds.

Nowadays, dresssing has become a way of expressing ourselves and our personality, so we decided to create a solution that could help people to dress better and more efficiently. Not only does our solution help people to find the clothes they like, but it also helps them to save money and time.


### Approach

Since the aim's original and versioned challenge is to detect similar images, we decided to use a pre-trained model called CLIP, which is able to understand the context of an image. This model is based on a transformer architecture and has been trained on a large dataset of images and text. On the other hand, refering to the clothing recommendation system, we have decided to use semantic segmentation to detect the clothes in the image and find similar clothes in the Inditex dataset.

## CLIP: We have used a pre-trained model to understand the context of the image.

CLIP is a great deep-learning model which works as an embedder understanding visual data in hands with text. This results in interesting performance experience showing a deep contextual understanding. 

Embedder is a really useful tool to understand the context of the image. It is able to understand the context of the image and provide a vector representation of the image. This vector representation can be used to compare images and detect similar images. The closer the vectors are, the more similar the images are. Paralelly, we have also used the same model to detect the context of the text. This way, we can compare the context of the image with the context of the text and detect similar images. In fact, we have used the cosine similarity to compare the vectors and detect similar images. It consists of a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them.


## Semantic Segmentation

It goes without saying that segmantic segmentation is key to detect pixels that belong to the same object. There are many applications such as autonomous driving and in our case, this is crucial to detect different clothes in the same image. 

![Alt Text](/other/semantic_sgm.gif)

More specifically we have used a pre-trained U-NET model. In general, U-NET is a convolutional neural network architecture that is widely used for image segmentation tasks. It consists of an encoder-decoder structure with skip connections that help preserve spatial information and improve segmentation accuracy by minimizing cross-entropy loss function. For further information click [here](https://github.com/levindabhi/cloth-segmentation) where the authors explain the architecture in detail. Besides, we have also attached a picture of the architecture below.

![Alt Text](/other/unet.png) 

As a result of this implementation, we are able to detect different clothes in the same image and segment them with success. A picture is also provided below as an example of the semantic segmentation.

![Alt Text](/other/segclothes.jpeg)

## Results 

Even though we don't have a ground truth to validate results, we have tested our solution with different images and the results are quite promising. As it can be seen in ```example gallery.html```, the model is able to detect similar images and recommend similar clothes. This enables the strength of our solution and its potential to be applied to a real-world problem.
