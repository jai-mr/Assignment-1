**TSOAI # ASSIGNMENT – 1**

**Jaideep R**

1. **What are Channels and Kernels (according to EVA)?**

**Channels:**

![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ7RslnmQlquugfmgnJnDk1ZcRgIkyjhTl1Cw&usqp=CAU)

Let's say if we have a word cloud of different words which is a mix of only smaller case then we can divide the image into 26 channels and in case the image had a mix of capital letters then we can divide the image into 52 channels. Essentially we need to understand that all similar features fit into the same channel and each different channel has a different feature.

An image kernel is a small matrix used to apply effects like the ones found in Photoshop or Gimp, such as blurring, sharpening, outlining or embossing. They are also used for 'feature extraction', a technique for determining the most important portions of an image. In this context the process is referred to more generally as "convolution".

When one feeds a CNN colored images, those images come in three channels: Red, Green, Blue.

Say we have a 32 x 32 image like in CIFAR-10. For each of the 32 x 32 pixels, there is a value for the red channel, the green, and the blue, (this value is likely different cross the channels). The CNN interprets one sample as 3 x 32 x 32 block.

In CNN, one can have more than 3 channels, with some networks having 100+ channels. These channels function just like the RGB channels, but these channels are an abstract version of color, with each channel representing some aspect of information about the image.

The layers operate on 3-D chunks of data, where the first two dimensions are (generally) the height and width of an image patch, and the third dimension is a number of such patches stacked over one another, which is also called the number of channels in the image volume. Thus, xx can be viewed as a H×W×CH×W×C vector, where H,WH,W are the dimensions of the image and CC is the number of channels of the given image volume.

**Kernels :**

In image processing, a  **kernel, convolution matrix, or mask**  is a small matrix. It is used for blurring, sharpening, embossing, edge detection, and more. This is accomplished by doing a convolution between a kernel and an image.

Convolution is using a 'kernel' to extract certain 'features' from an input image. Let me explain. A kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner.

In image processing, convolution is the process of transforming an image by applying a kernel over each pixel and its local neighbors across the entire image. The kernel is a matrix of values whose size and values determine the transformation effect of the convolution process.

The Convolution Process involves these steps.

1. It places the Kernel Matrix over each pixel of the image (ensuring that the full Kernel is within the image), multiplies each value of the Kernel with the corresponding pixel it is over.
2. Then, sums the resulting multiplied values and returns the resulting value as the new value of the center pixel.
3. This process is repeated across the entire image.

![](https://miro.medium.com/max/464/0*e-SMFTzO8r7skkpc)

As we see in the image above, describing the operation for 3x3 kernel being convoluted over a 7x7 source image :

Center Element of the kernel is placed over the source pixel. The source pixel is then replaced with a weighted sum of itself and surrounding pixels. The output is placed in the destination pixel value. In this example, at the first position, we have 0 in source pixel and 4 in the kernel. 4x0 is 0, then moving to the next pixel we have 0 and 0 in both places. 0x0 is 0. Then again 0x0 is 0. Next at the center there is 1 in the source image and 0 in the corresponding position of kernel. 0x1 is 0. Then again 0x1 is 0. Then 0x0 is 0 and 0x1 is 0 and at the last position it is -4x2 which is -8. Now summing up all these results we get -8 as the answer so the output of this convolution operation is -8. This result is updated in the Destination image.(Refer table in above image on top right showing the computations.

**Identity Kernel:**

The output of the convolution process changes with the changing kernel values. For example, an Identity Kernel shown below, when applied to an image through convolution, will have no effect on the resulting image. Every pixel will retain its original value as shown in the following figure.

![](https://miro.medium.com/max/93/0*r5ARjKpVERojnPFu)

![](https://miro.medium.com/max/700/1*ukrsCZSyKUYsX9hR2ItJog.png)

**Sharpen Kernel:**

A Sharpen Kernel like this when applied to an image through convolution, will have an image sharpening effect to the resulting image. The precise values can be customized for varying levels of sharpness as shown in the following figure.

![](https://miro.medium.com/max/456/1*DZyIk0Gx2K174hkZym0mmg.png)

![](https://miro.medium.com/max/700/1*VWCiLmwKi-EEeUYQ-tA7gQ.png)

2. **Why should we (nearly) always use 3x3 kernels?**

In image processing, a  **kernel** ,  **convolution matrix** , or  **mask**  is a small matrix. It is used for blurring, sharpening, embossing, edge detection, and more. This is accomplished by doing a convolution between a kernel and an image.

![](https://miro.medium.com/max/464/0*e-SMFTzO8r7skkpc)

A convolution filter passes over all the pixels of the image in such a manner that, at a given time, we take 'dot product' of the convolution filter and the image pixels to get one final value output. We do this hoping that the weights (or values) in the convolution filter, when multiplied with corresponding image pixels, gives us a value that best represents those image pixels. We can think of each convolution filter as extracting some kind of feature from the image.

Therefore, convolutions are done usually keeping these two things in mind -

- Most of the features in an image are usually local. Therefore, it makes sense to take few local pixels at once and apply convolutions.
- Most of the features may be found in more than one place in an image. This means that it makes sense to use a single kernel all over the image, hoping to extract that feature in different parts of the image.

Now that we have convolution filter sizes as one of the hyper-parameters to choose from, the choice is can be made between smaller or larger filter size.

Here are the things to consider while choosing the size —

**Smaller Filter Sizes**
1. We only look at very few pixels at a time. Therefore, there is a smaller receptive field per layer.
2. The features that would be extracted will be highly local and may not have a more general overview of the image. This helps capture smaller, complex features in the image. 
3. The amount of information or features extracted will be vast, which can be further useful in later layers.
4. In an extreme scenario, using a 1x1 convolution is like considering that each pixel can give us some useful feature independently. 
5. Here, we have better weight sharing, thanks to the smaller convolution size that is applied on the complete image. 


**Larger Filter Sizes**
1. We look at lot of pixels at a time. Therefore, there is a larger receptive field per layer.
2. The features that would be extracted will be generic, and spread across the image. This helps capture the basic components in the image. 
3. The amount of information or features extracted are considerably lesser (as the dimension of the next layer reduces greatly) and the amount of features we procure is greater.
4. In an extreme scenario, if we use a convolution filter equal to the size of the image, we will have essentially converted a convolution to a fully connected layer. 
5. Here, we have poor weight sharing, due to the larger convolution size.

Now considering that we have a general idea about the extraction using different sizes, we will follow this up with an experiment convolution of 3X3 and 5X5 —

**Smaller Filter Sizes** 
1. If we apply 3x3 kernel twice to get a final value, we actually used (3x3 + 3x3) weights. So, with smaller kernel sizes, we get lower number of weights and more number of layers. 
2. Due to the lower number of weights, this is computationally efficient. 
3. Due to the larger number of layers, it learns complex, more non-linear features.
4. With more number of layers, it will have to keep each of those layers in the memory to perform backpropogation. This necessitates the need for larger storage memory. 

**Larger Filter Sizes**
1. If we apply 5x5 kernel once, we actually used 25 (5x5) weights. So, with larger kernel sizes, we get a higher number of weights but lower number of layers. 
2. Due to the higher number of weights, this is computationally expensive. 
3. Due to the lower number of layers, it learns simpler non linear features.`
4. With lower number of layers, it will use less storage memory for backpropogation.

Based on the points listed in the above table and from experimentation, smaller kernel filter sizes are a popular choice over larger sizes.

Another question could be the preference for odd number filters or kernels over 2X2 or 4X4.

The explanation for that is that though we may use even sized filters, odd filters are preferable because if we were to consider the final output pixel (of next layer) that was obtained by convolving on the previous layer pixels, all the previous layer pixels would be symmetrically around the output pixel. Without this symmetry, we will have to account for distortions across the layers. This will happen due to the usage of an even sized kernel. Therefore, even sized kernel filters aren't preferred.

1X1 is eliminated from the list as the features extracted from it would be fine grained and local, with no consideration for the neighboring pixels. Hence, 3X3 works in most cases, and it often is the popular choice.

4. **How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 \&gt; 197x197...)**

To come up with a generalized formula for this

Let's say :

1. Original Image is of size n*m ( n &  m can be same as well)
2. Kernel size choosen is : p*q ( p & q can be same as well)

(for this example we will consider this as a single channel image )

Now the new image will be of size (n-p+1) * (m-q+1)

Now the next layer will have a size of ( n-2p+2) * (m-2q+2)

So to generalize if we take l as the number of layer then the size of the image any any given layer will be (n-lp+2) * (m-lq+2).

Now rewritting the equation we find

N = n-lp+l and M = m-lq+l

(where N amd M are new dimensions of the image)

So from above we find

lp = n+l-N ==>  l = (n-N)/(p-1)

similarly

lq = m+l-M ==>  l = (m-M)/(q-1)

Now let's solve the above problem in this example

n=m = 199 ; p=q = 3 ; N=M = 1

now l = (199-1)/(3-1) = 198/2 = 99

so it will take 99 times to reach 1x1 from 199x199

199x199  > 197x197  > 195x195  > 193x193  > 191x191  >

189x189  > 187x187  > 185x185  > 183x183  > 181x181  >

179x179  > 177x177  > 175x175   > 173x173  > 171x171 >

169x169  > 167x167  > 165x165  > 163x163  > 161x161  >

159x159  > 157x157  > 155x155  > 153x153  >  151x151 > 

149x149  > 147x147  > 145x145  > 143x143  > 141x141  > 

139x139  > 137x137  > 135x135  > 133x133  > 131x131  > 

129x129  > 127x127  > 125x125  > 123x123  > 121x121  > 

119x119  > 117x117  > 115x115  > 113x113  > 111x111  > 

109x109  > 107x107  > 105x105  > 103x103  > 101x101  >

99x99    >  97x97   > 95x95    > 93x93    > 91x91    >

89x89    >  87x87   > 85x85    > 83x83    > 81x81    >

79x79    > 77x77    > 75x75    > 73x73    > 71x71    > 

69x69    > 67x67    > 65x65    > 63x63    > 61x61    > 

59x59    > 57x57    > 55x55    > 53x53    > 51x51    > 

49x49    > 47x47    > 45x45    > 43x43    > 41x41    >

39x39    > 37x37    > 35x35    > 33x33    > 31x31    >

29x29    > 27x27    > 25x25    > 23x23    > 21x21    > 

19x19    > 17x17    > 15x15    > 13x13    > 11x11    >

9x9      > 7x7      > 5x5      > 3x3      >  1x1	

Every ">" represents a convolution = 99	

So 99 convolutions are needed to reach 1x1 matrix or pixel from 199x199

4. **How are kernels initialized? **

Initialization of neural networks isn't something we think a lot about nowadays. It's all hidden behind the different Deep Learning frameworks we use, like TensorFlow or PyTorch. However, it's at the heart of why and how we can make neural networks as deep as they are today, and it was a significant bottleneck just a few years ago

The kernels are usually initialized at a seemingly arbitrary value and then one would use a gradient descent optimizer to optimize the values so that the kernels solve your problem.

A neural net can be viewed as a function with learnable parameters and those parameters are often referred to as weights and biases. Now, while starting the training of neural nets these parameters (typically the weights) are initialized in a number of different ways –

- sometimes, using contant values like 0's and 1's,
- sometimes with values sampled from some distribution (typically a unifrom distribution or normal distribution),
- sometimes with other sophisticated schemes like Xavier Initialization. Xavier initializers also sample from a distribution but truncate the values based on the kernel complexity.

The performance of a neural net depends a lot on how its parameters are initialized when it is starting to train. Moreover, if we initialize it randomly for each runs, it's bound to be non-reproducible (almost) and even not-so-performant too. On the other hand, if we initialize it with contant values, it might take it way too long to converge. With that, we also eliminate the beauty of randomness which in turn gives a neural net the power to reach a covergence _quicker_ using gradient-based learning.

Careful initialization of weights not only helps us to develop _more reproducible neural nets but also it helps us in training them better.

Question arises as to what would be the outcome of initialization values that were too small, too large or appropriate?

1. A too-large initialization leads to exploding gradients
(That means that either the weights of the model explode to infinity)

2. A too-small initialization leads to vanishing gradients.
(That means that either the weights of the model vanish to 0)

which make training deep neural networks very challenging. And the deeper the network, the harder it becomes to keep the weights at reasonable values

All in all, initializing weights with inappropriate values will lead to divergence or a slow-down in the training of your neural network.

So the actual question is how to find appropriate initialization values?

To prevent the gradients of the network's activations from vanishing or exploding, we will stick to the following rules of thumb:

1. The mean of the activations should be zero.
2. The variance of the activations should stay the same across every layer.

Under these two assumptions, the backpropagated gradient signal should not be multiplied by values too small or too large in any layer. It should travel to the input layer without exploding or vanishing.

Ensuring zero-mean and maintaining the value of the variance of the input of every layer guarantees no exploding/vanishing signal . This method applies both to the forward propagation (for activations) and backward propagation (for gradients of the cost with respect to activations).

Xavier and He Initialization are two commonly used method that deal with the above discussed problems and provide a pretty nifty solution based on the number of layers and the layer size that will result in a faster convergence than just picking any random value.

The goal of **Xavier Initialization** is to initialize the weights such that the variance of the activations are the same across every layer. This constant variance helps prevent the gradient from exploding or vanishing.

In **He-et-al Initialization** method, the weights are initialized keeping in mind the size of the previous layer which helps in attaining a global minimum of the cost function faster and more efficiently. The weights are still random but differ in range depending on the size of the previous layer of neuron

In essence He initialization works better for layers with ReLu activation while Xavier initialization works better for layers with sigmoid activation

5. **What happens during the training of a DNN?**

Training a neural network typically consists of two phases: A forward phase, where the input is passed completely through the network. A backward phase, where gradients are backpropagated (backprop) and weights are updated

The steps that go into this process are broken down as follows:

- **Step 1: Convolution**
- **Step 1b: ReLU Layer**
- **Step 2: Pooling**
- **Step 3: Flattening**
- **Step 4: Full Connection**

**Detecting Facial Expressions**

Let's look at a case in which a convolutional neural network is asked to read a human smile

 For the purpose of simplification, the following figure shows an extremely basic depiction of a human smile.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/77_blog_image_6.png)

As one can see, the grid table on the far right shows all of the pixels valued at 0's while only the parts where the smiley face appears are valued at 1. In the table above, white cells are represented as 0's, and black cells are represented as 1's, which means that there are no other possible shades that can appear in this image.

What we do when training a convolutional neural network to detect smiles is to teach it the patterns of 0's and 1's that are normally associated with the shape of a smile.

If one looks at the arc of 1's that ends in the second row from the bottom one would be able to recognize the smile.

**Step 1 - Convolution Operation**

In purely mathematical terms, convolution is a function derived from two given functions by integration which expresses how the shape of one is modified by the other.

The following example will provide one with a breakdown of everything one need to know about this process.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_2.png)

Here are the three elements that enter into the convolution operation:

- Input image
- Feature detector
- Feature map

As one can see, the input image is the same smiley face image that we had in the previous tutorial. Again, if one look into the pattern of the 1's and 0's, one will be able to make out the smiley face in there.

Sometimes a 5x5 or a 7x7 matrix is used as a feature detector, but the more conventional one, and that is the one that we will be working with, is a 3x3 matrix. The feature detector is often referred to as a "kernel" or a "filter" which one might come across as one dig into other material on the topic.

It is better to remember both terms to spare yourself the confusion. They all refer to the same thing and are used interchangeably, including in this course.

**To know how a convolution operation work** one can think of the feature detector as a window consisting of 9 (3x3) cells. Here is what you do with it:

- You place it over the input image beginning from the top-left corner within the borders you see demarcated above, and then you count the number of cells in which the feature detector matches the input image.
- The number of matching cells is then inserted in the top-left cell of the feature map.
- You then move the feature detector one cell to the right and do the same thing. This movement is called a and since we are moving the feature detector one cell at time, that would be called a stride of one pixel.
- What one will find in this example is that the feature detector's middle-left cell with the number 1 inside it matches the cell that it is standing over inside the input image. That's the only matching cell, and so one write "1" in the next cell in the feature map, and so on and so forth.
- After one have gone through the whole first row, one can then move it over to the next row and go through the same process.

It's important not to confuse the feature map with the other two elements. The cells of the feature map can contain any digit, not only 1's and 0's. After going over every pixel in the input image in the example above, we would end up with these results:

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_3.png
)

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_4.png)

By the way, just like feature detector can also be referred to as a kernel or a filter, a feature map is also known as an activation map and both terms are also interchangeable.

**What is the point from the convolution operation?**

There are several uses that we gain from deriving a feature map. These are the most important of them: Reducing the size of the input image, and one should know that the larger your strides (the movements across pixels), the smaller your feature map.

In this example, we used one-pixel strides which gave us a fairly large feature map.

When dealing with proper images, one will find it necessary to widen your strides. Here we were dealing with a 7x7 input image after all, but real images tend to be substantially larger and more complex.

**Do we lose information when using a feature detector?**

The answer is  **YES**. The feature map that we end up with has fewer cells and therefore less information than the original input image. However, the very purpose of the feature detector is to sift through the information in the input image and filter the parts that are integral to it and exclude the rest.

Basically, it is meant to separate the wheat from the chaff.

**How do convolutional neural networks actually perform this operation?**

The example we gave above is a very simplified one, though. In reality, convolutional neural networks develop multiple feature detectors and use them to develop several feature maps which are referred to as convolutional layers (see the figure below).

Through training, the network determines what features it finds important in order for it to be able to scan images and categorize them more accurately.

Based on that, it develops its feature detectors. In many cases, the features considered by the network will be unnoticeable to the human eye, which is exactly why convolutional neural networks are so amazingly useful. With enough training, they can go light years ahead of us in terms of image processing.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_5.png) 

**What are other uses of convolution matrices?**

There's another use for convolution matrix, which is actually part of the reason why they are called "filters" The word here is used in the same sense we use it when talking about Instagram filters.

One can actually use a convolution matrix to adjust an image. Here are a few examples of filters being applied to images using these matrices.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_6.png)

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_7.png)

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_8.png)

**Step 1(b): The Rectified Linear Unit (ReLU)**

The Rectified Linear Unit, or ReLU, is not a separate component of the convolutional neural networks' process.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_1.png)

The purpose of applying the rectifier function is to increase the non-linearity in our images.

The reason we want to do that is that images are naturally non-linear.

When one looks at any image, one'll find it contains a lot of non-linear features (e.g. the transition between pixels, the borders, the colors, etc.).

The rectifier serves to break up the linearity even further in order to make up for the linearity that we might impose an image when we put it through the convolution operation.

To see how that actually plays out, we can look at the following picture and see the changes that happen to it as it undergoes the convolution operation followed by rectification.

**The Input Image**

This black and white image is the original input image.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_2.png)

**Feature Detector**

By putting the image through the convolution process, or in other words, by applying to it a feature detector, the result is what one see in the following image.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_3.png)

As one sees, the entire image is now composed of pixels that vary from white to black with many shades of gray in between.

**Rectification**

What the [rectifier function] does to an image like this is remove all the black elements from it, keeping only those carrying a positive value (the grey and white colors).

The essential difference between the non-rectified version of the image and the rectified one is the progression of colors. If one looks closely at the first one, one will find parts where a white streak is followed by a grey one and then a black one. After we rectify the image, one will find the colors changing more abruptly.

The gradual change is no longer there. That indicates that the linearity has been disposed of.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_4.png)

**Step 2 - Max Pooling**

**The Cheetah Example**

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/72_blog_image_1.png)

In the example above, the same cheetah image is presented in different ways. It is normal in its first version, rotated in the second, and horizontally squashed in the third. The purpose of max pooling is enabling the convolutional neural network to detect the cheetah when presented with the image in any manner.

This second example is more advanced. Here we have 6 different images of 6 different cheetahs (or 5, there is 1 that seems to appear in 2 photos) and they are each posing differently in different settings and from different angles.

Again, max pooling is concerned with teaching your convolutional neural network to recognize that despite all of these differences that we mentioned, they are all images of cheetah. In order to do that, the network needs to acquire a property that is known as "spatial variance"

This property makes the network capable of detecting the object in the image without being confused by the differences in the image's textures, the distances from where they are shot, their angles, or otherwise.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/72_blog_image_2.png)

In order to reach the pooling step, we need to have finished the convolution step, which means that we would have a feature map ready.

**Types of Pooling**

Before getting into the details, one should know that there are several types of pooling. These include among others the following:

- Mean pooling
- Max pooling
- Sum pooling

**Pooled Feature Map**

The process of filling in a pooled feature map differs from the one we used to come up with the regular feature map. This time you&#39;ll place a 2x2 box at the top-left corner, and move along the row.

For every 4 cells your box stands on, you'll find the maximum numerical value and insert it into the pooled feature map. In the figure below, for instance, the box currently contains a group of cells where the maximum value is 4.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/72_blog_image_3.png)

In the third movement along the same row, one will find yourself stuck with one lonely column.

One would still proceed despite the fact that half of your box will be empty. You still find your maximum value and put it in the pooled feature map. In the least step, you will face a situation where the box will contain a single cell. You will take that value to be the maximum value.

Just like in the convolution step, the creation of the pooled feature map also makes us dispose of unnecessary information or features. In this case, we have lost roughly 75% of the original information found in the feature map since for each 4 pixels in the feature map we ended up with only the maximum value and got rid of the other 3. These are the details that are unnecessary and without which the network can do its job more efficiently.

The reason we extract the maximum value, which is actually the point from the whole pooling step, is to account for distortions. Let's say we have three cheetah images, and in each image the cheetah's tear lines are taking a different angle.

The feature after it has been pooled will be detected by the network despite these differences in its appearance between the three images. Consider the tear line feature to be represented by the 4 in the feature map above.

Imagine that instead of the four appearing in cell 4x2, it appeared in 3x1. When pooling the feature, we would still end up with 4 as the maximum value from that group, and thus we would get the same result in the pooled version.

This process is what provides the convolutional neural network with the "spatial variance" capability. In addition to that, pooling serves to minimize the size of the images as well as the number of parameters which, in turn, prevents an issue of "overfitting" from coming up.

**Step 3: Flattening**

After finishing the previous two steps, we're supposed to have a pooled feature map by now. As the name of this step implies, we are literally going to flatten our pooled feature map into a column like in the image below.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/73_blog_image_1.png)

The reason we do this is that we&#39;re going to need to insert this data into an artificial neural network later on.

The reason we do this is that we&#39;re going to need to insert this data into an artificial neural network later on.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/73_blog_image_2.png)

As one sees in the image above, we have multiple pooled feature maps from the previous step.

What happens after the flattening step is that you end up with a long vector of input data that you then pass through the artificial neural network to have it processed further

**Step 4: Full Connection**

It's here that the process of creating a convolutional neural network begins to take a more complex and sophisticated turn.

As one sees from the image below, we have three layers in the full connection step:

- Input layer
- Fully-connected layer
- Output layer

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/74_blog_image_1.png)

**The Full Connection Process**

The input layer contains the vector of data that was created in the flattening step. The features that we distilled throughout the previous steps are encoded in this vector.

The role of the artificial neural network is to take this data and combine the features into a wider variety of attributes that make the convolutional network more capable of classifying images, which is the whole purpose from creating a convolutional neural network.

By the end of this channel, the neural network issues its predictions.

**Class Recognition**

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/74_blog_image_3.png)

This full connection process practically works as follows:

- The neuron in the fully-connected layer detects a certain feature; say, a nose.
- It preserves its value.
- It communicates this value to both the "dog" and the "cat" classes.
- Both classes check out the feature and decide whether it's relevant to them.

What we end up with is what one sees in the image below. As this process goes on repeat for thousands of times, one finds yourself with an optimized neural network.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/74_blog_image_4.png)

**The Application**

The next step becomes putting our network's efficacy to the test. Say, we give it an image of a dog.

As one see in the step below, the dog image was predicted to fall into the dog class by a probability of 0.95 and other 0.05 was placed on the cat class.

Think of it this way: This process is a vote among the neurons on which of the classes the image will be attributed to. The class that gets the majority of the votes wins. Of course, these votes are as good as their weights.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/74_blog_image_7.png)

**To Summarize ** The Convolution Process**

In the diagram below, one can see the entire process of creating and optimizing a convolutional neural network that we covered throughout the section.

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/75_blog_image_1.png)

The process goes as follows:

- We start off with an input image.
- We apply filters or feature maps to the image, which gives us a convolutional layer.
- We then break up the linearity of that image using the rectifier function.
- The image becomes ready for the pooling step, the purpose of which is providing our convolutional neural network with the faculty of "spatial invariance" which you'll see explained in more detail in the pooling tutorial.
- After we're done with pooling, we end up with a pooled feature map.
- We then flatten our pooled feature map before inserting into an artificial neural network.

Throughout this entire process, the network's building blocks, like the weights and the feature maps, are trained and repeatedly altered in order for the network to reach the optimal performance that will make it able to classify images and objects as accurately as possible.

**REFERENCES:**
1. https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37
2. https://buzzrobot.com/whats-happening-inside-the-convolutional-neural-network-the-answer-is-convolution-2c22075dc68d
3. https://www.deeplearning.ai/ai-notes/initialization/
4. https://www.superdatascience.com/blogs/the-ultimate-guide-to-convolutional-neural-networks-cnn
5. https://pouannes.github.io/blog/initialization/
6. https://www.wandb.com/articles/the-effects-of-weight-initialization-on-neural-nets
7. https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-what-are-convolutional-neural-networks
8. https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-summary
