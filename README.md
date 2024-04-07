# Head-Counting
This project utilizes image segmentation techniques to perform head counting in images. The method employed here involves training a convolutional neural network (CNN) on labeled images with corresponding mask images, enabling accurate segmentation of heads within the images. The segmented regions are then counted to provide an estimate of the number of heads present.

The convolutional neural network architecture used in this project follows the U-Net structure. U-Net is a popular architecture for image segmentation tasks, characterized by its encoder-decoder design with skip connections. The encoder pathway captures context, while the decoder pathway facilitates precise localization. The final layer of the network employs a sigmoid activation function to generate pixel-wise predictions, indicating the probability of each pixel belonging to the target class.

Key Features:

- Utilizes  image segmentation techniques for accurate head detection.

- Training performed on a dataset of labeled images and mask images, ensuring precise segmentation.

- Implementation using deep learning framework TensorFlow for efficient training and inference.

- Can be extended to handle various scenarios and environments where head counting is required, such as crowd monitoring, security surveillance, or event management.


![Screenshot (824)](https://github.com/Mesheswarage/Head-Counting/assets/97176530/b050d7ff-9c0f-4018-901f-a76d34b09d59)

![Screenshot (873)](https://github.com/Mesheswarage/Head-Counting/assets/97176530/43ef3fda-11bb-494e-8423-168114f95d19)
