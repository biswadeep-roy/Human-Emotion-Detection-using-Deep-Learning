# Human-Emotion-Detection-using-Deep-Learning
This project implements a deep learning model for the detection of human emotions in images. It uses Convolutional Neural Networks (CNNs) to classify images into three emotional categories: angry, happy, and sad.

![image](https://github.com/biswadeep-roy/Human-Emotion-Detection-using-Deep-Learning/assets/74821633/fbb5eabe-0dc9-46eb-92fb-b049ac031958)


## Dataset
The dataset used in this project is the "Human Emotions Dataset" available on Kaggle. It consists of labeled images representing different emotions.

### Dataset Link
[Human Emotions Dataset](https://www.kaggle.com/muhammadhananasghar/human-emotions-datasethes)

## Model Architecture
The model architecture is based on a modified LeNet-5 CNN architecture. It includes layers for data preprocessing, convolution, batch normalization, max-pooling, dropout, and fully connected layers. Regularization techniques are used to prevent overfitting.

## Configuration
The project is configured with various hyperparameters, including batch size, image size, learning rate, and dropout rate, which can be adjusted as needed.

## Training
The model is trained using TensorFlow and Keras. The training dataset is shuffled and preprocessed, and the model's performance is evaluated using a validation dataset.

## Usage
To use this project, follow these steps:

1. Clone the repository:
<br/>
`git clone https://github.com/biswadeep-roy/Human-Emotion-Detection-using-Deep-Learning/`
<br/>
`cd human-emotion-detection`
<br/>


2. Install the required libraries:

`pip install -r requirements.txt`


3. Download and unzip the dataset (or use your own dataset) into the appropriate directories:
- Training data: `/content/dataset/Emotions Dataset/Emotions Dataset/train`
- Validation data: `/content/dataset/Emotions Dataset/Emotions Dataset/test`

4. Adjust the configuration in the code as needed.

5. Train the model.


6. Evaluate the model.



7. Make predictions on new data:




## Results and Outcomes
# Model Performance
The model achieved a significant level of accuracy in classifying human emotions in images.
Training and validation metrics, such as accuracy and loss, can be found in the training logs.
# Key Outcomes
Successful implementation of a modified LeNet-5 architecture for emotion detection.
Efficient preprocessing of image data, including resizing and rescaling.
Effective use of regularization techniques to prevent overfitting.
Deployment of a TensorFlow-based deep learning model.
# Challenges Faced
Mention any challenges you encountered during the project, such as data quality issues, overfitting, or performance optimization.
# Future Improvements
Provide insights into potential future improvements or enhancements for the project.
Ideas may include fine-tuning hyperparameters, exploring different architectures, or incorporating real-time emotion detection from a video stream.
# Acknowledgments
Express gratitude to any contributors, libraries, or datasets that played a significant role in achieving the project's outcomes.
# Visualizations
Include any relevant visualizations, such as confusion matrices or ROC curves, to showcase the model's performance.

## Author
- (https://github.com/biswadeep_roy)

## License
This project is licensed under the MIT License - see the [MIT](LICENSE) file for details.

Feel free to reach out if you have any questions or suggestions!


