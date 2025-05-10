# NeuroScan
 NeuroScan classification Web App
This project is a web-based application for brain tumor classification using deep learning models. The app allows users to upload medical images (such as MRI scans) of the brain, and it classifies them into four categories:

Glioma

Meningioma

Pituitary

No Tumor

The application utilizes a convolutional neural network (CNN) trained on medical image datasets, which has been implemented using TensorFlow and Keras.
![prediction_1245b3a4](https://github.com/user-attachments/assets/aff448dc-e6d3-4ce5-90ec-f57ab21aa958)
![prediction_b661fc31](https://github.com/user-attachments/assets/fe9ed940-9451-430d-902c-40ae3bc73bd4)

Features:
File Upload: Users can upload an image of the brain for classification.

Prediction Output: The model predicts the class of the uploaded image with associated probability scores.

Annotated Image: The app provides an annotated image with the prediction result and probabilities displayed directly on the image.

Image Download: Users can download the annotated image for further analysis.

User Interface: The app provides an easy-to-use interface with a clean design and interactive elements for better usability.

Tech Stack:
FastAPI: For building the backend API to handle image uploads and predictions.

TensorFlow & Keras: For building and deploying the deep learning model.

HTML/CSS/JavaScript: For the front-end to enable interaction with the user.

Pillow: For image processing and adding annotations to the predicted images.

How to Use:
Upload an MRI scan or a brain image in PNG/JPEG format.

The model will classify the image and display the prediction along with probabilities.

The image will be annotated with the predicted result, and you can download it for further review.

Installation:
Clone this repository:
git clone https://github.com/ComputerVision804/brain-tumor-classification
cd brain-tumor-classification
Install the required dependencies:
pip install -r requirements.txt
Run the app:
uvicorn main:app --reload
Visit http://127.0.0.1:8000 in your browser to interact with the application.

Contributing:
Feel free to fork this repository, create an issue for any bugs, or submit a pull request for any improvements or additional features.![101](https://github.com/user-attachments/assets/95141d17-c909-4d0d-b1cc-28877c1784ba)
