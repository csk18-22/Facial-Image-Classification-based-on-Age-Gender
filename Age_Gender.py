import cv2
import gradio as gr
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from skimage import exposure

import numpy as np
from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
from numpy.linalg import eigh, solve
from numpy.random import randn

import math
import argparse
import os
from joblib import dump, load
import argparse
import sys

#openCV
"""docstring for FaceRecognition"""
class FaceRecognition():
    def __init__(self):
        self.faceProto="models/opencv_face_detector.pbtxt"
        self.faceModel="models/opencv_face_detector_uint8.pb"
        self.face_net = cv2.dnn.readNet(self.faceModel, self.faceProto)

    def detect_face(self, image, threshold = 0.7):
        temp_image = image.copy()
        height, width, channels = temp_image.shape
        blob = cv2.dnn.blobFromImage(temp_image, 1.0, (300, 300), [104, 117, 123], True, False)

        self.face_net.setInput(blob)
        detected = self.face_net.forward()
        faceBoxes = []
        for row in range(detected.shape[2]):
            confidence = detected[0, 0, row, 2]
            if confidence > threshold:
                left = int(detected[0, 0, row, 3] * (width ))
                top = int(detected[0, 0,row, 4] * height )
                right = int(detected[0, 0, row, 5] * width )
                bottom = int(detected[0, 0, row, 6] * height )

                faceBoxes.append([left, top, right, bottom])
                cv2.rectangle(temp_image, (left, top), (right, bottom), (0, 255, 0), int(round(height / 150)), 8)

        return temp_image, faceBoxes

#svm
class Gender():
    """docstring for Gender"""
    def __init__(self):
        self.samples_path = "models/dataset/colored/jpg2/"
        self.samples_cnt = 0
        self.sample_size = 140

        self.svm_model = "models/gender/gender_svm.pkl"

    def create_svm_classifier(self):
        self.samples_cnt = len(os.listdir(self.samples_path))
        # Delete the previous svm_model
        if os.path.exists(self.svm_model):
            os.remove(self.svm_model)
        # Iterates for training
        # The number of iterates affect the accuracy of SVM model.
        iterates = 1 
        while iterates > 0:
            whole_data = np.zeros((self.samples_cnt, self.sample_size * self.sample_size))

            sample_label = []
            count = 0
            for sample_name in os.listdir(self.samples_path):
                #print(sample_name[7])
                sample_label.append(sample_name[7]) # Gender: Male or Female
                
                sample_path = self.samples_path + sample_name
                sample = cv2.imread(sample_path, 0)
                hist = array(exposure.equalize_hist(sample)) # Equalize historgram for samples
                sample_data = np.asarray(hist).reshape(-1) 

                whole_data[count, :] = sample_data
                count = count + 1

            # Samples split into trainset and testset
            traineddata, testdata, trainingLabel, testLabel = train_test_split(whole_data, sample_label, test_size = 0.2, random_state = 30)
           
            PCA_cnt = int(self.samples_cnt / 10) # Principal Component Analysis - Numbers
            train_PCA = PCA(n_components = PCA_cnt)
            train_PCA.fit(traineddata)

            test_PCA = PCA(n_components = PCA_cnt)
            test_PCA.fit(testdata)

            # Create SVM Classifier
            svm_classifier = svm.SVC(kernel = 'poly', C = 1.0, gamma = 0.10000000000000001)
            trainLabels = np.array(trainingLabel[:])
            testLabels = np.array(testLabel[:])
            model = svm_classifier.fit(traineddata, trainLabels)
            # Save SVM Classifier
            # self.svm_model = pickle.dump(svm_classifier)
            
            dump(svm_classifier, self.svm_model)

            # Test the Classifier & Show the result
            #print("Step ", iterates)
            test_classifier = svm_classifier.predict(testdata)
            T_F_N_P = confusion_matrix(testLabels, test_classifier)
            print("True & False - Negative & Positive:\n", T_F_N_P)
            accuracy = accuracy_score(testLabels, test_classifier)
            print("Accuracy is ", accuracy)
            report = classification_report(testLabels, test_classifier)
            print("Model Report is - \n", report)

            iterates = iterates - 1

    def load_svm_classifier(self):
        return load(self.svm_model)

    def predict_gender(self, face):
        """if not os.path.exists(self.svm_model):
            self.create_svm_classifier()"""
        
        self.create_svm_classifier()
        
                    
        gender_svm = self.load_svm_classifier()

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (self.sample_size, self.sample_size))
        validate_face = array(exposure.equalize_hist(face))
        remap_data = np.asarray(validate_face).reshape(-1)

        validate_data = np.zeros((1, self.sample_size * self.sample_size))
        validate_data[0, :] = remap_data

        validation = gender_svm.predict(validate_data)
        
        if validation[0] == 'f':
            return "Female"
        else:
            return "Male"

# caffe model
"""docstring for Gender"""
class AgeRecognition():
    def __init__(self):
        self.ageProto="models/age_deploy.prototxt"
        self.ageModel="models/age_net.caffemodel"
        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)

        self.age_category = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def predict_age(self, face):
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), self.MODEL_MEAN_VALUES, swapRB=False)
        self.ageNet.setInput(blob)
        guess_age = self.ageNet.forward()
        
        age = self.age_category[guess_age[0].argmax()]
        print("Age:{},confidence={:.3f}".format(age, guess_age[0].max()))

        return self.age_category[guess_age[0].argmax()]


    
def final(frame):
    face = FaceRecognition()
    gender = Gender()
    age = AgeRecognition()
        
    #frame = cv2.imread(image)

    result_face, faceBoxes = face.detect_face(frame)
    padding = 20

    if not faceBoxes:
        return(result_face,"No Face detected","-")
    else:
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1] - padding):
                    min(faceBox[3] + padding, frame.shape[0] - 1), max(0,faceBox[0] - padding)
                    :min(faceBox[2] + padding, frame.shape[1] - 1)]

            M_or_F = gender.predict_gender(face)
            pre_age = age.predict_age(face)
    
    return(result_face,M_or_F,pre_age)
             
               
iface = gr.Interface(final,gr.inputs.Image(source="upload", tool="editor",type="numpy"),[gr.outputs.Image(type="auto",label="Face Detected"),gr.outputs.Textbox(type="auto",label="Gender"),gr.outputs.Textbox(type="auto",label="Age")],interpretation="default",title="Age And Gender Prediction ",)                        
 

if __name__ == '__main__':
    iface.launch(share=True)
    """parser = argparse.ArgumentParser(description='Gender and Age Classification-To exit, press\'q\'')
    parser.add_argument('-i', help='Insert a specific image file(*.jpg) path please')
    args = parser.parse_args()
    print("args=",args)"""
    
           
                