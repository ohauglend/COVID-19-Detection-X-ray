# COVID-19-Detection-X-ray
CNN model for classifying instances as either healthy, COVID-19, lung opacity, tuberculosis, or viral pneumonia.

The following code is created for an exam submission in a deep learning course at Copenhagen Business School.

There has been a lot of research examining chest X-rays to classify various pulmonary diseases,
many of which revolve around a three-class classification task, usually COVID-19, healthy and
a form of pneumonia that is not COVID-19-related. This paper introduces a multi-label
classifier that distinguishes between healthy instances, COVID-19, lung opacity, viral
pneumonia, and tuberculosis infected instances. The model uses ResNet50V2 and
Xception as bases, model inspiration is drawn from Rahimzadeh and Attar
(2020). 

![alt text](https://user-images.githubusercontent.com/64472833/175811616-f4ca2487-b801-4ae2-bfac-9f5298e2b832.png)


We find that the model is able to classify all five classes with a competitive accuracy of 95%, recall of 94.9% and precision
of 95.4%. 
![alt text](https://user-images.githubusercontent.com/64472833/175811756-20aa5587-ad27-4fb2-bf20-1634971e3000.png)


The paper emphasises the linsk to the underlying biologi and shows that the model picks up on common radilogical traits for all classes.
This is done by using Grad-CAM on isntances where the model was a 100% sure of its prediction.
![alt text](https://user-images.githubusercontent.com/64472833/175811948-c5865bff-90e9-480d-9620-18ef592b1c1d.png)






