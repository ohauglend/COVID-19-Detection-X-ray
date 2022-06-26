# COVID-19-Detection-X-ray
CNN model for classifying instances as either healthy, COVID-19, lung opacity, tuberculosis, or viral pneumonia.

The following code is created for an exam submission in a deep learning course at Copenhagen Business School.

There has been a lot of research examining chest X-rays to classify various pulmonary diseases,
many of which revolve around a three-class classification task, usually COVID-19, healthy and
a form of pneumonia that is not COVID-19-related. This paper introduces a multi-label
classifier that distinguishes between healthy instances, COVID-19, lung opacity, viral
pneumonia, and tuberculosis infected instances. The model uses ResNet50V2 and
Xception as bases, model inspiration is drawn from Rahimzadeh and Attar
(2020). We find that it is able to classify all five classes with competitive accuracy of 95%, recall of 94.9% and precision
of 95.4%. We then examine how the model distinguishes the classes using Grad-CAM and
look into the prediction errors the model made. Finally, we comment on the underlying bias of
the data and provide recommendations for future work.

![alt text](https://user-images.githubusercontent.com/64472833/175811616-f4ca2487-b801-4ae2-bfac-9f5298e2b832.png)



