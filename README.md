# COVID-19-Detection-X-ray
CNN model for classifying instances as either healthy, COVID-19, lung opacity, tuberculosis, or viral pneumonia.

The following code is created for an exam submission in a deep learning course at Copenhagen Business School.

There has been a lot of research examining chest X-rays to classify various pulmonary diseases,
many of which revolve around a three-class classification task, usually COVID-19, healthy and
a form of pneumonia that is not COVID-19-related. This paper introduces a multi-label
classifier that distinguishes between healthy instances, COVID-19, lung opacity, viral
pneumonia, and tuberculosis infected instances. The model uses ResNet50V2 and
Xception as bases, model inspiration is drawn from Rahimzadeh and Attar (2020). 
![alt text](https://user-images.githubusercontent.com/64472833/175811616-f4ca2487-b801-4ae2-bfac-9f5298e2b832.png)


We trained the model for 150 epochs, and over this period the loss graphs show a steady decline.
They do not flatten out, indicating that that a lower minimum could be found if the model
would be trained for more epochs. There are some spikes in the validation curve, which might
be explained by the fact that we have a batch size of 20. A smaller batch size increases the
variability in the validation curve since smaller batches are less likely to contain enough samples
to represent the whole population (Radiuk, 2017). Except for these spikes, the validation curve
follows the evolution of the training curve closely.
![alt text](https://user-images.githubusercontent.com/64472833/175812140-ee670e9c-9270-4cb2-96b3-fc7ffc3b39b5.png)


In order to show the performance of our model this section presents the accuracy, precision, recall
and F1 score of our model’s predictions. The first performance indicator used is accuracy. 
Accuracy is a commonly used metric for measuring model performance and is calculated as follows:
![alt text](https://user-images.githubusercontent.com/64472833/175812191-4480b71e-2bd8-47ae-ad85-0798315525b8.png)
Our model achieved an overall accuracy of 95% on the test set.


Since the overall accuracy does not take the predictions distribution among classes into account,
we have presented a confusion matrix (Géron, 2019). Table 3 shows the model’s accuracies for
all of the five classes.
![alt text](https://user-images.githubusercontent.com/64472833/175812341-23545ef2-ee90-4b99-9a34-1c9b8d319c64.png)
For the test set, the model reached accuracies of 97% for tuberculosis, 96% for viral pneumonia, 96% for COVID-19, 94% for lung opacity, and 93% for the normal cases. The biggest discrepancies were between lung opacity and normal, where in 6% of the instances when the one
class was the true label, the model predicted the other. This might be due to the fact that lung
opacity can have one of several causes, and therefore not be as clearly definable as the other
classes.


In order to provide more concise metrics, the precision and recall of the model’s predictions
are presented (Géron, 2019). The former shows the accuracy of the positive predictions and the
latter the portion of the positive instances correctly identified by the model.
![alt text](https://user-images.githubusercontent.com/64472833/175812258-dbf94e62-1870-4c66-bb48-64f82330db46.png)
On the test set, our model achieved an overall precision and recall of 94.9% and 95.4% respectively.
Combining these two metrics yield the Fβ-score, where the β parameter indicates the weight
of importance between precision and recall (Chinchor and Sundheim, 1993). It is impossible 
to increase both simultaneously, this is also known as the precision-recall trade-off. The most
widely used Fβ-score is the F1-score, this is also the one selected by this paper as we wanted
to allow for our model to be easily compared with others. The F1-score ables more suitable
inference to be drawn from the model’s predictions, compared to the overall accuracy. There are
different variations of the F1-score, and the F1-macro is chosen in this paper since we wanted to
attribute equal importance to the five classes. The macro-averaged F1 does this by computing
the F1-score for each classes separately before aggregating them. The following formula presents
the macro Fβ-score:
![alt text](https://user-images.githubusercontent.com/64472833/175812297-281dcfd5-fb4f-4905-b9a9-0083d2d991c9.png)
Our model achieved an F1-score of 95.1% on the test set.
Table 4 presents the above-mentioned performance indicators of the model at class- and overall
level.
![alt text](https://user-images.githubusercontent.com/64472833/175812326-899d8414-7c5a-44aa-89e4-4df4802dce5f.png)
The precision was the highest for the Viral pneumonia class (98.9%) followed by the Tuberculosis class (98.2%). The recall was highest for the Tuberculosis class (97.4%) and then for the COVID-19 class (96.4%). The model had the highest F1-score for the Tuberculosis class, at
97.8%. It is worth noting that the model had a higher recall than precision for the COVID-19
class. This is desirable since a high recall is important when dealing with transmissible diseases.
Building on this point a future exploration could be to have the model emphasise even more on
recall for the COVID-19 class.


In order to profoundly understand the way our model allocated predicted classes we decided
to utilize Gradient-weighted Class Activation Mapping (Grad-CAM) (Selvaraju et al., 2017).
Grad-CAM is an algorithm developed to more thoroughly explain CNN-based models. It utilizes
gradients of the target class that are getting passed to the final convolutional layer in order to
model a rough heatmap indicating the most influential and important regions that affected the
final prediction (Selvaraju et al., 2017).

<img align="center" img width="300" height="700" alt="image" src="https://user-images.githubusercontent.com/64472833/175813042-d72cda1e-288a-49bb-a1d8-b0164ad8c736.png">

Figure 6 presents the heatmaps for all the classes. The instances visualized in the figure are examples when our model was 100% convinced that
this image belongs to corresponding class. Although these are individual cases, general tendencies for all analyzed medical conditions can be seen.
For instance, the model is able to not indicate any odd structures and patterns in the Normal case. The COVID-19 example shows signs of bilateral presence, which is proved to be a very common radiological sign (Chamorro et al., 2021). Tuberculosis is often diagnosed by the presence
of well-defined smaller lumps, which can also be noted in the Grad-CAM example. These instances are indicators that our model is capable of extracting common and discriminating features of all medical conditions in question.


In an attempt to analyze the prediction errors the model made, we decided to examine 10
randomly selected mistaken instances of each class. By doing that we were able to trail back
the lapses in our model and explain the source of the errors.
![alt text](https://user-images.githubusercontent.com/64472833/175812822-54bf620f-61e5-4514-a967-d043279ac3d6.png)
The instances above are representatives of the most common faults observed among the
wrongly classified cases. Pink coloring haunted mainly radiographs of patients with COVID-19.
The abnormal characteristic probably confused the model, making it unable to indicate COVID19 patterns. Medical devices used to monitor the patients condition, such as electrocardiogram, chest leads, and electrodes can be observed on multiple scans. Their presence could draw the
model’s attention, therefore leading to false predictions. One of the most common scans’ lapses
was lack of lung’s contour and visibility due to the exaggerated brightness of the performed
scans. Such scans make it virtually impossible to detect patterns of any medical condition in
question. The same applies for scans that happened to be overly cropped. Removal of faulty
and low-quality radiographs could have been applied in order to ameliorate the performance of
the model. However, as Abhinaya and Arvind (2021) indicated, the task of assessing the quality
of an X-ray is problematic. Therefore, we decided to not take any actions regarding filtering
potentially faulty instances.


The dataset used for the model represents a combination of other datasets collected
from various sources (Basu et al., 2021). Since demographics such as gender, age and race were
not provided along with the data, this introduces a bias to our model. The likelihood is that
the white European/American male is over-represented in our dataset (Roberts et al., 2021).
This hampers the generalizability of the model since it is likely to perform worse on subjects not
hailing from this group. Further, since this is an external dataset we were not able to verify the
ground truth of the data. A point that is further exaggerated by the fact that this is a collection
of external datasets.


This paper has proposed a model for a multi-label classification of X-rays. Most papers on the
topic distinguishes between three classes, mainly COVID-19, healthy and a pneumonia variant
that is non-COVID. Our model classifies instances as either normal, COVID-19, lung opacity,
viral pneumonia, and tuberculosis. To the best of our knowledge this is the only model that
makes this distinction, and it is therefore, a valuable contribution to the field of research. The
model introduced builds on the features extracted from ResNet50V2 and Xception, with a
convolutional block and three identity blocks added on top. The model presented had an overall
test accuracy of 95%. It struggled the most distinguishing between lung opacity- and normal
instances, and had a relative tendency to classify these as each other. Examining the errors
of the model it is clear that poor quality of individual images affected the models prediction.
The poor quality was either in the form of pink colouring or medical devices being present
in the X-ray. This study did not filter out images of poor quality and this is something that
would likely improve the performance of the model. Images affected by high brightness and
cropped images were the other notable errors we found in the predictions. This points to the
fact that in augmenting the data we could have been less conservative with the magnitude of
these techniques. It also shows that there is room for fine-tuning the model. Finally, it is worth
noting that the generalizability of the model is limited by the fact that the dataset used does
not include demographics of the subjects and that we are not able to verify the ground truth of
the data.


Future work:
This paper has followed all the mandatory criteria from the CLAIM checklist (Roberts et al.,
2021), however there are a few non-mandatory criteria we have not complied with. Most notable
of which are the absence of an external test set to test the generalizability of our model. This
along with filtering out bad quality X-rays, and tuning the model is something we leave for future
work. Recommended tuning include assigning more importance to the recall of the COVID-19
class and adding more aggressive augmentation, in particular with regards to the brightness and
transformation. We also encourage others to extend the scope of this paper by including more
diseases.






