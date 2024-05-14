# Fine-Tuned Vilt Visual Question Answering Model on Nursing Data

## Abstract 

Modern healthcare is predicated on the effective coordination of professionals with diverse skill sets. Simulation-based training (SBT) has emerged as a pivotal element in fostering teamwork among healthcare professionals, particularly nurses. This study leverages a Vision-and-Language (ViLT) based Visual Question Answering (VQA) model,  fine-tuned on human labeled dataset derived from nursing simulation training sessions, to classify actions during nurses' simulation based training. The dataset comprises images labeled across ten classes of nursing actions, including the administration of IV push medication, oral medication, verifying patient vitals such as pulse and heart rate, and maneuvering patient bed positions, among others. Our results demonstrate a significant enhancement in model performance, with the mean average precision (mAP) of action detection increasing from a baseline of 52.7\% to 71.8\% with our fine-tuned ViLT model. This fine-tuning enables the VQA model to identify and categorize key collaborative actions, providing insights into the nuances of nurse collaboration during simulated training exercises. Through this, we aim to contribute to the understanding of nurse collaboration dynamics during training sessions.

## 1 Introduction 

The efficacy of healthcare systems hinges on the collaborative efforts of healthcare providers, where nurses play a fundamental role - effective patient care relies on complex interactions and collaborative practices among nursing staff. The advent of simulation-based training (SBT) has marked a significant stride in the pedagogical development of nursing professionals. SBT allows for the recreation of clinical scenarios where nurses can engage in critical tasks, ranging from the mundane to the lifesaving, within a controlled, risk-free setting.

This paper introduces a refined application of the Vision-and-Language Transformer (ViLT) Visual Question Answering (VQA) model that has been fine-tuned for the nuanced domain of nursing simulations. The ViLT VQA model, known for its efficacy in handling various VQA tasks, is adapted here to interpret and analyze a dataset capturing key nurse-patient interactions. The dataset, labeled with ten classes of nursing actions, forms the basis for training and testing the model. The interactions range from administering various forms of medication to monitoring vital signs, each constituting a critical component of nursing care.

The application of VQA in this context is motivated by the need to provide objective assessments of nursing actions in training simulations. The binary yes/no questions constructed for the model—such as "Is the nurse putting on latex gloves?" or "Is the nurse administering IV push medication?"—are designed to improve the model's prediction accuracy of real-life clinical scenarios. These questions serve as probes to assess the model’s understanding of the visual content, thereby quantifying the presence and accuracy of specific actions.

## 2 Related Work

he integration of Visual Question Answering (VQA)
into healthcare simulation training is an intersection
of multiple disciplines, primarily computer vision,
natural language processing, and healthcare educa-
tion. This section is a comprehensive background
research that entails examining these fields both in
isolation and in conjunction with one another.
 
### 2.1 Computer Vision in Healthcare
The application of computer vision in healthcare is
not novel; it spans from medical imaging analysis,
such as MRI and CT scans, to monitoring patient
activities for rehabilitation. Studies like Esteva et al.
(2019) [3] and Litjens et al. (2017) [7] have demon-
strated the potential of deep learning models in diag-
nosing diseases with precision comparable to that of
human experts. However, the use of computer vision
for interpreting human actions within a healthcare
setting, particularly in SBT, is less explored. Works
by Liu et al. (2019) [11] begin to bridge this gap
by using action recognition algorithms to assess the
performance of healthcare workers in simulated surg-
eries.

#### 2.1.1 Visual Question Answering (VQA)
VQA as a field has seen substantial growth, primar-
ily due to its dual challenge of image understand-
ing and language comprehension. The foundational
paper by Antol et al. (2015) [1] introduced the
VQA dataset, offering a platform to combine both
visual cues and textual questions to produce rele-
vant answers. Successive works have expanded upon
this, utilizing transformer models like ViLT, first pre-
sented by Kim et al. (2021) [5], which eliminated the
need for expensive region proposal networks, simpli-
fying the pipeline for VQA tasks.

### 2.1.2 Simulation-Based Training (SBT) in
Nursing Education
The body of literature on SBT in nursing education,
as reviewed by Cant and Cooper (2017) [2], has con-
sistently highlighted its effectiveness for clinical skill
development. Authors like Ravert (2002) [9] affirm
that SBT provides a realistic, safe, and repeatable
environment conducive to learning. The transition
to assessing teamwork and collaborative skills within
SBT is a natural progression, with papers by Fernan-
dez et al. (2008) [8] emphasizing the importance of
evaluating teamwork competencies in nursing.

#### 2.1.3 Teamwork Assessment in Simulation
Based Training
One common approach to teamwork assessment in
SBT is the use of observational rating scales. These
scales typically involve trained observers who assess
team behaviors and interactions based on predefined
criteria. For example, the Observational Teamwork
Assessment for Surgery (OTAS) developed by Yule et
al. (2006) [10] provides a structured framework for
evaluating teamwork in surgical settings, focusing on
communication, coordination, and leadership.
Another method for teamwork assessment in SBT
is the use of self-assessment tools, where team mem-
bers reflect on their own performance and provide
feedback to their peers. The TeamSTEPPS Team-
work Attitudes Questionnaire (T-TAQ) developed
by King et al. (2008) [6] is an example of a self-
assessment tool designed to measure teamwork atti-
tudes and behaviors among healthcare professionals.
In addition to observational rating scales and self-
assessment tools, objective performance metrics can
also be used to assess teamwork in SBT. These met-
rics may include measures of task completion time,
error rates, and adherence to protocols. For instance,
Goldberg et al. (2021) [4] proposed the GIFT exter-
nal assessment engine for analyzing individual and
team performance during dismounted battle drills,
which integrates objective performance metrics with
subjective evaluations to provide a comprehensive as-
sessment of teamwork effectiveness.

### 2.1.4 Evaluating Teamwork through VQA
The evaluation of teamwork using VQA is a grow-
ing area. While existing literature extensively cov-
ers the evaluation of individual clinical skills through
various metrics, the assessment of collaborative ac-
tions using computational models is emerging. The
confluence of VQA and distributed cognition in the
context of SBT for nurses presents a novel research
avenue, which our work aims to pioneer. By align-
ing the classification capabilities of VQA models with
teamwork assessment, we propose a unique method-
ology for observing, quantifying, and analyzing the
nuances of nurse collaboration during simulations.

## 3 Methods

### 3.1 Set-up and Environment
We established the computational environment tai-
lored specifically for the Vision-and-Language Trans-
former (ViLT). ViLT’s architecture, designed for pro-
cessing both visual and textual inputs, required the
Transformers library, which is equipped with the nec-
essary functionalities for handling multimodal data.

### 3.2 Data Loading
Our study utilized a custom dataset developed and
annotated by our research team, specifically tailored
to the unique requirements of nursing simulation
training. This dataset included images captured from
nurse training sessions, which were manually anno-
tated to reflect critical nursing interactions and ac-
tivities. These annotations were crucial for training
the ViLT model to recognize and understand specific
nursing actions relevant to practical healthcare set-
tings.
The dataset comprises:
• Images: A collection of hand-annotated images
depicting various nursing activities.
• Questions JSON: A JSON file containing
questions specifically designed to query the con-
tent of the images in ways that test the model’s
understanding of nursing tasks.
• Annotations JSON: Another JSON file with
annotations, providing the correct answers to
each question, which serve as ground truth for
model training and validation.
Hand Annotated Training and Testing
Nursing Datasets
The following images present a glimpse into the
dataset utilized for fine-tuning the ViLT model.
These samples depict nurses engaged in various clini-
cal activities, captured from two different camera an-
gles. The images include nurses listening to a pa-
tient’s lungs, taking the patient’s temperature, and
checking the patient’s pulse. Such images are integral
to the dataset, as they not only provide the visual
context for the model’s training but also encompass
the diversity of tasks nurses perform.
The visual component of our dataset plays a piv-
otal role in the fine-tuning process of the Vision-
and-Language Transformer (ViLT) model. Consist-
ing of 295 annotated images, the training set cap-
tures a wide array of clinical activities performed by
nursing staff. These images were derived from high-
fidelity simulated nursing scenarios, designed to repli-
cate real-world clinical settings.
For testing and validation purposes, a separate set
of 99 images was utilized, each accompanied by its
corresponding questions and annotations. The test
set was crucial in assessing the model’s proficiency in
accurately interpreting and answering clinically rele-
vant questions based on visual cues.
Here we showcase examples of the annotated im-
ages:

The annotation framework was intentionally de-
signed to ensure a balanced evaluation of both the
occurrence and absence of specific nursing actions
within video recordings, addressing the critical need
for unbiased data representation in machine learn-
ing models. For each action category, the train-
ing dataset included a near-equal split of affirmative
(’Yes’) and negative (’No’) annotations, with an av-
erage distribution approaching 50/50. This method-
ological rigor was mirrored in the test dataset, which
similarly employed a 50/50 split in annotator re-
sponses. This balanced annotation strategy is essen-
tial to train the model not only to recognize when an
action occurs but also to accurately detect when an
action does not take place, thus preventing any bias
towards either outcome in the model’s predictions.
Such an approach ensures that the model developed
from this dataset can reliably identify and differenti-
ate between actions and non-actions, which is crucial
for practical applications in monitoring and evaluat-
ing nursing care.

### 3.3 Data Processing
Upon loading the data, we proceeded to parse the
questions from the JSON file. For images, a func-
tion was defined to extract the ID from a given file-
name, establishing a correspondence between image
files and their associated IDs. This facilitated the
construction of two dictionaries for filename-ID map-
ping, enabling the retrieval of specific images linked
to particular questions.

Table 2: Summary of Testing Annotations for Nurs-
ing Actions
Action Category Test Yes Test No
Putting on latex gloves 26 10
Administering IV push medication 5 4
Administering IV hang medication 6 5
Raising the head of patient’s bed 2 0
Checking patient’s pulse 10 2
Checking patient’s chest 12 6
Having a phone call 4 6
Preparing iv hang medication 6 10
Putting on inhaler 6 4

### 3.4 Image and Text Processing
Using the ViltProcessor, we processed each image-
text pair, tokenizing the text and normalizing the
images to create a uniform input format suitable for
the model. This included generating inputids, atten-
tionmask, and tokentypeids for the text, as well as
pixelvalues and pixelmask for the images.

### 3.5 Dataset Creation
We constructed a PyTorch dataset to manage our
data efficiently, facilitating the subsequent training
process. The dataset was defined to align with the
PyTorch data handling paradigms, ensuring compat-
ibility and leveraging the functionalities provided by
the ViltProcessor.

### 3.6 Model Definition and Training
We employed the ViltForQuestionAnswering model
pre-trained on the MLM objective, appending a ran-
domly initialized classification head. The model was
fine-tuned end-to-end, optimizing with the AdamW
algorithm over multiple epochs. We applied the bi-
nary cross-entropy loss function, framing VQA as a
multi-label classification task.

### 3.7 Inference
Post-training, we conducted inference to validate the
model’s learning, applying a sigmoid activation func-
tion on the logits to interpret them as probabilities.
The top-k predictions were extracted to assess the
model’s performance qualitatively.

## 4 Results
In evaluating our fine-tuned ViLT model, we mea-
sured performance by assessing its accuracy and
mean Average Precision (mAP) across a set of 99
test questions derived from our simulation training
dataset. The accuracy metric quantifies the propor-
tion of correct predictions out of all predictions made,
while the mAP provides a more comprehensive view
by considering the precision of the model across dif-
ferent classes taking an average of the precision in
each of the classes.

### 4.1 Overall Accuracy
Our accuracy calculation was straightforward: out of
the 99 predictions made by the model, each predic-
tion was compared with the ground truth. A predic-
tion was deemed correct if the top prediction matched
the answer labeled by the human annotators. The ac-
curacy was found to be approximately 63%, while the
baseline model had an accuracy of 53%.

### 4.2 Mean Average Precision (mAP)
Score
To gain further insight into the model’s performance,
we computed the mAP score. The mAP is an evalua-
tion metric used to assess the quality of object detec-
tion models, considering both precision and recall. It
provides an aggregate measure of performance across
multiple classes. Precision in this context is the ratio
of correct positive predictions to the total predicted
positives, and recall is the ratio of correct positive
predictions to the total actual positives. For each
question type, we calculated the average precision,
then computed the mean of these average precision
scores to obtain the mAP. This involved summing the
true positives for each class and dividing by the total
number of predictions for that class, followed by aver-
aging these values across all classes. The mAP score,
computed as the average of the average precisions,
was found 71.8% to be the value that translates to
the model’s consistency in correctly identifying and
classifying actions across various classes, while the
mAP of the baseline model was 52%.
Our results showcase the effectiveness of the fine-
tuned ViLT model in interpreting complex medical
simulations. The accuracy of 63% demonstrates the
model’s robustness in correctly answering the posed
binary questions. Moreover, the mAP score of 71.8%,
reflects the model’s reliable performance across differ-
ent types of nursing actions, ensuring that the model
is not just biased toward more frequent or easier-to-
classify actions.

### 4.3 Different Precision Scores in Each
Category
Action Category AP
Nurse putting on latex gloves 0.59
Nurse raising the head of the patient bed 1.0
Nurse preparing IV hang medication 0.67
Nurse having a phone call 0.57
Nurse putting the inhaler on the patient 0.71
Nurse administering IV hang medication 0.63
Nurse taking the patient’s pulse 0.75
Nurse listening to the patient’s chest 0.77
Nurse administering IV push medication 0.78
Table 3: Performance (in terms of Average Precision) of the model on different action categories.

The performance of the model varied across differ-
ent action categories, as shown in Table 3. Several
factors contribute to this variability.
Firstly, the clarity and visibility of certain actions
significantly influence the model’s ability to accu-
rately classify them. For instance, actions like ”Nurse
having a phone call” exhibited a lower mAP score of
0.57. This reduced performance can be attributed to
the occlusion of the nurse by the monitor, resulting
in only a partial view of the nurse, which hindered
the model’s ability to accurately classify the action.

### 4.4 Loss Values
The success of any machine learning model heavily
depends on its ability to minimize a chosen loss func-
tion during the training or fine-tuning process. In our
study, we monitored the loss values as a key indicator
of the model’s learning progress and convergence.
At the conclusion of the training or fine-tuning
process, the model achieved a loss values of 0.342.
The loss value represent the discrepancy between the
model’s predictions and the ground truth labels in
our training or fine-tuning dataset. A lower loss
value indicates better alignment between the pre-
dicted outputs and the actual targets, signifying im-
proved model performance.
While the majority of the loss values in our train-
ing or fine-tuning process demonstrate a decreas-
ing trend, showcasing a low discrepancy between the
model’s predictions and the ground truth labels in
our training or fine-tuning dataset. Overall, the ob-
served loss values provide valuable insights into the
training or fine-tuning dynamics of our model, guid-
ing future iterations and optimizations aimed at en-
hancing its predictive capabilities.
Overall, our findings affirm the viability of em-
ploying advanced vision-and-language models in the
domain of healthcare education, particularly within
nursing simulation training. The relatively high ac-
curacy and mAP score indicate that such models can
successfully parse visual information and correspond-
ing queries, paving the way for automated, objective
assessments of collaborative nursing actions in edu-
cational simulations.

## 5 Discussion
This study represents a foundational exploration into
fine-tuning the ViLT model for visual question an-
swering in the context of nursing simulation train-
ing. While we achieved promising results, this re-
search opens avenues for future expansions and en-
hancements that could further refine the application
of VQA systems in healthcare training environments.

### 5.1 Expansion of Annotation Cate-
gories
One significant area for future development is the ex-
pansion of the annotation categories used in training
the VQA model. Currently, our annotations encom-
pass ten distinct nursing actions, a scope that, while
effective for preliminary analysis, may benefit from
enlargement. Extending the range of actions to in-
clude more granular or additional relevant behaviors
could improve the model’s utility and accuracy, pro-
viding a richer dataset for more nuanced understand-
ing and evaluation of nurse interactions. This expan-
sion would require a comprehensive review and re-
vision of current training protocols and annotations,
potentially involving domain experts more deeply in
the annotation process to ensure that all relevant ac-
tions are captured and accurately labeled.

### 5.2 Integration with Other Databases
Additionally, there is potential for fine-tuning the
ViLT model using other comprehensive databases
such as HICO-DET, which specializes in human-
object interactions. Incorporating data from such
databases could enhance the model’s ability to recog-
nize and interpret a broader range of interactions and
activities that are common in healthcare settings but
not sufficiently represented in the current dataset.
This approach would help in creating a more robust
model capable of understanding complex, multi-step
interactions that are crucial in high-stakes environ-
ments like healthcare.

### 5.3 Multimodal Analysis of Nurse
Collaboration
Looking ahead, integrating additional modalities
such as eye-tracking, voice analysis, and motion-
tracking could significantly enrich the analysis of
nurse collaboration. Eye-tracking could offer in-
sights into attention and focus, voice analysis could
interpret communication efficiency and clarity, and
motion-tracking could provide data on the physical
coordination of team members. Combining these
technologies with VQA could lead to a holistic view
of performance in nursing simulations, encompassing
both the physical and cognitive aspects of team dy-
namics. Such an integrative approach could redefine
observational training methodologies and provide a
more comprehensive toolset for educators to assess
and enhance teamwork among nursing professionals.

### 5.4 Limitation: Lack of Temporality
in Image Analysis
One notable limitation of our approach lies in the in-
herent constraint of analyzing single frames within
the image data. By focusing solely on individual
frames, our model lacks the ability to consider tempo-
ral dynamics or context within the nursing activities
captured. Unlike video-based approaches that lever-
age sequential frames to infer temporal information
and context, our reliance on isolated frames restricts
the model’s understanding of the progression of ac-
tions over time. The absence of temporal frames in
the training data poses challenges in capturing the
nuanced interactions and transitions between nurs-
ing tasks. As a result, the model may struggle to dis-
cern the temporal flow of activities and may exhibit
limitations in predicting actions that rely heavily on
temporal cues or sequential dependencies.
Addressing this limitation requires a shift towards
incorporating temporal information into the training
process. Future iterations of our research could ex-
plore methodologies to augment the dataset with se-
quential frames or adopt video-based approaches that
enable the model to capture temporal context effec-
tively. By incorporating temporality into the image
analysis, we can enhance the model’s ability to infer
contextual information, leading to improved accuracy
and robustness in predicting nursing actions during
simulation-based training scenarios.

## 6 Conclusion
In conclusion, this study serves as a stepping stone
towards the sophisticated application of visual ques-
tion answering models in healthcare training. By ac-
knowledging the limitations and envisioning potential
expansions, we set the stage for future research that
could transform how nurse training is conducted and
assessed. The integration of advanced machine learn-
ing models with multimodal data analysis stands to
offer unprecedented insights into the complexities of
healthcare teamwork, ultimately leading to improved
training methods and better patient care outcomes.

## References
[1] Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol,
Margaret Mitchell, C. Lawrence Zitnick, Dhruv Ba-
tra, and Devi Parikh. Vqa: Visual question answer-
ing, 2016. 2
[2] Robyn P Cant and Simon J Cooper. Use of
simulation-based learning in undergraduate nurse
education: An umbrella systematic review. Nurse
Education Today, 49:63–71, 2017. 2
[3] A. Esteva, A. Robicquet, B. Ramsundar, V.
Kuleshov, M. DePristo, K. Chou, C. Cui, G. Cor-
rado, S. Thrun, and J. Dean. A guide to deep learn-
ing in healthcare. Nat Med, 25(1):24–29, 2019. Epub
2019 Jan 7. 2
[4] Benjamin S Goldberg, Chaitanya Vatral, Naveed
Mohammed, and Gautam Biswas. Gift external as-
sessment engine for analyzing individual and team
performance for dismounted battle drills. In Proceed-
ings of the 9th Annual Generalized Intelligent Frame-
work for Tutoring User Symposium, pages 109–127,
Orlando, FL: US Army Combat Capabilities Com-
mand Center, 2021. 3
[5] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt:
Vision-and-language transformer without convolu-
tion or region supervision. In Proceedings of the 38th
International Conference on Machine Learning, vol-
ume 139, pages 5583–5594. PMLR, 2021. 2
[6] HB King, James Battles, David P Baker, An-
gel Alonso, Eduardo Salas, James Webster, Larry
Toomey, Margaret Salisbury, Kerm Henriksen, Anne
Johnson, and Daniel Parker. Teamstepps: team
strategies and tools to enhance performance and pa-
tient safety. In Advances in Patient Safety: New Di-
rections and Alternative Approaches, volume 3, Aug
2008. 2
[7] G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F.
Ciompi, M. Ghafoorian, J. A. W. M. van der Laak,
B. van Ginneken, and C. I. S ́anchez. A survey on
deep learning in medical image analysis. Med Image
Anal, 42:60–88, Dec. 2017. Epub 2017 Jul 26. 2
8
[8] Suzanne Polis, Megan Higgs, Vicki Manning, Gayle
Netto, and Ritin Fernandez. Factors contributing to
nursing teamwork in an acute care tertiary hospital.
Collegian, 24(1):19–25, 2017. 2
[9] RN Ravert, Patricia MS. An integrative review of
computer-based simulation in the education process.
CIN: Computers, Informatics, Nursing, 20(5):203–
208, September 2002. 2
[10] Steven Yule, Rhona Flin, Simon Paterson-Brown,
and Nikki Maran. Development of a rating system
for surgeons’ non-technical skills. Medical Education,
40(11):1098–1104, Apr 2006. 2
[11] A. Zia, L. Guo, L. Zhou, et al. Novel evaluation of
surgical activity recognition models using task-based
efficiency metrics. Int J CARS, 14:2155–2163, 2019.
2

