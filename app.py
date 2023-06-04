import pandas as pd
url = "https://github.com/rahulkumarmishra1011/NLP_fake_original_checking/raw/main/fake_papers_train_part_public.csv"
train_df = pd.read_csv(url)
url = "https://github.com/rahulkumarmishra1011/NLP_fake_original_checking/raw/main/fake_papers_test_public_extended.csv"
test_df = pd.read_csv(url)

# Download helper functions script
#!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
# Import series of helper functions for the notebook
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys

train_sentences, train_labels, test_sentences, test_labels = train_df["text"], train_df["fake"], test_df["text"], test_df["fake"] 
# Check the lengths
len(train_sentences), len(train_labels), len(test_sentences), len(test_labels)

# Find average number of tokens (words) in training Tweets
round(sum([len(i.split()) for i in test_sentences])/len(test_sentences))


import tensorflow as tf
from tensorflow.keras.layers import TextVectorization # after TensorFlow 2.6

# Setup text vectorization with custom variables
max_vocab_length = 10000 # max number of words to have in our vocabulary
max_length = 141 # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_sentences)

sample = train_sentences[0]

# Create sample sentence and tokenize it
sample_sentence = sample
text_vectorizer([sample_sentence])

tf.random.set_seed(42)
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=128, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             input_length=max_length, # how long is each input
                             name="embedding_1") 

embedding

# Get a random sentence from training set
import random
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nEmbedded version:")

# Embed the random sentence (turn it into numerical representation)
sample_embed = embedding(text_vectorizer([random_sentence]))
sample_embed

# Build model with the Functional API
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings
x = text_vectorizer(inputs) # turn the input text into numbers
x = embedding(x) # create an embedding of the numerized numbers
x = layers.GlobalAveragePooling1D()(x) # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
outputs = layers.Dense(1, activation="sigmoid")(x) # create the output layer, want binary outputs so use sigmoid activation
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense") # construct the model

# Compile model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# Fit the model
model_1_history = model_1.fit(train_sentences, train_labels,epochs=7, validation_data=(test_sentences, test_labels))

# Check the results
model_1.evaluate(test_sentences, test_labels)

def predict(sentence):
  #print(1)
  val = model_1.predict(pd.Series(sentence))
  rounded_value = val[0][0]
  if rounded_value > 0.5:
    rounded_value = 1
  else:
    rounded_value = 0
  if rounded_value == 1:
    ans = "Fake document"
  else:
    ans = "Original document"
  return ans

print(train_df['text'][20])

example_list = [
    ["SENTENCE 1: Modern two-dimensional imaging is of such quality that echocardiography is now capable of detecting intrapericardial formations. Three morphological types of abnormal intrapericardial echoes have been described: round masses, mattresses and linear echoes. These have been observed in effusions of various origin and seem to be lacking in aetiological specificity. In order to determine more precisely the echocardiographic signs of pericardial metastases, the authors have analyzed 7 cases of intrapericardial masses visualized in a series of 10 patients with metastatic pericardial effusion and examined in two-dimensional mode. These were echogenic and dense masses implanted on the pericardium and subject to cyclic movements linked with those of that membrane. Morphologically, they fell into two categories: - round and sessile masses (6 cases) 8 to 23 mm high and 22 to 48 mm wide at their implantation; they were found mostly opposite the cardiac apex (4 cases) and/or in the lateral wall of the right ventricle (3 cases), - oval formations (2 cases) which were 70 mm long and 17 mm wide in one case and 50 mm long and 15 mm wide in the other. One patient had two masses of different shapes. A review of the literature showed that these two echocardiographic images corresponded to two macroscopic types of pericardial invasion: either tumoral nodules or infiltration plaques betraying a diffuse invasion of the pericardium. All masses observed by the authors were located on the visceral leaflet of the pericardium. This predominantly epicardial location might be due to the visceral leaflet being selectively invaded by retrograde lymphatic embolization from the mediastinal lymph nodes. In cancerous patients such intrapericardial masses strongly suggest that the effusion is metastatic, but these images should be interpreted with caution: fibrin nodules, blood clots or even fatty deposits may produce pseudotumoral images. Owing to be scarcity of cases found in the literature, the frequency of these images in metastatic effusions cannot be determined. In non-cancerous patients the finding of echogenic implanted on the pericardium must lead to invasive exploration of that membrane (surgical or pericardioscopic biopsy) and to a search for primary carcinoma of the bronchi or breast, lymphoma or leukaemia which are the main diseases responsible for malignant pericarditis."],
    ["SENTENCE 2: Background: The optimal sequence of systemic palliative chemotherapy in metastatic breast cancer is unknown Background: The optimal sequence of systemic palliative chemotherapy in metastatic breast cancer is unknown. The objective of this study was to compare the optimal sequence of systemic chemotherapy (in combination with chemotherapy) for the treatment of breast cancer patients with and without recurrence. The study was carried out in the National University Hospital. Among 757 patients with metastatic breast cancer diagnosed from January 1993 to April 2014, a total of 1245 patients were analyzed in terms of their medical records. The primary and recurrence-free survival patterns were as follows: the primary failure was defined as death before the primary diagnosis, while 10% of patients had a second and subsequent recurrence, 5% had recurrence only, and 12% had a relapse after the primary diagnosis. Patients with recurrence were significantly older men, more likely to be women and less likely to have a history of distant metastases, more often than patients without recurrence, had a family history of breast cancer and had more frequent use of hormonal contraceptives. There were no differences between the primary failure and recurrence groups on the distribution of time between the primary and the second and subsequent recurrence. After a median follow up of 10 months, there were no significant differences observed in the primary failure after surgery between the patients without recurrence and those who had a recurrence (14.4 vs. 18.4, P = 0.99). A longer follow up duration was a significantly better prognostic factor for recurrence than patients without recurrence. The optimal sequence of systemic chemotherapy for breast cancer patients with and without recurrence is still to be determined."],
    ['SENTENCE 3: This paper discusses the latest developments in the field of road resurfacing and other related topics. In particular, it focuses on the use of a road reflectometer to determine the chemical and physical properties of road surfaces and mineral substances by means of a Radiometer. It also discusses the recent publication of a guide for determining the "illuminating" and "irradiating" properties of pavement surfaces using a road reflectiveometer. All of this is presented in the interest of presenting "additional explanations and explanations of the recent published guidelines for testing the "Illuminating Engineering properties" of road pavement surfaces and minerals using a Road reflectometer.'],
    ['SENTENCE 4: This paper describes a new system called PARTER, which is a "system for rapid and reliable texture recognition". It uses statistical texture classification to discriminate between different regions of an image using sparse information such as the densities of the gray levels and the gradients of the light-gray levels. It works well and has been successfully used in a variety of applications including texture recognition. The paper describes how PARTER works and describes the training procedure used to train the system. The authors state that the system has had many applications and has had good results in achieving "encertior and time performance" in both image recognition and texture classification.'],
    ["SENTENCE 5: In this paper, H.C.M. defines the reading level of the patients at the Hospital deas Cli-Cli-M. de S. Paulo and uses this information to determine the level of literacy of the doctors and nurses at the university and to make recommendations about how best to implement writing of 'informed consent form' into the outpatient patients' medical records. The purpose of this paper is to determine how to best implement the written consent form into the medical records of the university's teaching hospitals in order to make it easier and more cost effective for doctors and patients to communicate and understand the contents of the document."],
]


import gradio as gr

# Create title, description and article strings
title = "Detecting Authenticity of Academic Papers using NLP"
description = "This project aims to develop a Natural Language Processing (NLP) system to determine the authenticity of academic papers. With the increasing prevalence of fraudulent and deceptive scholarly publications, it is crucial to establish methods for identifying fake papers. "


# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=["text"], # what are the inputs?
                    outputs=["text"], # our fn has two outputs, therefore we have two outputs
                    examples=example_list, 
                    title=title,
                    description=description)

# Launch the demo!
demo.launch(inline=False) # generate a publically shareable URL?


