# Master Thesis

This master thesis delves into the exploration of ranking models, with a particular focus
on architectures based on BERT (Bidirectional Encoder Representations from Trans-
formers). The primary objective of the thesis is to train these ranking models using
textual data extracted from systematic reviews. The purpose is to develop a tool that
can assist researchers in conducting systematic reviews by prioritizing the most relevant
studies, thus eliminating the need to screen non-relevant studies in the initial stages.
The study includes deep research on creating a labeled dataset from a collection of sys-
tematic reviews by fetching additional textual content from PubMed. In addition, the
study presents a training framework for training various document ranking models.

# Preprocessing

The dataset needed to train the algorithms is created by running the preprocess.py file. This script
reads in systematic review data from the data directory. These data are not included in the girhub repo
due to access restrictions. If you get hold on the data, make sure to put them in two seperate folders with names
reviews_part_1 and reviews_part_2. Reason for this is that the dataset was initially created with only half of the
systematic reivews. When the script is finished the preprocessed dataset is stored in the data folder. The data
is stored in 3 seperate pickles for making it possible to upload them here to github.

# Training Algorithms

The algorithms is trained by running the train_all.py script. 


