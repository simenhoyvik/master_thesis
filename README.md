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

The algorithms is trained by running the train_all.py script. This script train and evaluate a series of model
where the best one is chosen and evaluated on the test set. The models trained are TF-IDF with logistic regression and
different variants of interaction and representation based BERT models. The script includes a series of hyperparameter
configurations that can be adjusted. When the finish training a model, the training history will be saved in a unique folder
inside a history folder. The selected model is evaluated on the test set where the results are stored in the eval folder.

# Visualizing training performance

The jupyter notebook file print_train_stats.ipynb visualize all training and validation performances that are stored in the history folder.
For each model, it is possible to get an overview of accuracy, loss and MAP score throughout the amount of epochs trained.

# Data Analysis

The jupyter notebook file data_analysis.ipynb visualize different forms of data statistics and analysis of the created dataset.
The script focuses on getting an overview of words and tokens count distribution for different text features.


