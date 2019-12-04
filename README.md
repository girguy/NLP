# Toxic comment classification using naive bayes and support vector machine classifier

NLP project on classification of toxic comments using different classification methods such as TF-idf,
mutual information and part of speech.

# Data

This project uses the data given by the Toxic Comment Classification Challenge of Kaggle<br/>
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data<br/>
Only the file train.csv is used and is separated in train and test dataset.

# CÉCI

Computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the Fonds de la Recherche Scientifique de Belgique (F.R.S.-FNRS) under Grant No. 2.5020.11 and by the Walloon Region

For this project, the Hercules cluster was used. Some installation has to be made in order to run your code using SLURM(Simple Linux Utility for Resource Management).<br/>
More information on http://www.ceci-hpc.be/faq.html#2.1 .


## Installation

```bash
pip install --user scikit-multilearn
pip install --user scipy
pip install --user  strings
pip install --user -U nltk
pip install -U scikit-learn
python -m nltk.downloader 'all'

```
WARNING : <br/>
- In function of your cluster, you might need to install more libraries.


## Usage

SLURM works with a run.sh file that needs to be personalized for every python. Some example are in the directories.

## Results

They will be put in due time.
