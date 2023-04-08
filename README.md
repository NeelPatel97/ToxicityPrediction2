# ToxicityPredictionChallenge II

##### With new chemicals being synthesized every day, toxicity prediction of newly synthesized chemicals is mandatory before they could be released in the market. For a long time, *in-vivo* methods have been used for toxicity prediction which involves studying bacteria, human cells, or animals. With thousands of new chemicals being synthesized every day, it is not feasible to detect toxicity with traditional laboratory animal testing. One great alternative to *in-vivo* methods is the *in-silico* techniques that have great potential to reduce the time, cost, and animal testing involved in detecting toxicity. This dataset is prepared using a resource that has data for approximately 9,000 chemicals with more than 1500 high-throughput assay endpoints that cover a range of high-level cell responses.

##### This repository contains code for predicting drug activity using various machine learning models. The dataset used contains SMILES strings of molecules and their respective activity values. The models make use of RDKit descriptors, Morgan fingerprints, and circular fingerprints to predict the activity of a molecule. Additionally, data is preprocessed and sanitized with datamol, and feature selection techniques are used to improve model performance.


# System setup used and Requirement
Programming-Language: **Python 3.9.13** 
Operating-System: Windows 
	
###### Python package requirements:
> ###### (Packages mentioned below can be installed using "pip", for e.g., "pip install pandas")


###To reproduce the results, follow these steps:

####1. Install dependencies
Ensure you have correct Python version installed. Install the following packages using pip:

Copy code
pip install numpy
pip install pandas
pip install scikit-learn
pip install tqdm
pip install rdkit-pypi
pip install simpleimputer  # Note: This package is already included in scikit-learn, so you might not need to install it separately.
pip install mlxtend
pip install xgboost
pip install lightgbm


####2. Prepare the dataset
Download the train and test datasets as train_II.csv and test_II.csv respectively, and place them in the root folder of this project.
Make sure that the dataset files have the correct format, with a header line and comma-separated values. 


####3. Run the code
Run the provided code in a Python environment. This will generate features using RDKit descriptors, Morgan fingerprints, and circular fingerprints, preprocess the raw data, and create train and test datasets. It will also perform feature selection and train various machine learning models.

bash
Copy code
python drug_activity_prediction.py
The script will generate several output files:

train_data_features1.csv and test_data_features1.csv containing the generated features.
train_data_datamol.csv and test_data_datamol.csv containing the preprocessed data with datamol.
train_data_final&Scaled.csv and test_data_final&Scaled.csv containing the final and normalized datasets.
####4. Evaluate the models
At the end of the script, the performance of the trained models will be displayed.

## Please note: To save time one can use the train_data_final&Scaled.csv and test_data_final&Scaled.csv to get dataset with all features and scaling already completed
#### Google drive link to the train dataset: 
https://drive.google.com/file/d/1GXevlHqR_Z7-JUr439dyBwc3uxc8EKHF/view?usp=share_link

#### Google drive to test dataset: 
https://drive.google.com/file/d/1PS-_2SqtaCkzPKEwJYCMcdRMMB3EwWte/view?usp=share_link

## Please note: To save time one can use following features generated after the feature selection process: 

selected_features = ['bit_114',
 'bit_717',
 'assay_id',
 'qed',
 'MolWt',
 'HeavyAtomMolWt',
 'ExactMolWt',
 'MaxAbsPartialCharge',
 'MinAbsPartialCharge',
 'BCUT2D_MWLOW',
 'BCUT2D_CHGLO',
 'BCUT2D_LOGPLOW',
 'Chi0n',
 'Chi1v',
 'Chi2v',
 'Chi3v',
 'Kappa1',
 'Kappa2',
 'LabuteASA',
 'SMR_VSA1',
 'SlogP_VSA5',
 'VSA_EState5',
 'VSA_EState6',
 'MolLogP',
 'MolMR',
 'fr_nitro_arom',
 'fr_Al_COO',
 'fr_COO2',
 'fr_NH2',
 'Col_516','n_lipinski_hbd', 'tpsa', 'n_aliphatic_carbocycles']
 

##Additional Information
###The code in this repository is structured as follows:

####1. Feature generation with RDKit, Morgan fingerprints, and circular fingerprints.
####2. Preprocessing and sanitizing the raw data with datamol.
####3. Merging and normalizing the generated features.
####4. Applying feature selection techniques (SelectKBest and Sequential Feature Selection).
####5. Training various machine learning models, including RandomForestClassifier, DecisionTreeClassifier, XGBoost, LightGBM, GradientBoostingClassifier, and MLPClassifier.
####6. Evaluating the performance of the models using various metrics, such as accuracy, confusion matrix, classification report, and F1 score.   

###Link for the Online competition:
https://www.kaggle.com/competitions/the-toxicity-prediction-challenge-ii