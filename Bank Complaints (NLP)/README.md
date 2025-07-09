Data Description

-- Complaint Number: Unique identifier for each customer complaint.
-- Product: The financial product associated with the complaint, categorized into five classes: 
credit reporting, debt collection, mortgages and loans, credit cards, and retail banking.
-- Narrative: The textual narrative of the customer's dispute or complaint submitted to the Consumer
Financial Protection Bureau (CFPB).

Objective:
The objective of this  project is to develop an NLP model for bank customer complaint analysis
to classify complaints into predefined categories based on their textual narratives. By automating the
classification process, the project aims to improve efficiency in dispute resolution and enhance customer
satisfaction by ensuring timely and accurate handling of complaints.

* Dropped Complaint Number cuz they are all unique
* There were 10 null values and deleted it
* There were 37735 dupilcates and removed it all
* There are 5 unique values in product and its bit imbalanced
* Used lammatized with proper pos tag
* used label encoder for the target variable
* find the right maximum features using logistic reggrssion and its 5000
* Used TD-IDF vecotrizer for text preprocessing
* handled imbalanced data using over sampling by smote
* Trained 5 models and find out random forest work well with 83 %
* Tuned random forst uisng random cv search and get precision and recall improved
* for better model this time i used word2vec instead of TFIDF
* used Tokenizer for converting words into numbers
* combined the numbers in list using sequences
* set all sentance same length using 
* created a dl model BiLSTM
* created early stop and tranined model with 15 epochs
* improved accuracy and recall
* save the model using pickle
* created streamlit app and created postgres data base for complaint to register
* and conncet the database with stremlit , so when user entering complaint it automatically cataegorizing based on model prediction
