# FakeNewsDetector
Group Exam Project For KU 2024 in Data Science

# Collaborators:
Rasmus W. Lyngsøe, Niels Peter Kroon, Gustav Jes Iversen and Mark Wiskum


# How to run the Project
To run the program you first have to make sure you have the correct requirements:

if you use conda to create an environment:

>> conda create -n envFakeNews python=3.12.2

>> conda activate envFakeNews

>> pip install -r requirements.txt

otherwise just:

>> pip install -r requirements.txt

# Cleaning and splitting
To run the code you have to have a file called “995000_rows.csv” in the FAKENEWSDETECTOR folder that contains the 995K FakeNewsCorpus subset file.
You also need a “BBC_data.csv” file that contains the extra reliable data scraped in graded exercise 2. 
Navigate to part 1 folder:

>> python3 Task_1.py

>> python3 Task_2.py

Navigate to part 2 folder:

>> python3 clean_BBC_data.py

>> python3 simple_model_naiveBayes_Liar.py

>> python3 simple_model_logreg_with_BBC.py

>> python3 simple_model_logreg_with_title.py

>> python3 simple_model_logreg.py

>> python3 simple_model_naiveBayes_Liar.py

>> python3 simple_model_naiveBayes.py



# Bi-LSTM model

Ensure that you have train_data.csv, val_data.csv and test_data.csv in the main folder after running part 2.
Navigate to the part 3 folder

>> python3 BidirectionalLSTM_Model.py

# Feedforward neural network:

This neural network was created using the argparse library, so you can run test_only and train_only, including visualize_model to get the visualizations of the model

to train and test the Feedforward model:
>> python3 main.py

to only train:
>> python3 your_script.py --train_only

to only test on the original test set:
>> python3 main.py --test_only

to test on LIAR set:
>> python3 main.py --test_only --visualize_model --new_data_file ../../train_liar_update.csv
