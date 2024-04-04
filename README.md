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


To create the training, validation and test sets, navigate to Part 1 folder, and run:

>> python3 Task_2.py



Feedforward neural network:

This neural network was created using the argparse library, so you can run test_only and train_only, including visualize_model to get the visualizations of the model

to train and test the Feedforward model:
>> python3 main.py

to only train:
>> python3 your_script.py --train_only

to only test on the original test set:
>> python3 main.py --test_only

to test on LIAR set:
>> python3 main.py --test_only --new_data_file data/train_liar_update.csv
