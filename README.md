#ALL MAIN CONTENTS ARE in main.py

#This feature engineering project is a module that consists of taking a raw dataset on 35 different stocks from the years of 1995-2022 and how they have changed over the years. To condense data for the purpose of creating a training & testing set, I have chosen to seperate it through the most recent timeframe, being the year of 2022. This project was created simply to demonstrate skillsets of data science principles of XGBoost, transforming pipelines, fitting a pipeline, data cleaning, variable permutation, & creating variable graphs to visualize results

#Separating the training & testing sets within the year of 2022 by the apogee of the Earth & the Sun (middle of the year) is utilizing a timestap, which happens to be a very convient way to separate data because it gives a great approximation for future data(testing/validation set) based on past data(training set). Therefore, by breaking the validation data by the second half & the training set by the previous test, we can fit a tangible model of data that will produce results to help us come to conclusions on how variables relate to one another.

#
