# Content-Based Fashion Image Retrieval #

## Group Members ##
* Fabio Cimmino
* Roberto Lotterio
* Gianluca Puleri

## Description ##
We have realized a system which allows the retrieve of the similar images given a query image.
We have implemented 4 methods of features extraction:
* SIFT with BOVW
* Daisy
* Deep Method
* Color

Given the query image:
1. we classify it
2. we extract, with these 4 methods, the features of each image contained into the dataset
3. we compute the similarity between the features of the query image and those of the dataset

The features of each image contained into the dataset are previously computed and then saved.

## Solution design ##
![solution_design](solution_design.PNG)