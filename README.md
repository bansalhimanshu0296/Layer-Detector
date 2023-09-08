# Layer-Detector

This project was done as a part of CSCI-B-551 Elements of Artificial Intelligence Coursework under Prof. Dr. David Crandall.

## Problem
Problem involves creating code that tries to find the two boundaries, i.e., air-ice and ice-rock in a given sonar image. We’ll make some assumptions to make this possible.

## Observation:

There were 3 functions given in the skeleton code, i. E., draw_boundary (draws the boundary line in the image depending on the color assigned by us), draw_asterisk (draws asterisk on the 2 points x1y1 and x2y2 in which one is located in the air-ice boundary and the other is located in the ice-rock boundary) and write_output_image (function calls draw_boundary and draws boundaries of different colors for simple, viterbi and viterbi_feedback). 
Image can be read in different pixels with different edge weights where edge weight is how dark a pixel is. 

## Approach and design decisions:

**Assumption:** The air-ice boundary will present in the top 25% part of the image.

A function named compute_simple is defined that computes boundaries through a simple bayesian model. In this function we iterate through each column to find the point of boundaries using a for loop. In the for loop, we find out the edge strength and its total length and convert it to probability values. We find the max edge and probability index. We also find the edge with the max probability which is at least 10 pixels below air-ice boundary and append it to the ice-rock list. This function returns both airice_simple and icerock_simple lists.

We defined a compute_viterbi_split function that calculates boundaries using viterbi. Another function we defined is the compute_viterbi that splits edge strength list to airice list. 

In the compute_viterbi_split function, we calculate the boundaries using viterbi. We find the number of rows and columns of edge strength and also feedback, initialise V_table and which_table, calculate the sum of all edge strength and convert that to probability. Then we append the emission probability to V_table and all the rows to which table. We then iterate through each column of the image in which we again calculate the emission probability and also iterate through each row and calculate the standard deviation to find normal distribution about the row and find the transition probabilities. We initialise the list for backtracking the viterbi table to return the result and find the point row for N-1 column. We again iterate through every column from last and append value point from which_table to list. We return the ice_viterbi value.

In the compute_feedback_split function, we calculate the boundaries using viterbi with feedback. We find the number of rows and columns of edge strength and also feedback, initialise V_table and which_table, calculate the sum of all edge strength and convert that to probability. Then we append the emission probability to V_table and all the rows to which table. We then iterate through each column of the image in which we again calculate the emission probability and also iterate through each row and calculate the standard deviation to find normal distribution about the row and find the transition probabilities. We check if we are in the same column as the feedback column then we set the transition probability of feedback rock as 1 and all others as 0. We initialise the list for backtracking the viterbi table to return the result and find the point row for N-1 column. We again iterate through every column from last and append value point from which_table to list. We return the ice_viterbi value.

In the compute_feedback function, we split edge strength list to article list with the same assumption that airice boundary is in the upper 25% of image. In this function, we find the number of row pixels in the image and call the viterbi function for air_ice boundaries by splitting. We also call the viterbi function for ice_rock boundaries by splitting and converting feedback accordingly. We add the rows which were removed during the split and return both airice_hmm and icerock_hmm boundaries.

## Challenges:

Figuring out the necessary assumptions required to solve this problem was difficult.

