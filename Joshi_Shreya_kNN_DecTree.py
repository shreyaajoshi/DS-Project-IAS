'''
Assignment no :09 kNN & Dtree
@Author: Shreya Joshi
'''
from csv import reader
import csv
import math
from scipy.spatial import distance
import numpy as np
import pandas as pd
from statistics import mode
import heapq
from collections import Counter

def mode_list(array):
    '''
    Calculates the mode of any list
    :param array: The list of which mode is to be calculated
    :return: Mode of the list
    '''
    most = max(list(map(array.count, array)))
    return list(set(filter(lambda x: array.count(x) == most, array)))

def kNN(datalist,target_val):
    '''
    KNN function runs for K iterations and selects the best value of K
    :param datalist: The datalist of which the best K value is to be selected
    :param target_val: The given classifications
    :return: The best K_val, List of indexes to be deleted and the list of target Classes to be deleted
    '''
    misclass=999999         #initialize miscalss to infinite
    k_iterations=25         #run the algorithm for 15 iterations
    final_list_del=[]       #final list of indices to be deleted

    '''
    The following codes calculates the Euclidean distance between each point
    '''
    for current_k_val in range(k_iterations):
        curr_misclass=0
        append_list2=[]
        for index in range(len(datalist)):
            first_pt=np.array(datalist[index])
            euclidean_list1={}                      #dict to store distances
            #print(first_pt)
            for jindex in range(len(datalist)):
                if index!=jindex:                   #calculate distance only if not the same point
                    second_pt=np.array(datalist[jindex])
                    dst = np.sqrt(np.sum((first_pt - second_pt) ** 2))
                    euclidean_list1[jindex]=dst

                    min_val=[]
            min_val.append(heapq.nsmallest(current_k_val+1, euclidean_list1.values())) #get k number of nearest neighbours
            index_min=[]
            for min_check in min_val[0]:
                for keys,values in euclidean_list1.items():
                    if min_check==values :                      #check the indices of the K nearest points
                        index_min.append(keys)

            target_minindex_list=[]
            for mindex in index_min:
                target_minindex_list.append(target_val[mindex])


            final_target= mode_list(target_minindex_list)           #calculate the mode of the K-nearest neighbours
            if len(final_target)>1:
                final_target=1

            if target_val[index]!=final_target:
                    append_list2.append(index)
                    curr_misclass=curr_misclass+1 #if the mode and the target class dont match increase the misclassification count

        print("Misclassifications are :",curr_misclass)

        if misclass>curr_misclass:
            best_k_val=current_k_val                #get the best k value
            misclass=curr_misclass
            final_list_del=append_list2             # get the final list to append

    print('The Best K value is',best_k_val+1)
    return best_k_val,final_list_del


def delete_pts(datalist,final_del_list,target_val):
    '''
    delete_pts deletes all the points which are misclassified at the best k value
    :param datalist: The datalist with all elements
    :param final_del_list: The  List of indexes to be deleted
    :param target_val: The list of target Classes to be deleted
    :return: modified datalist with the target list
    '''

    final_del_list.sort(reverse=True)

    for index in final_del_list:
        datalist.pop(index)
        target_val.pop(index)           #delete the misclassified instances

    return datalist,target_val

def main():
    '''
    main function reads data points and calls kNN
    :return: None
    '''
    #filename = "/home/shreya/Desktop/sem3_1/BDA/hw/HW09_DEC_TREE_TRAINING_data__v720.csv"

    filename=input("Enter the filename")
    df = pd.read_csv(filename, header = 0)
    ifile  = open(filename, "r")
    attr_list = list(reader(ifile))
    data_list=attr_list[1:]       #create the final list for passing
    target_val=df['RedDwarf'].values
    final_data=[]
    for index in range(len(data_list)):
        convert_data=[float(data_list[index][i]) for i in range(len(data_list[index])-1)]
        final_data.append(convert_data)

    target_list=[i for i in target_val]
    best_k,final_del_list=kNN(final_data,target_list)


    #write the final data to a modified csv file used in Dtree Training
    f = open('training.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow( ('Attr1', 'Attr2', 'Attr3', 'Class') )
    for i in range(len(data_list)):
        writer.writerow( (data_list[i][0],data_list[i][1],data_list[i][2],target_val[i]) )

    f.close()

main()