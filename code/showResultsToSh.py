#coding=utf-8

'''
将聚类结果写入各个聚类对应的txt文件中，直观地查看聚类结果好坏
同时，也将聚类结果对应的图像文件复制进相应的聚类文件目录中，更加直观地查看聚类结果好坏
'''

import os
import numpy as np

dir_prefix = "resultdir/"
filelist_prefix = "resultfile/"


def mkresultfilelist(k):
    filelist = []
    for path_num in range(k):
        path = filelist_prefix + str(path_num) + ".txt"
        f=open(path,'w')
        #f.close()
        filelist.append(f)
    return filelist


def writeResultTotxt(filelist, indexlist, imagelist):
    k = len(imagelist)
    print len(filelist)
    for i in range(k-1):
        res = imagelist[i]
        #print indexlist[i]
        matchfile = filelist[indexlist[i]]
        print >> matchfile, res


def load_indexlist(result_file_name):
    #indexlist_file = open(result_file_name)
    indexMatrix = np.loadtxt(result_file_name)
    indexMatrix = indexMatrix.astype(int)
    num = indexMatrix.shape[0]-1
    indexlist = []
    for i in range(num):
        indexlist.append(indexMatrix[i][0])
    return indexlist


def load_image_list(images_list_file_name):
    image_name_list_file = open(images_list_file_name)
    result_list = []
    line = image_name_list_file.readline()
    line=line.strip('\n')
    while line:
        result_list.append(line)
        line = image_name_list_file.readline()
        line=line.strip('\n')
    image_name_list_file.close()
    return result_list


def computeResult(resultPath):
    image_name_list_file = open(resultPath)
    result_list = []
    line = image_name_list_file.readline()
    while line:
        result_list.append(line)
        line = image_name_list_file.readline()
    image_name_list_file.close()

    result_list = sorted(result_list)
    num = len(result_list)
    maxNum = 1
    currentNum = 0
    lastClass = result_list[0]
    lastClass = lastClass.split('/')
    lastClass = lastClass[2]
    for i in range(num-1):
        currentClass = result_list[i]
        currentClass = currentClass.split('/')
        #print currentClass
        currentClass = currentClass[2]
        if lastClass == currentClass:
            currentNum += 1
        else :
            if currentNum > maxNum:
                maxNum = currentNum
            currentNum = 1
            lastClass = currentClass
    return maxNum, num

# imagename_list = load_image_list("caltech.txt")
# indexlist = load_indexlist("clusterAssment.txt")
# filelist = mkresultfilelist(102)
# writeResultTotxt(filelist, indexlist, imagename_list)

#currentNum, num = computeResult("resultfile/0.txt")
def computeRate(path):
    list2_dirs = os.walk(path)
    rightnum = 0
    allnum = 0
    for root, dirs, files in list2_dirs: 
        for d in files:
            maxNum, num = computeResult("resultfile/" + d)
            rightnum += maxNum
            allnum += num
    
    print rightnum, allnum

computeRate("resultfile")
            
