# K-Nearest-Neighbors-Algorithm from Scratch
K-Nearest Neighbors Algorithm (KNN), KNN算法
KNN算法实现，适用于二分类和多分类问题

# Dataset:
for Binary classification: Breast_cancle Dataset

for Multicalss classification: Iris Dataset


# Algorithm steps:
1. Data normalization  
2. Calculate distance between test data and All sample data - Euclidean Distance
3. Sort distances by ascending
4. Takes the first k datas which the distance is small, object is classified by a plurality vote of its K neighbors(K typically small and is an odd)

    (K value usually take sqart(N), N is the feature number of your dataset)
    
6. calculate the accuracy


