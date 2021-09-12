import numpy as np
import pandas as pd
import random

"""
KNN

1.  数据集标准化       X_normalization = (df - Df_min)/(Df_max- Df_min)     
        对dataframe里面每个数都要做这步，so参数为df, 返回也是df形式
2. 求距离， 用test的一条数据(array形式) 减去 train_df (把标签剔除)    formular: [ (x1-y2)**2 + (x2 -y2)**2------] 再开根号 也就是*0.5
3. 距离按小到大排列， 去前K个
4. value_counts 一下前K个的标签， 取value_counts.index[0], 就是概率最大的那个
5. 把概率最大那个，作为预测类别

6. 预测结果.append to list, then, add to test_df
7. 看原标签列和预测列是否一样， 一样的为True， 否则False ，也弄一列correction_check
8. calculate_accuracy: correction_check.mean() 
"""


# prepare data
"""
Dataset 1: Binary classification with breast_cancle Dataset

#path = "D:/Data/Dataset/breast_cancle.csv"
#df = pd.read_csv(path,)
#df["label"] = df.diagnosis
#df = df.drop(["id", "diagnosis"], axis = 1)
"""

#Dataset 2: Multiclass classification with iris Dataset
path = "D:/Data/Dataset/iris.csv"
df = pd.read_csv(path)
df = df.drop("Id", axis =1)
df = df.rename(columns={"species": "Label"})

print(df.head())
print(df.info())


def train_test_split(df, test_size):
        if isinstance(test_size, float):
            test_size = round(test_size * len(df)) # Round function to take only Int

        data_index = df.index.tolist()
        test_index = random.sample(population = data_index, k = test_size)

        test_df = df.loc[test_index]
        train_df = df.drop(test_index)
        #print("train\n{}\n test\n{}".format(train_df,  test_df))

        return train_df, test_df

#normalization
def normalization(df):
        label = df.iloc[:, -1]
        df = df.iloc[:, :-1]
        min_df = df.min()
        max_df = df.max()
        normed_df = (df - min_df) / (max_df - min_df)
        normed_df["label"] = label
        return normed_df


def calculate_distance(train, one_test):  # 一次只算一条test_data
        train_label = train.iloc[:, -1]
        train = train.iloc[:, :-1]
  
        one_test = one_test.values                                                  #  运算时是用array 减去 dataframe
        distance = ((one_test - train) ** 2).sum(axis = 1) * 0.5            #axis=1 =columns, 才是按行求和 #是列上相加
        #print("distance{}".format(distance))

        distance_df = pd.DataFrame(data = distance, columns = ["distance"])
        distance_df["label"] = train_label
        #print("distance_df:\n", distance_df)

        return distance_df

def classify(distance_df, k):
       sorted_distance = distance_df.sort_values(by ="distance")[: k]
       label_value = sorted_distance.iloc[:, -1]. value_counts()  # 标签列统计
       majority_label = label_value.index[0]
       return majority_label

def Knn_algorithm(df, k):
    normed_df = normalization(df)                                               #標準化
    train, test = train_test_split(normed_df , test_size= 0.1)        #train, test, split

    prediction_list = []
    for i in range(len(test)):                                                            # 对这个测试集:
          distance = calculate_distance(train, test.iloc[i, :-1])                      # 计算距离: array - df
          prediction = classify(distance, k)                                                    #  做预测
          prediction_list.append(prediction)

    test["prediction"] = prediction_list                                   # 结果确认
    correction_check = test["prediction"] == test["label"]
    #correction_check = test.iloc[:, -2] == test.iloc[:, -1]
    test["correction_check"] = correction_check 
    accuracy = correction_check.mean()

    #global  None
    if accuracy==1.0:
        error = None
    else:
        label_True = test.iloc[:, -1].value_counts().index[0]      # true
        True_index = test[test["correction_check"] == label_True].index    # 把预测正确的index给出来， 
        error = test.drop(index = True_index, axis = 0)                                #再把那些行给drop掉， 返回分类错的dataframe

    return accuracy, error, test



accuracy_list = []
column_names = df.columns
error_df = pd.DataFrame(columns = column_names, index = [0])
for i in range(10):
        accuracy, error, test = Knn_algorithm(df, k = 5)
        accuracy_list.append(accuracy)

        error_df = pd.concat([error_df, error], axis = 0)  
        print("\nfor loop {}, test is:\n{}, \nerror:\n{}".format(i, test, error))
        print("accuracy:{}".format(accuracy))
        
        
error_df = error_df.drop(error_df.index[0], axis = 0)
error_counts = (error_df.index).value_counts()
average_accuracy = np.array(accuracy_list).mean()

print("\naccuracy_list:\n{}".format(accuracy_list, ))
print("\naverage_accuracy:{}".format(average_accuracy))
print("\nerror_df:\n{}\n".format(error_df))
print("\nerror_counts:\n{}\n".format(error_counts))





