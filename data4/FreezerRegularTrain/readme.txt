,0
0,# Electrical Load Measurement data sets
1,"The collection of these data was part of the project titled Personalised Retrofit Decision Support Tools for UK Homes using Smart Home Technology (REFIT). The REFIT data set includes data from 20 households from Loughborough area over the period 2013-2014 (see [1], [2])."
2,"We use data of freezers in House 1 to make two data sets, *FreezerRegularTrain* and *FreezerSmallTrain*. As the names suggest, these two data sets share a same test set and only differ in the number of training instances. "
3,"There are two classes, one representing the power demand of the fridge freezer in the kitchen, the other representing the power demand of the (less frequently used) freezer in the garage. They are hard to tell apart globally but they differ locally. "
4,## FreezerRegularTrain
5,Train size: 150
6,Test size: 2850
7,Missing value: No
8,Number of classses: 2
9,Time series length: 301
10,## FreezerSmallTrain
11,Train size: 28
12,Test size: 2850
13,Missing value: No
14,Number of classses: 2
15,Time series length: 301
16,We use data of House 20 to make one data set. There are two classes. The first class is household aggregate usage of electricity. The second class is aggregate electricity load of tumble dryer and washing machine.
17,## HouseTwenty
18,Train size: 40
19,Test size: 119
20,Missing value: No
21,Number of classses: 2
22,Time series length: 2000
23,There is nothing to infer from the order of examples in the train and test set.
24,"Data created by David Murray et al. (see [2]). Data edited by Shaghayegh Gharghabi, Hoang Anh Dau and Eamonn Keogh."
25,[1] http://www.refitsmarthomes.org/
26,"[2] Murray, David et al., ""A data management platform for personalised real-time energy"
27,"feedback"", Proceedings of the 8th International Conference on Engery Efficiency in Domestic Appliances and Lighting, 2015."
