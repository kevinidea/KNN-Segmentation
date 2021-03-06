###  A project submission to the Data Incubator programming challenge

I would like to propose a project to help non-profit organizations such as colleges, universities, or hospitals to find the relevant set of potential donors (public charities and private foundations) to financially support their operations and research projects. 

In order to find a comprehensive list of potential donors, I can download the latest exempt organization business master file extract (EO BMF) that is published by the Internal Revenue Service (IRS). 

I will explore the dataset with some preliminary analysis such as descriptive statistics to get a general sense of the population of these potential donors and their characteristics. Dicing and slicing the dataset in many scenarios and running various pivot tables among many variables to see their relationships can be very helpful here. For numerical variables, at the first pass, I will also do some visualization by plotting various scatter plots and boxplots to see trends and outliners. Next, I will clean the dataset by investigating outliners and anomalies and depending on the situations, I will transform the dataset accordingly to make it ready to feed into a machine learning algorithm in Python. I will once again do some visualization similarly to the previous step mentioned. For an example, please see Figure2 in my submission. 

Once the data is ready, I will move onto analysis step. In this particular context, I will create segmentation by grouping potential donors into various segments or clusters based on their similar characteristics. To make this segmentation process being highly objective, scientific, robust, and reasonable, I will choose an unsupervised machine learning algorithm such as K Nearest Neighbors (KNN) algorithm. For a draft analysis, please see Figure1 in my submission. 

To facilitate the segmentation analysis using KNN algorithm, it is preferred to select numerical features (in statistics field, feature is usually referred as variable). The reason is that the algorithm relies on how close the distance between the data with respect to the selected features to make decisions (grouping various data points into different clusters/segments). In our dataset, there seem to be only three clear numerical features: asset amount, income amount, and revenue amount. Therefore, these 3 features make a very good starting point for applying KNN algorithm. 

I only choose 2 features (asset amount and income amount) because I can represent the segments in a 2 dimensional visualization.  Also, since income and revenue are almost identical for majority of the records, it makes more sense to pick only one and not two. In regard to other features, Even though they might look like numerical, they are really coded in numbers to represent factors as categories. 

The work above is still only the beginning. There are couple more steps I will go from there. 

1.	Features engineering: I will spend more time examining other features including the categorical ones to come up with new features. Example could be redefining the categorical feature into a boolean feature (0/1). Another example would be transforming categorical feature such as USA states into numerical feature such as distance of each states away from a certain location. Domain knowledge, creativity and deep understanding of the underlying information are extremely helpful here. 

2.	Features selection: The focus here is picking a set of features that makes the most sense. This is really an art.

3.	Fine tuning KNN algorithm results: Specifically, the number of clusters to use is a big question here. There is also really no absolutely right or wrong answer since this is an unsupervised learning algorithm where we don’t have labels. Therefore, qualitative analysis has to be done to understand what organization’s viewpoint and need are. 

4.	Try other clustering algorithm to compare the results. 

5.	Validating the results: This has to be done in the real world. Results need to be interpreted and communicated clearly to the organization. A system of monitor and control should be in place to see if the organizations actually find their right potential donors for financial support. 

6.	Recommend actionable initiatives to the organization: This is the higher level of interpreting the hidden implication of the analysis results. This is the real deal of why we even do data analysis. It is ironic that while this step is the most important, many technical analysts actually ignore it. 

