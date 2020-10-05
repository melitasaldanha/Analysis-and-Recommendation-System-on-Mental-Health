import re
import json
import numpy as np
from numpy import array
from numpy.linalg import inv
import sys
from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from math import sqrt
import math
from pyspark.sql import SQLContext



sc = SparkContext('local[*]', 'assign3')
spark = SQLContext(sc)


# Function to create vectors from the dictionary to get the cosine distance
def setup_vec(dict1, dict2):
    dict1_missing = list(set(dict2.keys()) - set(dict1.keys()))
    dict2_missing = list(set(dict1.keys()) - set(dict2.keys()))
    for i in dict1_missing:
        dict1[i] = 0
    for i in dict2_missing:
        dict2[i] = 0
    vec1 = []
    vec2 = []
    for i in dict1.keys():
        vec1.append(dict1[i])
        vec2.append(dict2[i])
    return ([vec1, vec2])


# Dot product of two vectors
def dot_prod(p, q):
    return sum([p[x] * q[x] for x in range(0, len(p))])


# Normalize the values
def norm(p):
    return math.sqrt(dot_prod(p, p))


# Calculate the cosine distance.
def cosine_metric(p, q):
    return dot_prod(p, q) / (norm(p) * norm(q))


# Function to compute cosine similarity between two SEQN's
def cosine_similarity(row):
    x = {}
    y = {}
    # Extract data for two different SEQN.
    # Joining two rdd's will have two columns with same names hence the _x and _y
    for field in row.__fields__:
        s1, s2 = field.split("_")
        if s2 == "x":
            x[s1] = row[field]
        else:
            y[s1] = row[field]
    # Extract ID from both the dictionaries
    id1 = x["SEQN"]
    id2 = y["SEQN"]
    # Don't calculate similarity with itself.
    if id1 == id2:
        return []
    # Delete the SEQN variable we don't need it in similarity.
    x.pop('SEQN', None)
    y.pop('SEQN', None)
    # We don't need the target value to calculate the similarity as well.
    x.pop(target.value, None)
    y.pop(target.value, None)
    # Setup the vectors fromt he dictionary
    vecs = setup_vec(x, y)
    # Calculate the cosine values
    val = cosine_metric(vecs[0], vecs[1])
    return [(id2, (id1, val))]


# Inverts the similarity matrix after the initial operations have been performed.
def invert_similarity_matrix(row):
    id = row[0]
    ret = []
    for v in row[1]:
        id1, simi = v
        tup = (id1, (id, simi))
        ret.append(tup)
    return ret


# Sorts the similarity for each SEQN we need a prediction for and extracts top 50
def sort_and_pick(row):
    id = row[0]
    vals = sorted(row[1])
    vals = vals[:50]
    return (id, vals)


# Uses the broadcasted similarity list and target column
# Broadcasts a rows contribution to each of the SEQN we need a prediction for.
def broadcast_weighted_contribution(row):
    id = row["SEQN"]
    similarities = []
    val = row[target.value]
    ret = []
    for tup in similarity_list.value:
        id1, li = tup
        if id1 == id:
            similarities = li
            break
    for tup in similarities:
        id1, similarity = tup
        new_tup = (id1, (val * similarity, similarity))
        ret.append(new_tup)
    return ret


# Function which uses the grouped contributions to calculate the predicted values.
def generate_predictions(row):
    id = row[0]
    num = 0
    denom = 0
    for v in row[1]:
        num_c, den_c = v
        num = num + num_c
        denom = denom + den_c
    weighted_avg = num / denom
    return id, weighted_avg

#Extract the true values of the SEQNs target variable.
def extract_true_target_values(row):
    id = row["SEQN"]
    t = row[target.value]
    return (id, t)


if __name__ == '__main__':
    # Read the files from hdfs first argument is the target column and the rest are
    n = len(sys.argv)
    if n < 2:
        print("Error: Need csv files")
        exit(-1)
    target_column = sys.argv[1]
    target = sc.broadcast(target_column)
    df = []
    for i in range(2, n):
        tmp = spark.read.csv(sys.argv[i], header=True)
        tmp = tmp.select(*(col(c).cast("float").alias(c) for c in tmp.columns))
        df.append(tmp)

    # Join the datasets
    df_main = df[0]
    for i in range(1, len(df)):
        df_main = df_main.join(df[i], ["SEQN"])

    #Rename the dataset
    df2 = df_main

    # Split the dataframe Test on just 20% of the data.
    main_df, test_df = df2.randomSplit([0.05, 0.9])

    # Take 5% of the data for the prediction
    y, x = main_df.randomSplit([.5, .95])
    # Find the similarities among each row
    m_f_x = x.select([F.col(c).alias('%s_x' % (c)) for c in list(x.columns)])
    m_f_y = y.select([F.col(c).alias('%s_y' % (c)) for c in list(y.columns)])

    #Join the two for calculating similarities.
    df3 = m_f_x.crossJoin(m_f_y)

    # Have an RDD of similarities just need to calculate for each.
    similarities = df3.rdd.flatMap(cosine_similarity)
    similarities_grouped = similarities.groupByKey()
    similarities_final = similarities_grouped.map(sort_and_pick)

    # Invert the similarities for easier use
    inverted_similarities = similarities_final.flatMap(invert_similarity_matrix)
    inverted_similarities_grouped = inverted_similarities.groupByKey()

    # Broadcast the similarities
    similarity_list = sc.broadcast(inverted_similarities_grouped.collect())

    #Fetch the contributions from each row
    contributions = x.rdd.flatMap(broadcast_weighted_contribution).groupByKey()
    #Add the contributions to generate predictions
    predictions = contributions.map(generate_predictions)
    #Extract the true value to compare predictions against.
    true_values = y.rdd.map(extract_true_target_values)
    #Do an inner join to compute the error together.
    joined_vals = true_values.join(predictions)

    #Calculate and print the mean squared error.
    MSE = joined_vals.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))
