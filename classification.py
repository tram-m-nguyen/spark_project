from pyspark import SparkContext
from collections import namedtuple
from lxml import etree
import xml.etree.ElementTree as ET
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LinearRegression
from pyspark.sql import SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np


sc = SparkContext("local[*]", "temp")

###### Users and Posts RDD
tags_train = sc.textFile("file:///home/tnguyen/projects/spark/spark-stats-data/train/*.xml")
tags_test = sc.textFile("file:///home/tnguyen/projects/spark/spark-stats-data/test/*.xml")


post_tag = (namedtuple('post_tag', ['postId','postType','body', 'tags']))

def get_tag_post(row):
    """
    row is a string
    Returns a tuple of (Tags and postTypeID).
    """    
    
    try:
        root = etree.fromstring(row.encode('utf-8'))
        postId = int(root.attrib['Id'])
        postType = int(root.attrib['PostTypeId'])
        body = root.attrib['Body']
        tags = root.attrib['Tags']
        
        
    except Exception:  
        return None
    
    else:
        return post_tag(postId, postType, body, tags) 

tags_count = (tags_train.filter(lambda x: x.strip().startswith('<row'))
           .map(get_tag_post)
           .filter(lambda tagpost: tagpost is not None and tagpost.postType == 1)
            .map(lambda post: [tag for tag in (re.split("<|>", post.tags)) if tag])
           .flatMap(lambda tagslist: [lst for lst in tagslist]) 
            .map(lambda word: (word.lower(), 1))
            .reduceByKey(lambda x,y: x + y)
)

top10_tags_count = tags_count.takeOrdered(10, key = lambda x: -x[1])

top10_tags_list = list(zip(*top10_tags_count))[0]

## Get PostId and Tags
def parse_body(body):
    """Body is a string. 
    
    Returns only words from the string. 
    """
    
    return " ".join([string for string in body.split(" ") if re.search(r"^\w+$", string) is not None])


def check_in_top10(post):
    """ Return tuple (postid, body, tags, 0 or 1).
        0 for tags not in top10_tags list.
        1 for tags in top10_tags list.
    """
    tags = post[2]

    for tag in tags:
        if tag in top10_tags:
            return (post[0], post[1], post[2], 1) 
        
    return (post[0], post[1], post[2], 0)


tags = (tags_train.filter(lambda x: x.strip().startswith('<row'))
                   .map(getTagPost)
                   .filter(lambda tagpost: tagpost is not None 
                           and tagpost.postType == 1)
                   .cache()
)

body_lab = (tags.map(lambda post: (post.postId, post.body, post.tags))
                     .map(lambda post: (post[0], parse_body(post[1]), post[2])) 
                     .map(lambda post: (post[0], post[1], 
                                        [tag for tag in re.split("<|>", post[2]) if tag]) ) #gets a list of tags
                     .map(check_in_top10)
)

body = (tags.map(lambda post: (post.postId, post.body, post.tags))
                     .map(lambda post: (parse_body(post[1])))
                     
)


#### Test Set
post_tags_test = (tags_test.map(lambda post: (post.postId, post.body, post.tags))
                           .map(lambda post: (post[0], parse_body(post[1]), post[2])) 
                           .map(lambda post: (post[0], post[1], 
                                        [tag for tag in re.split("<|>", post[2]) if tag]) )
)

body_test = (tags_test.map(lambda post: (post.postId, post.body, post.tags))
                            .sortBy(lambda post: post[0])                  
                           .map(lambda post: (parse_body(post[1]))) 
                            .collect()
)

#### Scikit-learn
bag_of_words_est = Pipeline([
    ('Vectorizer',CountVectorizer(min_df=0.01,max_df=0.98)),
    ('TfidfTrans',TfidfTransformer()),
    ('LinRegress', LinearRegression())
])

bag_of_words_est.fit(body, body_lab)


preds = bag_of_words_est.predict(body_test)
preds2 = []
for item in preds:
    if item > 0.5:
        preds2.append(1)
    else:
        preds2.append(0)



#### Dataframe
training = sqlContext.createDataFrame(post_tags_labelled,["postid", 'body', 'tag', 'label']).cache()
test = sqlContext.createDataFrame(post_tags_test,["postid", 'body', 'tag'])

tokenizer = Tokenizer(inputCol="body", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
logreg = LogisticRegression(maxIter=10, regParam=0.001)

tokens = tokenizer.transform(training)
hashes = hashingTF.transform(tokens)
model = logreg.fit(hashes) 

# MAKE PREDICTION ON TEST SET
test_tokens = tokenizer.transform(test)
test_hashes = hashingTF.transform(test_tokens)

prediction = model.transform(test_hashes)
selected = prediction.select("body", "tag", 'prediction')


prediction_int = prediction.withColumn('prediction', prediction.prediction.cast('int'))
predict_sorted = prediction_int.sort('postid')


predict_res = predict_sorted.select('prediction').rdd.collect()

class_predict = [row[0] for row in predict_res]
class_predict[:4]

## Tune Hyperparameters
pipeline = Pipeline(stages=[tokenizer, hashingTF, logreg])


#TUNE HYPERPARAMTERS
paramGrid = (ParamGridBuilder()
             .addGrid(hashingTF.numFeatures, [100, 1000])
             .addGrid(logreg.regParam, [0.1, .01])
             .build()

)

crossval = CrossValidator(estimator=pipeline,
                             estimatorParamMaps=paramGrid,
                             evaluator=BinaryClassificationEvaluator(),
                            numFolds=5
                         
                         )

cvModel = crossval.fit(training)
better_prediction = cvModel.transform(test)
selected_cv = better_prediction.select("body",'prediction')

better_prediction_int = (better_prediction.withColumn(
                                        'prediction', better_prediction.prediction.cast('int')))



better_predict_sorted = better_prediction_int.sort('postid')

predict_tuneParams_res = better_predict_sorted.select('prediction').rdd.collect()

class_predict_cv = [row[0] for row in predict_tuneParams_res]

stages = cvModel.bestModel.stages

best_run = np.array(cvModel.avgMetrics).argmin()
paramGrid[best_run]