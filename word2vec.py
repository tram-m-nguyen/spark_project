from collections import namedtuple
from pyspark import SparkContext
from xml import etree
import re
from pyspark.ml.feature import Word2Vec


sc = SparkContext("local[*]", "temp")

###### Users and Posts RDD
all_users_lines = sc.textFile("file:///home/tnguyen/projects/spark/spark-stats-data/allUsers/*.xml")
all_posts_lines = sc.textFile("file:///home/tnguyen/projects/spark/STATS_DATA/allPosts/*.xml")


p = (namedtuple('p', ['userid_of_p', 'p_info']))
p_info_t = (namedtuple('p_info_t', ['tags','p_type', 'p_CreatedDate',
                                'view_ct', 'ans_ct', 
                                'fav_ct', 'score']))

def get_post(row):
    """
    row is a String.
    Returns a tuple of (OwnerUserId, postTypeID, CreatinDate, ViewCount, AnswerCount,
    FavoriteCount, Score).
    """    
    
    try:
        root = etree.fromstring(row.encode('utf-8'))
        
        userid_of_p = int(root.attrib['OwnerUserId'])
        tags = root.attrib['Tags']
        p_type = int(root.attrib['PostTypeId'])
        p_CreatedDate = root.attrib['CreationDate']
        view_ct = float(root.attrib.get('ViewCount', 0.0))
        ans_ct = float(root.attrib.get('AnswerCount', 0.0))
        fav_ct = float(root.attrib.get('FavoriteCount', 0.0))
        score = float(root.attrib.get('Score', 0.0))
        

    except Exception:  
        return None
    
    else:
        return ( p(userid_of_p, 
                  p_info_t(tags, p_type, p_CreatedDate, view_ct,ans_ct, 
                  fav_ct,score)) 
               )


posts_fulldataset = (all_posts_lines.filter(lambda x: x.strip().startswith('<row'))
                .map(get_post)
                .filter(lambda post: post is not None)
               .cache() 
)               


tags_text = ( posts_fulldataset.map(lambda line: line[1].tags)
                  .map(lambda tag: ([item for item in re.split("<|>", tag) if item], 1))
)

text_df = tags_text.toDF(['text', 'score'])

w2v = Word2Vec(inputCol="text", outputCol="vectors", vectorSize=100, seed=42)
model = w2v.fit(text_df)

model_res = model.findSynonyms('ggplot2', 25)

rdd = model_res.rdd
res_list = rdd.take(25)

final_word2vec = [(row[0],row[1]) for row in res_list]
