from collections import namedtuple
from pyspark import SparkContext
from pyspark.sql import functions
from pyspark.sql.functions import col, unix_timestamp
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from xml import etree


sc = SparkContext("local[*]", "temp")

###### Users and Posts RDD
all_users_lines = sc.textFile("file:///home/tnguyen/projects/spark/spark-stats-data/allUsers/*.xml")
all_posts_lines = sc.textFile("file:///home/tnguyen/projects/spark/STATS_DATA/allPosts/*.xml")

### Posts
post = (namedtuple('post', ['userid_of_p', 'p_info']))

#post info
p_info = (namedtuple('p_info', ['p_type', 'p_CreatedDate', 
                                    'score', 'view_ct', 
                                    'ans_ct', 'fav_ct']))

def parse_post_toIdVet(row):
    """
    Returns a tuple of (OwnerUserId, postTypeID, CreationDate, ViewCount, AnswerCount,
    FavoriteCount, Score).

    row is a string.
    
    post is a dictionary.
    """
    
    try:
        root = etree.fromstring(row.encode('utf-8'))
        userid_of_p = int(root.attrib['OwnerUserId'])
        p_type = int(root.attrib['PostTypeId'])
        p_CreatedDate = root.attrib['CreationDate']
        
        score = float(root.attrib.get('Score', 0.0))
        view_ct = float(root.attrib.get('ViewCount', 0.0))
        ans_ct = float(root.attrib.get('AnswerCount', 0.0))
        fav_ct = float(root.attrib.get('FavoriteCount', 0.0))
        
    except Exception:  
        return None
    
    else:
        return ( post(userid_of_p, 
                 p_info(p_type, p_CreatedDate, score, view_ct,
                          ans_ct, fav_ct))) 


all_posts = (all_posts_lines.filter(lambda x: x.strip().startswith('<row'))
                .map(parse_post_toIdVet)
                .filter(lambda post: post is not None)
                .cache()
)

# Working in DF
all_post_df = all_posts.toDF()

post_q_df_init = (all_post_df.select(all_post_df['userid_of_p'], 
                          all_post_df['p_info']['p_type'].alias('p_type'),
                          all_post_df['p_info']['p_CreatedDate'].alias('p_CreatedDate'),
                          all_post_df['p_info']['score'].alias('score'),
                          all_post_df['p_info']['view_ct'].alias('view_ct'),
                          all_post_df['p_info']['ans_ct'].alias('ans_ct'),
                          all_post_df['p_info']['fav_ct'].alias('fav_ct'))
      )

post1_q_df = post_q_df_init.filter(post_q_df_init['p_type'] == 1)

post_q_df = post1_q_df.withColumn('postCreated',F.to_timestamp('p_CreatedDate'))

post_q_df = (post_q_df.drop(post_q_df.p_CreatedDate))

w=Window().partitionBy('userid_of_p').orderBy("postCreated")

firstQ_post_df = (post_q_df.withColumn("1stQ_date",F.first("postCreated").over(w))
                           .filter("postCreated=1stQ_date")
                 )
firstQ_post_df = firstQ_post_df.drop("postCreated")

u_info = (all_users_lines.filter(lambda line: line.strip().startswith('<row'))
                         .map(getUser)
                         .filter(lambda user: user is not None)
        )


#### Users 
user_tup = namedtuple('user_tup', ['userid', 'actCreatedDate'])

def getUser(row):
    """For a user, return a tuple of ('userid', 'actCreatedDate')
    
    row is a string."""
    
    try: 
        root = etree.fromstring(row.encode('utf-8'))
        userid = int(root.attrib['Id'])
        actCreatedDate = root.attrib['CreationDate']
        
    except Exception:
        
        return None
    
    else:
        return user_tup(userid, actCreatedDate) 

### Working with DF
u_info_df = u_info.toDF()

u_info_df = u_info_df.withColumn('actCreated',F.to_timestamp('actCreatedDate'))

user_info_df_final = u_info_df.drop("actCreatedDate")

postid_cd_df = (all_post_df.select(all_post_df['userid_of_p'],
                            all_post_df['p_info']['p_CreatedDate'].alias('p_CreatedDate'))
                         )


post_final_df = postid_cd_df.withColumn('p_CreatedDate',F.to_timestamp('p_CreatedDate'))


#### Combine Users and Posts
vetORnot_df = (post_final_df.join(user_info_df_final, 
                post_final_df.userid_of_p == user_info_df_final.userid, "inner")
               
                            .withColumn("diffInSec", 
                                            F.col("p_CreatedDate").cast("long") - F.col('actCreated').cast("long"))
                            .withColumn('usr_type', F.when(
                                                  (F.col('diffInSec') >= (100*60*60*24)) & 
                                                  (F.col('diffInSec') <= (150*60*60*24)), 1).otherwise(0))
                              .groupby("userid")
                              .sum().select("userid", "sum(usr_type)")
                              .withColumn('usr_type', F.when(F.col('sum(usr_type)')>0, 1).otherwise(0))
                              .select('userid', "usr_type")    
         )

idVet_res = (vetORnot_df.join(firstQ_post_df, vetORnot_df.userid == firstQ_post_df.userid_of_p, "inner")
                         .groupBy("usr_type")
                         .agg(F.mean("score"), F.mean("view_ct"), F.mean("ans_ct"), F.mean("fav_ct"))
             
            )


idVet_list = idVet_res.rdd.collect()


res_dict = {}
for row in idVet_list:
    
    if row[0] == 1:
        res_dict["vet_score"] = row[1]
        res_dict["vet_views"] = row[2]
        res_dict["vet_answers"] = row[3]
        res_dict["vet_favorites"] = row[4]
    else:
        res_dict["brief_score"] = row[1]
        res_dict["brief_views"] = row[2]
        res_dict["brief_answers"] = row[3]
        res_dict["brief_favorites"] = row[4]
