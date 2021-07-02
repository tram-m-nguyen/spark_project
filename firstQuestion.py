from collections import namedtuple
from pyspark import SparkContext
from pyspark.sql import functions
from pyspark.sql.functions import col
from pyspark.sql.functions import unix_timestamp
import pyspark.sql.functions as F


sc = SparkContext("local[*]", "temp")

###### Users and Posts RDD
all_users_lines = sc.textFile("file:///home/tnguyen/projects/spark/spark-stats-data/allUsers/*.xml")
all_posts_lines = sc.textFile("file:///home/tnguyen/projects/spark/STATS_DATA/allPosts/*.xml")


###### Process Users

user_tuple = namedtuple('user_tuple', ['userid', 'rep', 'act_created_date'])

def parse_user_info(user):
    """
    Return a user tuple of (userid, reputation, creationDate)
    
    User is a dictionary, some may or may not have ID or Reputation.
    """
    
    try: 
        userid = int(user['Id'])
        rep = user['Reputation']
        act_created_date = user['CreationDate']
        
    except KeyError:
        return None

    else: 
        return user_tuple((userid), int(rep), act_created_date) 


####### Process Posts

post_type = namedtuple('post_type', ['userid_of_posts', 'postType', 'post_created_date'])

def parse_post_forquestion(post):
    """Returns a tuple of (userid_of_posts, postType, post_created_date).
    
    postType: 1 - question, 2 - answer 
    
    post is a dictionary
    """

    try:
        userid_of_posts = post['OwnerUserId']
        postType = post['PostTypeId']
        post_created_date = post['CreationDate']

    except KeyError:
        return None
    
    else:
        return post_type(userid_of_posts, int(postType), post_created_date)  


def re_encode_row(row):
	"""
	
	Re-encode row to UTF-8 because text is unicode.

	Returns a dictionary of the row content. 

	Row dictionary: {'AnswerCount': '0', 'Body': '<p>I\'m having trouble with a 
	basic machine learning methodology question....I\'m not sure whether or 
	not I understand the concept of   nested cross vhere"></p>\n', 
	'CommentCount': '2', 
	'CreationDate': '2014-06-04T13:18:18.120', 'Id': '101120', 
	'LastActivityDate': '2014-06-06T09:46:55.340', 
	'LastEditDate': '2014-06-06T09:46:55.340', 'LastEditorUserId': '16609', 
	'OwnerUserId': '16609', 'PostTypeId': '1', 'Score': '0', 
	'Tags': '<machine-learning><cross-validation>', 
	'Title': 'Correct methodology to repeat testing of classifier to get 
	good estimate of performance', 'ViewCount': '62'}
	"""

	try:
		root = etree.fromstring(row.encode('utf-8'))

		return dict(root.attrib)

	except Exception:

		return None




post_questions = (all_posts_lines.filter(lambda x: x.strip().startswith('<row'))
                                .map(re_encode_row)
                                .filter(lambda x: x is not None)
                                .map(parse_post_forquestion)
                                .filter(lambda x: x is not None)
                                .filter(lambda postTup: postTup.postType == 1)
                        )

users_act = (all_users_lines.filter(lambda x: x.strip().startswith('<row'))
                    .map(re_encode_row)
                    .filter(lambda x: x is not None)
                     .map(parse_user_info)
                     .filter(lambda x: x is not None)
        	)


###### Using DataFrames ######

# Posts DF
post_questions_df = post_questions.toDF()

question_df = post_questions_df.withColumn('postCreatedDate_timestp',
                       F.to_timestamp('postCreatedDate'))

# First question date
first_question_date_df = (question_df.orderBy('postCreatedDate_timestp')
                       				.groupby("userid_of_posts")
									.agg(functions.min("postCreatedDate_timestp")
									.alias("1st_Q_CreatedDate_timestp"), 
                            			functions.first("postType").alias("postType"))

    				    )

# Users DF
users_df_init = users_act.toDF()

users_df = (users_df_init.withColumn('actCreatedDate_timestp', 
										F.to_timestamp('actCreatedDate')))



# Get User's first question date
users_first_question_date_df = (
	first_question_date_df.select(['userid_of_posts', '1st_Q_CreatedDate_timestp',
                                     'postType'])
    					  .join(
							  users_df, first_question_date_df.userid_of_posts == users_df.userid, 
                  			'inner')
            				.withColumn('dayDiff', functions.round((unix_timestamp(
                    		first_question_date_df['1st_Q_CreatedDate_timestp']) - 
                			unix_timestamp(users_df.actCreatedDate_timestp))/(60*60*24), 2).cast('integer'))
								
								)


sorted_dayDiff_df = users_first_question_date_df.sort(col('rep').desc())

### df of userid and days from when account was created to first post
days_since_first_post_df = (sorted_dayDiff_df.drop("firstQuestPostDate", 
									"postType", 
									"actCreatedDate", 
									"userid_of_posts",
									'rep',
									'actCreatedDate_timestp')
         	 )

## Top 100 users and their first questions
top_100_user_firstQ = (days_since_first_post_df.select("userid", "dayDiff")
                       .rdd
					   .map(lambda x: (int(x[0]),x[1]))
					   .collect())

