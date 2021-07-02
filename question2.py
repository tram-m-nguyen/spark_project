from pyspark import SparkContext
sc = SparkContext("local[*]", "temp")

# Posts RDD
all_posts_lines = sc.textFile("file:///home/tnguyen/projects/spark/STATS_DATA/allPosts/*.xml")
all_posts_lines.count()

from collections import namedtuple

post_fav_score = namedtuple('post_fav_score', ['favScore', 'score'])

def getFavoriteCount_score(post):
	"""
	For each post, return a tuple of ('FavoriteCount', Score)
	Post is a dictionary, some maynot have FavoriteCount or Score.

	Want post with tuple of both FavoriteCount and Score.

	"""

	try: 
		favScore = post['FavoriteCount']
		score = post['Score']

	except KeyError:

		return None
	
	else:
		return post_fav_score(int(favScore), int(score))



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


all_posts_fav_score = all_posts_lines.filter(lambda line: line.strip().startswith('<row'))\ 
									.map(re_encode_row)\
									.filter(lambda rootAttriDic: rootAttriDic is not None)\
									.map(getFavoriteCount_score)\
									.filter(lambda favCount_Score_tup: favCount_score_tup is not None)\
									.mapValues(lambda x: (x,1))\
									.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
									.map(lambda x: (x[0], x[1][0] / x[1][1]))\
									.sortByKey()\
									.take(50)