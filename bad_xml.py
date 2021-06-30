from pyspark import SparkContext
sc = SparkContext("local[*]", "temp")

# to create an RDD from file
all_posts_lines = sc.textFile("file:///home/tnguyen/projects/spark/STATS_DATA/allPosts/*.xml")
all_posts_lines.count()

from lxml import etree

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


badxml = all_posts_lines.filter(lambda x: x.strip().startswith('<row') \
						.map(re_encode_row)	\
						.filter(lambda x: x is None).count()