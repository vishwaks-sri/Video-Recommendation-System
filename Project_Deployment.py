from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
import pandas as pd

data=pd.read_csv('Finaldata.csv')
title=data['Title'].values
def recommend(title):
	for i in data:
		if data[i].isnull().sum()!=0:
			data[i]=data[i].fillna('')
	if title not in data['Title'].unique():
		return "This video is not present in our database.\nPlease check if you spelled it correct."
	else:
		data['Text']=data['Title']+data['Description']
		cv = CountVectorizer()
		count_matrix=cv.fit_transform(data['Text'])
		similarity = cosine_similarity(count_matrix)
		tf=TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=0,stop_words='english')
		tfidf_matrix=tf.fit_transform(data['Text'])
		titles=data['Title']
		indices=pd.Series(data.index,index=data['Title'])
		idx=indices[title]
		scores=list(enumerate(similarity[idx]))
		scores=sorted(scores,key=lambda x:x[1],reverse=True)
		scores=scores[1:6]
		video_index=[i[0] for i in scores]
		A=list(data['Video Id'].iloc[video_index])
		B=list(data['Title'].iloc[video_index])
		videos_detail={A[i]:B[i] for i in range(len(A))}
		return videos_detail


app=Flask(__name__)
@app.route("/")
@app.route("/home",methods=["GET"])
def home():
    #if request.method == "GET":
    return render_template("index1.html",title=title)
		
@app.route("/r",methods=["POST"])
def recommendation():
	title=request.form['input']
	rcm=recommend(title)
	#if type(rcm)==type('string'):
	return render_template('recommend.html',prediction_text=rcm)
	#else:
		#m_str=list(rcm)
		#return render_template('index1.html',prediction_text=m_str,title=title)
		
if __name__ == '__main__':
	app.run(debug=True)