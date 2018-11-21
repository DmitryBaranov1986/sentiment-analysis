from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.pipeline import Pipeline

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [" ".join(movie_reviews.words(fileids=[f])) for f in negids]
posfeats = [" ".join(movie_reviews.words(fileids=[f])) for f in posids]

texts = negfeats + posfeats
labels = [0] * len(negfeats) + [1] * len(posfeats)

# Write out number of reviews
with open("output/simple_model_answer1.txt", "w") as f:
    f.write(str(len(texts)))

# Write out positive reviews ratio
with open("output/simple_model_answer2.txt", "w") as f:
    f.write(str(float(len(posfeats))/len(texts)))

mean_accuracy = cross_val_score(
    Pipeline([("vectorizer", CountVectorizer()),("classifier", LogisticRegression())]),
    texts, labels, scoring=metrics.make_scorer(metrics.accuracy_score)).mean()

with open("output/simple_model_answer3.txt", "w") as f:
    f.write(str(mean_accuracy))


mean_rocauc = cross_val_score(
    Pipeline([("vectorizer", CountVectorizer()),("classifier", LogisticRegression())]),
    texts, labels, scoring='roc_auc').mean()

with open("output/simple_model_answer4.txt", "w") as f:
    f.write(str(mean_rocauc))


