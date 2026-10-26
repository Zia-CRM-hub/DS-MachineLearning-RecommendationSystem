"""Minimal Flask JSON API around IBMRecommender.

Run:  pip install -e ".[web]"  &&  python webapp/app.py
Try:  curl localhost:5000/recommend/user/20
      curl "localhost:5000/recommend/article/25?method=svd"
"""
import os

from flask import Flask, jsonify, request

from ibm_recommender import IBMRecommender

DATA = os.environ.get(
    "IBM_DATA", os.path.join(os.path.dirname(__file__), "..", "starter", "data", "user-item-interactions.csv")
)
app = Flask(__name__)
rec = IBMRecommender.from_csv(DATA)


def _payload(ids):
    return jsonify({"article_ids": [int(i) for i in ids], "titles": rec.article_names(ids)})


@app.get("/recommend/top")
def top():
    """Most popular articles (use for brand-new users)."""
    return _payload(rec.top_articles(request.args.get("n", 10, type=int)))


@app.get("/recommend/user/<int:user_id>")
def for_user(user_id):
    """User-user CF recommendations; unknown users get the most popular articles."""
    return _payload(rec.user_recs(user_id, request.args.get("m", 10, type=int)))


@app.get("/recommend/article/<int:article_id>")
def similar_article(article_id):
    """Similar articles; ?method=content (default) or ?method=svd."""
    if article_id not in rec.titles_.index:
        return jsonify({"error": "unknown article_id"}), 404
    n = request.args.get("n", 10, type=int)
    method = request.args.get("method", "content")
    return _payload(rec.svd_recs(article_id, n) if method == "svd" else rec.content_recs(article_id, n))


if __name__ == "__main__":
    app.run(debug=False)
