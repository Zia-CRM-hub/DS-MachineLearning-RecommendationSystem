# Recommendation System: IBM Community

Recommendations for articles on the IBM Watson Studio platform, built from user-article interaction data
(45,993 interactions, 5,149 users, 714 articles).

| Part | Method | Where |
|---|---|---|
| I | Exploratory data analysis | notebook |
| II | Rank-based (popularity) recommendations | notebook |
| III | User-user collaborative filtering (cosine similarity, tie-broken by activity and popularity), new-user fallback | notebook |
| IV | Content-based: TF-IDF + LSA + KMeans on article titles (elbow + silhouette to pick the cluster count) | notebook |
| V | Matrix factorization: SVD, choice of latent features (in-sample metrics vs. held-out recall@10), article-article similarity | notebook |

The executed notebook is [`starter/Recommendations_with_IBM.ipynb`](starter/Recommendations_with_IBM.ipynb)
(HTML export: [`starter/Recommendations_with_IBM.html`](starter/Recommendations_with_IBM.html)). All provided
tests in `starter/project_tests.py` pass.

Key findings: the median user has 3 interactions and the most-read article (1429) has 937; adding latent
features keeps improving *in-sample* accuracy/recall but held-out recall@10 peaks at about 20 latent features
and drops below the most-popular baseline by 300 - see Part V of the notebook.

## Reusable package and web API

The same recommenders are packaged as a class, `IBMRecommender`:

```bash
pip install -e .            # or: pip install -e ".[web,test]"
```

```python
from ibm_recommender import IBMRecommender

rec = IBMRecommender.from_csv("starter/data/user-item-interactions.csv")
rec.top_articles(10)         # rank-based (new users)
rec.user_recs(20, m=10)      # user-user CF; unknown users fall back to the top articles
rec.content_recs(25, n=10)   # same title cluster, most-read first
rec.svd_recs(4, n=10)        # nearest articles in SVD latent space
rec.article_names([1429])    # ids -> titles
```

A small Flask API is in [`webapp/app.py`](webapp/app.py) (`pip install -e ".[web]"`, then `python webapp/app.py`):

* `GET /recommend/top?n=10`
* `GET /recommend/user/<user_id>?m=10`
* `GET /recommend/article/<article_id>?n=10&method=content|svd`

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests      # checks the class against the notebook's known answers
```

The notebook's own checks use `starter/project_tests.py`.

## Layout

```
starter/        notebook, data/user-item-interactions.csv, provided tests
ibm_recommender/  recommender class (pip installable)
webapp/         Flask API
tests/          pytest suite for the package
```

## License

[License](LICENSE.txt)
