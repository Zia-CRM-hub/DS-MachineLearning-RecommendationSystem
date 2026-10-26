"""Class-based recommendation engine (rank, user-user CF, content-based, SVD).

Mirrors the functions developed in ``starter/Recommendations_with_IBM.ipynb`` so the
same logic can be reused in a web application.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class IBMRecommender:
    """Fit once on a user-item interactions table, then recommend articles.

    Parameters
    ----------
    n_clusters : number of KMeans title clusters for content recommendations.
    n_latent : number of SVD latent features for article-article similarity.
    random_state : seed for KMeans / SVD.
    """

    def __init__(self, n_clusters=50, n_latent=200, random_state=42):
        self.n_clusters = n_clusters
        self.n_latent = n_latent
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    @classmethod
    def from_csv(cls, path, **kwargs):
        """Build and fit a recommender from the ``user-item-interactions.csv`` file."""
        df = pd.read_csv(path, dtype={"article_id": int, "title": str, "email": str})
        return cls(**kwargs).fit(df)

    def fit(self, df):
        """Fit on a dataframe with ``article_id``, ``title`` and ``email`` (or ``user_id``) columns."""
        df = df.copy()
        if "user_id" not in df:
            emails = df["email"].fillna("unknown_user")
            codes = {e: i for i, e in enumerate(emails.unique(), start=1)}
            df["user_id"] = emails.map(codes)
        self.df_ = df[["user_id", "article_id", "title"]]
        self.titles_ = df.drop_duplicates("article_id").set_index("article_id")["title"]
        self.popularity_ = df["article_id"].value_counts()  # interactions per article
        self.user_item_ = (
            df[["user_id", "article_id"]].drop_duplicates().assign(v=1)
            .pivot(index="user_id", columns="article_id", values="v").fillna(0).astype(int)
        )
        self._fit_content()
        self._fit_svd()
        return self

    def _fit_content(self):
        """TF-IDF -> LSA -> KMeans on article titles."""
        articles = self.titles_.reset_index()
        tfidf = TfidfVectorizer(max_df=0.75, min_df=5, stop_words="english", max_features=200)
        lsa = make_pipeline(TruncatedSVD(n_components=50, random_state=self.random_state), Normalizer(copy=False))
        x = lsa.fit_transform(tfidf.fit_transform(articles["title"]))
        km = KMeans(n_clusters=self.n_clusters, n_init=5, max_iter=50, random_state=self.random_state).fit(x)
        self.cluster_ = pd.Series(km.labels_, index=articles["article_id"])

    def _fit_svd(self):
        """Article-article cosine similarity in SVD latent space."""
        # Fit all components, then keep the first n_latent (as in the notebook): the randomized solver
        # gives slightly different trailing components when fewer are requested.
        n_all = self.user_item_.shape[1]
        svd = TruncatedSVD(n_components=n_all, n_iter=5, random_state=self.random_state).fit(self.user_item_)
        vt = svd.components_[: self.n_latent]
        self._article_sim = pd.DataFrame(
            cosine_similarity(vt.T), index=self.user_item_.columns, columns=self.user_item_.columns
        )

    # --------------------------------------------------------------- lookups
    def article_names(self, article_ids):
        """Titles for a list of article ids."""
        return self.titles_.loc[list(article_ids)].tolist()

    def user_articles(self, user_id):
        """Article ids the user has interacted with."""
        row = self.user_item_.loc[user_id]
        return row[row == 1].index.tolist()

    def has_user(self, user_id):
        """True if the user is known (so personalised recs are possible)."""
        return user_id in self.user_item_.index

    # ----------------------------------------------------------- recommenders
    def top_articles(self, n=10):
        """Rank-based: the n most-interacted-with article ids."""
        return self.popularity_.head(n).index.tolist()

    def similar_users(self, user_id):
        """Neighbour table (neighbor_id, similarity, num_interactions), best neighbours first."""
        sims = cosine_similarity(self.user_item_.loc[[user_id]], self.user_item_)[0]
        out = pd.DataFrame({
            "neighbor_id": self.user_item_.index,
            "similarity": sims,
            "num_interactions": self.user_item_.sum(axis=1).values,
        })
        out = out[out["neighbor_id"] != user_id]
        return out.sort_values(["similarity", "num_interactions"], ascending=False, kind="mergesort").reset_index(drop=True)

    def user_recs(self, user_id, m=10):
        """User-user collaborative filtering; unknown (new) users fall back to the top articles."""
        if not self.has_user(user_id) or not self.user_articles(user_id):
            return self.top_articles(m)
        seen = set(self.user_articles(user_id))
        recs = []
        for neighbor in self.similar_users(user_id)["neighbor_id"]:
            new = [a for a in self.user_articles(neighbor) if a not in seen and a not in recs]
            recs.extend(sorted(new, key=lambda a: self.popularity_[a], reverse=True))
            if len(recs) >= m:
                break
        return recs[:m]

    def content_recs(self, article_id, n=10):
        """Articles in the same title cluster, most-read first."""
        same = self.cluster_[self.cluster_ == self.cluster_[article_id]].index.drop(article_id)
        ranked = self.user_item_[list(same)].sum(axis=0).sort_values(ascending=False, kind="mergesort")
        return [int(a) for a in ranked.index[:n]]

    def svd_recs(self, article_id, n=10):
        """Articles closest to ``article_id`` in SVD latent space (cosine similarity)."""
        sims = self._article_sim[article_id].drop(article_id).sort_values(ascending=False, kind="mergesort")
        return [int(a) for a in sims.index[:n]]
