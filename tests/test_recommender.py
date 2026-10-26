import os

import pytest

from ibm_recommender import IBMRecommender

DATA = os.path.join(os.path.dirname(__file__), "..", "starter", "data", "user-item-interactions.csv")


@pytest.fixture(scope="module")
def rec():
    return IBMRecommender.from_csv(DATA)


def test_matrix_shape(rec):
    assert rec.user_item_.shape == (5149, 714)
    assert rec.user_item_.sum(axis=1)[1] == 36


def test_top_articles_match_notebook(rec):
    assert rec.top_articles(10)[0] == 1429
    assert set(rec.top_articles(10)) == {1314, 1429, 1293, 1427, 1162, 1364, 1304, 1170, 1431, 1330}


def test_similar_users(rec):
    assert rec.similar_users(1)["neighbor_id"].iloc[0] == 3933
    assert rec.similar_users(2)["neighbor_id"].iloc[5] == 3586
    assert rec.similar_users(131)["neighbor_id"].iloc[9] == 383


def test_user_recs_are_unseen(rec):
    recs = rec.user_recs(20, 10)
    assert len(recs) == 10 and not set(recs) & set(rec.user_articles(20))
    assert recs == [1293, 1162, 1185, 1165, 254, 1271, 1328, 1410, 40, 1402]  # notebook's user_user_recs_part2


def test_new_user_gets_popular(rec):
    assert rec.user_recs(10**9, 10) == rec.top_articles(10)


def test_content_and_svd(rec):
    assert 25 not in rec.content_recs(25, 10) and len(rec.content_recs(25, 10)) == 10
    assert set(rec.svd_recs(4, 10)) == {1199, 1068, 486, 1202, 176, 1120, 244, 793, 58, 132}
