import torch

from caff_efficientvpr.retrieval import recall_at_k


def test_small_retrieval_recall():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = torch.tensor([[0.95, 0.05], [0.1, 0.9], [-1.0, 0.0]])
    query_rows = [
        {"image_path": "q1.jpg", "place_id": "a"},
        {"image_path": "q2.jpg", "place_id": "b"},
    ]
    gallery_rows = [
        {"image_path": "g1.jpg", "place_id": "a"},
        {"image_path": "g2.jpg", "place_id": "b"},
        {"image_path": "g3.jpg", "place_id": "c"},
    ]
    result = recall_at_k(queries, gallery, query_rows, gallery_rows, (1, 2))
    assert result["valid_queries"] == 2
    assert result["recall"]["R@1"] == 1.0
    assert result["recall"]["R@2"] == 1.0

