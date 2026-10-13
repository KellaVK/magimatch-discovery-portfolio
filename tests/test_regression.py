import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.checkpoint import Checkpoint
from src.config_loader import get_project_root, paths
from src.query_engine import cosine_similarity_matrix, merge_results, parse_intent, semantic_search


def test_cosine_similarity_handles_zero_vectors():
    scores = cosine_similarity_matrix(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    np.testing.assert_allclose(scores, [1, 0, 0])
    assert np.isfinite(scores).all()


def test_semantic_search_ranks_and_limits_results_without_network():
    with patch('src.query_engine.embed_single', return_value=np.array([1.0, 0.0])):
        results = semantic_search('cards', np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]), np.array([10, 20, 30]), top_k=2)
    assert [item[0] for item in results] == [20, 30]


def test_intent_normalizes_names_and_falls_back_on_bad_json():
    with patch('src.query_engine.chat_completion', return_value=json.dumps({'topic': 'cards', 'excluded_persons': [' Vernon '], 'referenced_persons': [' Mello ']})):
        parsed = parse_intent('cards like Mello, not Vernon')
    assert parsed['excluded_persons'] == ['vernon']
    assert parsed['referenced_persons'] == ['Mello']
    with patch('src.query_engine.chat_completion', return_value='invalid'):
        assert parse_intent('coins')['topic'] == 'coins'


def test_checkpoint_persists_and_deduplicates(tmp_path):
    cp = Checkpoint('test', tmp_path)
    cp.mark_done(12)
    cp.mark_done('12')
    cp.set_meta('batch', 3)
    restored = Checkpoint('test', tmp_path)
    assert restored.is_done(12)
    assert restored.count_done() == 1
    assert restored.get_meta('batch') == 3


def test_paths_follow_renamed_repository():
    assert get_project_root() == Path(__file__).resolve().parents[1]
    assert paths.db == get_project_root() / 'data/processed/magimatch.db'


def test_merge_deduplicates_title_and_semantic_results():
    row = {'id': 1, 'book_id': 7, 'authors': 'Example Author', 'credited_to': '', 'title': 'Example', 'match_type': 'title_exact'}
    result = merge_results([row], [(1, 0.9)], {1: row}, [])
    assert len(result) == 1
    assert result[0]['id'] == 1
