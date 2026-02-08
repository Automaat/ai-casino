"""Tests for Truth Social fetcher."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import httpx
import pytest

from src.data.truth_social import TrumpPostData, TruthPost, TruthSocialFetcher


@pytest.fixture
def sample_archive_response():
    return [
        {
            "id": "123456",
            "content": "Great day for America! $TSLA going to the moon!",
            "created_at": "2025-01-15T10:30:00.000Z",
            "favourites_count": 50000,
            "reblogs_count": 10000,
            "replies_count": 5000,
            "url": "https://truthsocial.com/@realDonaldTrump/posts/123456",
        },
        {
            "id": "123455",
            "content": "Tariffs on China working great!",
            "created_at": "2025-01-14T08:00:00.000Z",
            "favourites_count": 45000,
            "reblogs_count": 8000,
            "replies_count": 3000,
            "url": "https://truthsocial.com/@realDonaldTrump/posts/123455",
        },
    ]


def test_truth_post_creation():
    post = TruthPost(
        id="123",
        content="Test post content",
        created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
        likes=1000,
        reposts=500,
        replies=200,
        url="https://truthsocial.com/post/123",
    )

    assert post.id == "123"
    assert post.content == "Test post content"
    assert post.likes == 1000
    assert post.reposts == 500


def test_trump_post_data_creation():
    posts = [
        TruthPost(
            id="1",
            content="Post 1",
            created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
            likes=100,
            reposts=50,
            replies=10,
            url="https://example.com/1",
        )
    ]

    data = TrumpPostData(
        posts=posts,
        total_count=1,
        latest_post_at=posts[0].created_at,
        fetched_at=datetime.now(UTC),
    )

    assert len(data.posts) == 1
    assert data.total_count == 1


def test_fetcher_init():
    fetcher = TruthSocialFetcher()
    assert fetcher._cache_dir.exists()


def test_fetcher_init_custom_cache():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = TruthSocialFetcher(cache_dir=tmpdir)
        assert str(fetcher._cache_dir) == tmpdir


def test_fetch_archive(sample_archive_response):
    with patch("src.data.truth_social.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_archive_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = TruthSocialFetcher()
        fetcher._cache.clear()  # Clear cache for test

        data = fetcher._fetch_archive()

        assert len(data) == 2
        mock_client.get.assert_called_once()


def test_fetch_recent(sample_archive_response):
    with patch("src.data.truth_social.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_archive_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = TruthSocialFetcher()
        fetcher._cache.clear()

        result = fetcher.fetch_recent(hours=48)

        assert isinstance(result, TrumpPostData)
        assert all(isinstance(p, TruthPost) for p in result.posts)


def test_fetch_since(sample_archive_response):
    with patch("src.data.truth_social.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_archive_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = TruthSocialFetcher()
        fetcher._cache.clear()

        cutoff = datetime(2025, 1, 14, 12, 0, 0, tzinfo=UTC)
        result = fetcher.fetch_since(cutoff)

        assert isinstance(result, TrumpPostData)
        # First post is after cutoff, second is before
        assert len(result.posts) == 1
        assert result.posts[0].id == "123456"


def test_fetch_all(sample_archive_response):
    with patch("src.data.truth_social.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_archive_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = TruthSocialFetcher()
        fetcher._cache.clear()

        result = fetcher.fetch_all()

        assert isinstance(result, TrumpPostData)
        assert len(result.posts) == 2


def test_get_latest_post_id(sample_archive_response):
    with patch("src.data.truth_social.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_archive_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = TruthSocialFetcher()
        fetcher._cache.clear()

        post_id = fetcher.get_latest_post_id()

        assert post_id == "123456"


def test_http_error():
    with patch("src.data.truth_social.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.get.side_effect = httpx.HTTPError("API Error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = TruthSocialFetcher()
        fetcher._cache.clear()

        with pytest.raises(httpx.HTTPError):
            fetcher._fetch_archive()


def test_retries_on_timeout(sample_archive_response):
    with patch("src.data.truth_social.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_archive_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.side_effect = [
            httpx.TimeoutException("timeout"),
            mock_response,
        ]
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = TruthSocialFetcher()
        fetcher._cache.clear()

        data = fetcher._fetch_archive()

        assert len(data) == 2
        assert mock_client.get.call_count == 2


def test_repr():
    fetcher = TruthSocialFetcher()
    assert "TruthSocialFetcher" in repr(fetcher)


def test_cache_key():
    fetcher = TruthSocialFetcher()
    key1 = fetcher._cache_key("test", "a", "b")
    key2 = fetcher._cache_key("test", "a", "b")
    key3 = fetcher._cache_key("test", "a", "c")

    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 32


def test_parse_datetime():
    fetcher = TruthSocialFetcher()

    # Standard ISO format
    dt1 = fetcher._parse_datetime("2025-01-15T10:30:00+00:00")
    assert dt1.year == 2025
    assert dt1.month == 1
    assert dt1.day == 15

    # With Z suffix
    dt2 = fetcher._parse_datetime("2025-01-15T10:30:00Z")
    assert dt2.year == 2025

    # With milliseconds
    dt3 = fetcher._parse_datetime("2025-01-15T10:30:00.123Z")
    assert dt3.year == 2025


def test_clear_cache():
    fetcher = TruthSocialFetcher()
    fetcher._cache.set("test_key", "test_value")
    fetcher.clear_cache()
    assert fetcher._cache.get("test_key") is None
