"""Integrations testleri."""

import os
from unittest.mock import patch, MagicMock

# Import error catch testi için önce HAS_GITHUB / HAS_NOTION flag'lerini manipüle edebiliriz.
from agentkit.tools.integrations.github import (
    github_get_issue,
    github_create_issue,
)
from agentkit.tools.integrations.notion import notion_read_page


def test_github_missing_token() -> None:
    """GitHub token yoksa hata dönmesi."""
    if "GITHUB_TOKEN" in os.environ:
        del os.environ["GITHUB_TOKEN"]
    res = github_get_issue.func(repo_name="test/test", issue_number=1)
    assert "çevre değişkeni bulunamadı" in res.lower()


@patch("agentkit.tools.integrations.github.Github")
def test_github_get_issue_success(MockGithub: MagicMock) -> None:
    """GitHub issue okuma başarısı."""
    os.environ["GITHUB_TOKEN"] = "fake"

    mock_issue = MagicMock()
    mock_issue.title = "Test Issue"
    mock_issue.state = "open"
    mock_issue.user.login = "testuser"
    mock_issue.body = "Test body"

    mock_repo = MagicMock()
    mock_repo.get_issue.return_value = mock_issue

    mock_instance = MockGithub.return_value
    mock_instance.get_repo.return_value = mock_repo

    res = github_get_issue.func(repo_name="test/test", issue_number=1)
    assert "Test Issue" in res
    assert "testuser" in res


@patch("agentkit.tools.integrations.github.Github")
def test_github_create_issue_success(MockGithub: MagicMock) -> None:
    """GitHub issue oluşturma başarısı."""
    os.environ["GITHUB_TOKEN"] = "fake"

    mock_issue = MagicMock()
    mock_issue.html_url = "http://github.com/test/1"

    mock_repo = MagicMock()
    mock_repo.create_issue.return_value = mock_issue

    mock_instance = MockGithub.return_value
    mock_instance.get_repo.return_value = mock_repo

    res = github_create_issue.func(repo_name="test/test", title="T", body="B")
    assert "başarıyla oluşturuldu" in res
    assert "http://github.com/test/1" in res


def test_notion_missing_token() -> None:
    """Notion token yoksa hata dönmesi."""
    if "NOTION_TOKEN" in os.environ:
        del os.environ["NOTION_TOKEN"]
    res = notion_read_page.func(page_id="123")
    assert "çevre değişkeni bulunamadı" in res.lower()


@patch("agentkit.tools.integrations.notion.Client")
def test_notion_read_page_success(MockClient: MagicMock) -> None:
    """Notion sayfa okuma başarısı."""
    os.environ["NOTION_TOKEN"] = "fake"

    mock_instance = MockClient.return_value
    mock_instance.blocks.children.list.return_value = {
        "results": [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Notion Test İçeriği"}]},
            }
        ]
    }

    res = notion_read_page.func(page_id="123")
    assert "Notion Test İçeriği" in res
