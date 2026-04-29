"""
GitHub entegrasyon araçları.
Kullanmak için `pip install "agentkit-ai[github]"` veya `pip install PyGithub` gereklidir.
"""

import os
from typing import Any
from agentkit.tools.decorator import tool
from loguru import logger

try:
    from github import Github

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False


def _get_github_client() -> Any:
    if not HAS_GITHUB:
        raise ImportError(
            "PyGithub kütüphanesi bulunamadı. Lütfen 'pip install PyGithub' komutunu çalıştırın."
        )

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN çevre değişkeni bulunamadı.")

    return Github(token)


@tool
def github_get_issue(repo_name: str, issue_number: int) -> str:
    """
    Belirtilen GitHub repository'sinden (örn: 'owner/repo') belirli bir issue'nun detaylarını getirir.
    """
    try:
        g = _get_github_client()
        repo = g.get_repo(repo_name)
        issue = repo.get_issue(number=issue_number)
        return f"Başlık: {issue.title}\nDurum: {issue.state}\nYazar: {issue.user.login}\nİçerik:\n{issue.body}"
    except Exception as e:
        logger.error(f"[GitHub Tool] Hata: {str(e)}")
        return f"Issue getirilemedi: {str(e)}"


@tool
def github_create_issue(repo_name: str, title: str, body: str) -> str:
    """
    Belirtilen GitHub repository'sine yeni bir issue oluşturur.
    """
    try:
        g = _get_github_client()
        repo = g.get_repo(repo_name)
        issue = repo.create_issue(title=title, body=body)
        return f"Issue başarıyla oluşturuldu! URL: {issue.html_url}"
    except Exception as e:
        logger.error(f"[GitHub Tool] Hata: {str(e)}")
        return f"Issue oluşturulamadı: {str(e)}"
