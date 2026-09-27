"""
Meta Graph API Connector for Facebook Comments
Fetches comments from a Facebook Page Post using Graph API v19.0 with pagination.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
import requests


class FacebookConnector:
    """Connector to interact with Meta Graph API v19.0 for post comments extraction."""

    BASE_URL = "https://graph.facebook.com/v19.0"

    def __init__(self, page_access_token: str):
        """
        Initialize connector with Meta Page Access Token.
        
        Args:
            page_access_token: Meta Page Access Token with 'pages_read_engagement' 
                               or 'pages_manage_posts' permissions.
        """
        self.access_token = page_access_token.strip() if page_access_token else ""

    @staticmethod
    def extract_post_id(input_str: str) -> str:
        """
        Extract Facebook Post ID from URL or raw post ID string.
        Examples:
            - "123456789012345_678901234567890" -> "123456789012345_678901234567890"
            - "https://www.facebook.com/page/posts/678901234567890" -> "678901234567890"
            - "https://www.facebook.com/permalink.php?story_fbid=678901234567890&id=12345" -> "678901234567890"
        """
        cleaned = input_str.strip()
        # Check if already a clean ID or page_post ID format
        if re.match(r"^\d+(_\d+)?$", cleaned):
            return cleaned

        # URL matching
        post_match = re.search(r"/posts/(?:pfbid[a-zA-Z0-9]+|(\d+))", cleaned)
        if post_match and post_match.group(1):
            return post_match.group(1)
        elif post_match:
            # pfbid format
            full_match = re.search(r"/posts/([a-zA-Z0-9]+)", cleaned)
            if full_match:
                return full_match.group(1)

        fbid_match = re.search(r"story_fbid=([a-zA-Z0-9]+)", cleaned)
        if fbid_match:
            return fbid_match.group(1)

        photo_match = re.search(r"/photos/[^/]+/(\d+)", cleaned)
        if photo_match:
            return photo_match.group(1)

        return cleaned

    def fetch_comments(
        self,
        post_id_or_url: str,
        max_comments: int = 100,
        filter_type: str = "toplevel"
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments for a Facebook Post with cursor pagination.
        
        Args:
            post_id_or_url: Post ID (e.g. "{page_id}_{post_id}") or full URL.
            max_comments: Maximum number of comments to retrieve.
            filter_type: 'toplevel' or 'stream'.
            
        Returns:
            List of standardized comment dicts:
            [
                {
                    'comment_id': str,
                    'platform': 'Facebook',
                    'author': str,
                    'author_avatar': str,
                    'comment': str,
                    'published_at': str,
                    'like_count': int,
                    'reply_count': int
                }, ...
            ]
        """
        if not self.access_token:
            raise ValueError("Facebook Page Access Token is required.")

        post_id = self.extract_post_id(post_id_or_url)
        if not post_id:
            raise ValueError("Invalid Facebook Post ID or URL.")

        comments: List[Dict[str, Any]] = []
        next_url: Optional[str] = f"{self.BASE_URL}/{post_id}/comments"

        params: Dict[str, Any] = {
            "access_token": self.access_token,
            "fields": "id,message,created_time,from,like_count,comment_count",
            "limit": min(100, max_comments),
            "filter": filter_type,
            "summary": "true"
        }

        try:
            while next_url and len(comments) < max_comments:
                if next_url == f"{self.BASE_URL}/{post_id}/comments":
                    response = requests.get(next_url, params=params, timeout=15)
                else:
                    response = requests.get(next_url, timeout=15)

                data = response.json()

                if "error" in data:
                    err = data["error"]
                    err_code = err.get("code")
                    err_subcode = err.get("error_subcode")
                    err_msg = err.get("message", "Unknown Meta Graph API error")
                    err_type = err.get("type", "")

                    if err_code == 190:
                        raise RuntimeError("Meta Access Token is invalid or has expired. Please regenerate your Page Access Token in Meta for Developers.")
                    elif err_code in (4, 17, 32):
                        raise RuntimeError(f"Meta Graph API Rate Limit exceeded ({err_msg}). Please try again shortly.")
                    elif err_code == 200:
                        raise RuntimeError(f"Permissions error: Your token lacks permission to read comments for this post. Ensure 'pages_read_engagement' permission is granted.")
                    elif err_code == 100:
                        raise RuntimeError(f"Graph API Error (100): {err_msg}. Verify that Post ID '{post_id}' is correct and accessible.")
                    else:
                        raise RuntimeError(f"Facebook Graph API Error [{err_type} - {err_code}]: {err_msg}")

                items = data.get("data", [])
                if not items:
                    break

                for item in items:
                    message = item.get("message", "")
                    if not message or not message.strip():
                        continue

                    author_obj = item.get("from", {})
                    author_name = author_obj.get("name", "Facebook User")
                    author_id = author_obj.get("id", "")
                    avatar_url = f"https://graph.facebook.com/v19.0/{author_id}/picture?type=square" if author_id else ""

                    comments.append({
                        "comment_id": item.get("id", ""),
                        "platform": "Facebook",
                        "author": author_name,
                        "author_avatar": avatar_url,
                        "comment": message.strip(),
                        "published_at": item.get("created_time", ""),
                        "like_count": item.get("like_count", 0),
                        "reply_count": item.get("comment_count", 0),
                    })

                    if len(comments) >= max_comments:
                        break

                paging = data.get("paging", {})
                next_url = paging.get("next")
                if not next_url:
                    break

            return comments

        except requests.exceptions.Timeout:
            raise RuntimeError("Request to Meta Graph API timed out. Please check your network connection.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error connecting to Facebook API: {str(e)}")
