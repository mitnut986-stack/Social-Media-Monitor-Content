"""
YouTube Data API v3 Connector
Fetches top-level comments and replies for a given video ID or URL with pagination.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class YouTubeConnector:
    """Connector to interact with YouTube Data API v3 for comment extraction."""

    def __init__(self, api_key: str):
        """
        Initialize connector with YouTube Data API v3 key.
        
        Args:
            api_key: Google Cloud API Key with YouTube Data API v3 enabled.
        """
        self.api_key = api_key.strip() if api_key else ""
        self._service = None

    def _get_service(self):
        """Lazy initialization of Google API client service."""
        if not self._service:
            if not self.api_key:
                raise ValueError("YouTube API Key is required. Please provide a valid key.")
            self._service = build("youtube", "v3", developerKey=self.api_key)
        return self._service

    @staticmethod
    def extract_video_id(input_str: str) -> str:
        """
        Extract 11-character YouTube video ID from URL or raw ID string.
        Supports:
            - https://www.youtube.com/watch?v=VIDEO_ID
            - https://youtu.be/VIDEO_ID
            - https://www.youtube.com/shorts/VIDEO_ID
            - Raw VIDEO_ID
        """
        cleaned = input_str.strip()
        # Patterns for YouTube URLs
        patterns = [
            r'(?:v=|\/v\/|embed\/|shorts\/|youtu\.be\/|\/e\/|watch\?v=|\&v=)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$'
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                return match.group(1)
        return cleaned

    def fetch_comments(
        self,
        video_id_or_url: str,
        max_comments: int = 100,
        order: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments for a YouTube video with pagination support.
        
        Args:
            video_id_or_url: YouTube Video ID or full URL.
            max_comments: Maximum number of comments to retrieve (default: 100).
            order: Sorting order ('relevance' or 'time').
            
        Returns:
            List of standardized comment dicts:
            [
                {
                    'comment_id': str,
                    'platform': 'YouTube',
                    'author': str,
                    'author_avatar': str,
                    'comment': str,
                    'published_at': str,
                    'like_count': int,
                    'reply_count': int
                }, ...
            ]
        """
        video_id = self.extract_video_id(video_id_or_url)
        if not video_id:
            raise ValueError("Invalid YouTube Video ID or URL.")

        service = self._get_service()
        comments: List[Dict[str, Any]] = []
        next_page_token: Optional[str] = None

        try:
            while len(comments) < max_comments:
                page_size = min(100, max_comments - len(comments))
                
                request_kwargs = {
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": page_size,
                    "order": order,
                    "textFormat": "plainText",
                }
                if next_page_token:
                    request_kwargs["pageToken"] = next_page_token

                response = service.commentThreads().list(**request_kwargs).execute()

                items = response.get("items", [])
                if not items:
                    break

                for item in items:
                    snippet = item.get("snippet", {})
                    top_comment = snippet.get("topLevelComment", {}).get("snippet", {})
                    
                    comment_text = top_comment.get("textOriginal", top_comment.get("textDisplay", ""))
                    if not comment_text or not comment_text.strip():
                        continue

                    comments.append({
                        "comment_id": item.get("id", ""),
                        "platform": "YouTube",
                        "author": top_comment.get("authorDisplayName", "Anonymous"),
                        "author_avatar": top_comment.get("authorProfileImageUrl", ""),
                        "comment": comment_text.strip(),
                        "published_at": top_comment.get("publishedAt", ""),
                        "like_count": top_comment.get("likeCount", 0),
                        "reply_count": snippet.get("totalReplyCount", 0),
                    })

                    if len(comments) >= max_comments:
                        break

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            return comments

        except HttpError as err:
            status_code = err.resp.status
            reason = err.error_details[0].get("reason", "") if err.error_details else str(err)
            message = err.error_details[0].get("message", "") if err.error_details else str(err)

            if status_code == 403:
                if "quotaExceeded" in reason:
                    raise RuntimeError("YouTube API Quota Exceeded for today. Please wait or check your Google Cloud Console quota.") from err
                elif "commentsDisabled" in reason:
                    raise RuntimeError("Comments are disabled for this YouTube video.") from err
                else:
                    raise RuntimeError(f"YouTube API Forbidden error (403): {message}") from err
            elif status_code == 404:
                raise RuntimeError(f"Video not found (404). Please verify that Video ID '{video_id}' exists and is public.") from err
            elif status_code == 400:
                raise RuntimeError(f"Bad Request (400): {message}. Please verify the Video ID format.") from err
            else:
                raise RuntimeError(f"YouTube API Error ({status_code}): {message}") from err
        except Exception as e:
            raise RuntimeError(f"Unexpected error fetching YouTube comments: {str(e)}") from e
