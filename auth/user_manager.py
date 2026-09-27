"""
User Authentication & Session Management Module
Handles user registration, password hashing (PBKDF2-HMAC-SHA256 / bcrypt),
verification, and credential persistence in data/users.json.
"""

from __future__ import annotations
import json
import os
import re
import secrets
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Check optional bcrypt availability
try:
    import bcrypt  # type: ignore
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False


class UserManager:
    """Manages user persistence, secure password hashing, and authentication."""

    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "users.json"
    )

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        self._ensure_storage_initialized()

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Securely hash a password with cryptographic salt.
        Uses bcrypt if available; otherwise uses PBKDF2-HMAC-SHA256 with 120,000 iterations.
        """
        if HAS_BCRYPT:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
            return f"bcrypt:{hashed.decode('utf-8')}"
        else:
            salt = secrets.token_hex(16)
            key = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                iterations=120000
            )
            return f"pbkdf2:{salt}:{key.hex()}"

    @staticmethod
    def verify_password(password: str, hashed_str: str) -> bool:
        """
        Verifies a plaintext password against the stored salt & hash.
        Supports both bcrypt and PBKDF2 formats.
        """
        if not hashed_str:
            return False

        try:
            if hashed_str.startswith("bcrypt:") and HAS_BCRYPT:
                stored_bcrypt_hash = hashed_str.split(":", 1)[1].encode("utf-8")
                return bcrypt.checkpw(password.encode("utf-8"), stored_bcrypt_hash)

            elif hashed_str.startswith("pbkdf2:"):
                _, salt, stored_key_hex = hashed_str.split(":", 2)
                computed_key = hashlib.pbkdf2_hmac(
                    "sha256",
                    password.encode("utf-8"),
                    salt.encode("utf-8"),
                    iterations=120000
                )
                return secrets.compare_digest(computed_key.hex(), stored_key_hex)

            elif hashed_str.startswith("$2") and HAS_BCRYPT:
                return bcrypt.checkpw(password.encode("utf-8"), hashed_str.encode("utf-8"))

            return False
        except Exception:
            return False


    def _ensure_storage_initialized(self):
        """Initializes users.json if it doesn't exist, pre-seeding a default demo account."""
        if not os.path.exists(self.data_path):
            initial_users: Dict[str, Dict[str, Any]] = {}
            # Pre-seed demo administrator account
            demo_username = "admin@khmer.ai"
            initial_users[demo_username.lower()] = {
                "full_name": "Sokha Chea (Admin)",
                "username": demo_username,
                "password_hash": self.hash_password("Password123!"),
                "role": "Administrator",
                "created_at": datetime.now().isoformat()
            }
            self._save_users(initial_users)

    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Loads users dictionary from JSON storage."""
        if not os.path.exists(self.data_path):
            return {}
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_users(self, users: Dict[str, Dict[str, Any]]):
        """Saves users dictionary to JSON storage atomically."""
        temp_path = f"{self.data_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.data_path)

    def register_user(
        self,
        full_name: str,
        username: str,
        password: str,
        confirm_password: str
    ) -> Tuple[bool, str]:
        """
        Registers a new user with validation:
        - Full Name must be >= 2 characters.
        - Username / Email must be non-empty and unique.
        - Password must be >= 6 characters.
        - Password and confirm_password must match.
        """
        full_name = full_name.strip()
        username = username.strip()

        if not full_name or len(full_name) < 2:
            return False, "Full Name must be at least 2 characters long."

        if not username or len(username) < 3:
            return False, "Email or Username must be at least 3 characters long."

        # Validate basic email or alphanumeric format
        if not re.match(r"^[\w\.\-\+@]+$", username):
            return False, "Username contains invalid characters. Use letters, numbers, '.', '-', or '@'."

        if len(password) < 6:
            return False, "Password must be at least 6 characters long."

        if password != confirm_password:
            return False, "Passwords do not match. Please verify your password confirmation."

        users = self._load_users()
        user_key = username.lower()

        if user_key in users:
            return False, f"Account '{username}' already exists. Please login instead."

        # Store user with hashed password
        users[user_key] = {
            "full_name": full_name,
            "username": username,
            "password_hash": self.hash_password(password),
            "role": "User",
            "created_at": datetime.now().isoformat()
        }
        self._save_users(users)
        return True, "Account registered successfully! You can now log in."

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Any]:
        """
        Authenticates a user by username/email and plaintext password.
        Returns:
            (True, user_dict) if authentication succeeds.
            (False, error_message_str) if authentication fails.
        """
        username = username.strip()
        if not username or not password:
            return False, "Please enter both username/email and password."

        users = self._load_users()
        user_key = username.lower()

        user_record = users.get(user_key)
        if not user_record:
            return False, "Invalid username or password."

        stored_hash = user_record.get("password_hash", "")
        if self.verify_password(password, stored_hash):
            # Return safe copy without password hash
            safe_user = dict(user_record)
            safe_user.pop("password_hash", None)
            return True, safe_user
        else:
            return False, "Invalid username or password."

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get safe user profile by username."""
        users = self._load_users()
        user = users.get(username.lower())
        if user:
            safe = dict(user)
            safe.pop("password_hash", None)
            return safe
        return None
