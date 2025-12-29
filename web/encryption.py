"""
Token Encryption Utilities

Encrypts sensitive tokens (Twitch, YouTube OAuth) before storing in database.
"""
import os
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import TOKEN_ENCRYPTION_KEY, SECRET_KEY


def _get_or_create_key() -> bytes:
    """Get encryption key from env or derive from SECRET_KEY"""
    if TOKEN_ENCRYPTION_KEY:
        # Use provided key
        return TOKEN_ENCRYPTION_KEY.encode() if isinstance(TOKEN_ENCRYPTION_KEY, str) else TOKEN_ENCRYPTION_KEY

    # Derive key from SECRET_KEY using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"clipppy-token-salt",  # Static salt is fine since SECRET_KEY should be unique
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(SECRET_KEY.encode()))
    return key


# Initialize Fernet with the encryption key
_fernet = Fernet(_get_or_create_key())


def encrypt_token(token_data: dict | str) -> str:
    """
    Encrypt a token or token dict for storage.

    Args:
        token_data: Either a string token or dict of token data

    Returns:
        Encrypted string safe for database storage
    """
    if isinstance(token_data, dict):
        plaintext = json.dumps(token_data)
    else:
        plaintext = token_data

    encrypted = _fernet.encrypt(plaintext.encode())
    return encrypted.decode()  # Return as string for DB storage


def decrypt_token(encrypted_token: str) -> dict | str:
    """
    Decrypt a token from storage.

    Args:
        encrypted_token: The encrypted token string from database

    Returns:
        Original token data (dict if it was JSON, string otherwise)
    """
    if not encrypted_token:
        return None

    decrypted = _fernet.decrypt(encrypted_token.encode()).decode()

    # Try to parse as JSON
    try:
        return json.loads(decrypted)
    except json.JSONDecodeError:
        return decrypted


def mask_token(token: str, visible_chars: int = 4) -> str:
    """
    Mask a token for display (e.g., in admin panel).
    Shows first and last few characters only.
    """
    if not token or len(token) < visible_chars * 2:
        return "••••••••"

    return f"{token[:visible_chars]}••••••••{token[-visible_chars:]}"
