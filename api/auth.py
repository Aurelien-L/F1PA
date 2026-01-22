"""
F1PA API - Authentication

HTTP Basic Authentication for API security.
"""
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Security scheme
security = HTTPBasic()

# Credentials (in production, use environment variables or secure storage)
API_USERNAME = "f1pa"
API_PASSWORD = "f1pa"


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Verify HTTP Basic credentials.

    Args:
        credentials: HTTP Basic credentials from request

    Returns:
        Username if valid

    Raises:
        HTTPException 401 if invalid
    """
    # Use secrets.compare_digest to prevent timing attacks
    username_correct = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        API_USERNAME.encode("utf-8")
    )
    password_correct = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        API_PASSWORD.encode("utf-8")
    )

    if not (username_correct and password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


def get_current_user(username: str = Depends(verify_credentials)) -> str:
    """
    Get current authenticated user.

    This is the main dependency to use in endpoints.
    """
    return username
