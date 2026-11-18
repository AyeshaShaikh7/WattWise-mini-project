"""
auth.py - Password hashing and login logic for the Smart Electricity Bill Analyzer.

This file never stores a password as readable text. Passwords are scrambled
(hashed) with PBKDF2-HMAC-SHA256 and a random salt. The hash and the salt
are both returned as hexadecimal TEXT, which is the format agreed with
database.py.

This is suitable for a college project. It is not a complete production
security system (no password reset, no limit on wrong attempts).
"""

import hashlib   # PBKDF2 password hashing (built into Python)
import hmac      # safe comparison of two hashes
import os        # os.urandom gives a random salt

ITERATIONS = 200_000        # how many times the scrambling is repeated
SALT_BYTES = 16             # size of the random salt (becomes 32 hex characters)
MIN_PASSWORD_LENGTH = 6     # used by login_or_create in Stage B
class InvalidInputError(ValueError):
    """Raised when the email or password typed by the user is not acceptable."""


class WrongPasswordError(Exception):
    """Raised when the email exists but the password is wrong."""

def hash_password(password: str) -> tuple[str, str]:
    """Scramble a password with a new random salt.

    Args:
        password: the password text typed by the user.

    Returns:
        (password_hash, salt), both as hex text. Same order as
        database.create_user(email, password_hash, salt).

    Raises:
        TypeError: if password is not text.
        ValueError: if password is empty.
    """
    if not isinstance(password, str):
        raise TypeError(f"password must be text, got {type(password).__name__}.")
    if password == "":
        raise ValueError("password cannot be empty.")

    salt = os.urandom(SALT_BYTES)    # fresh random bytes for THIS user
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS
    )
    return digest.hex(), salt.hex()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    """Check a typed password against a stored salt and hash.

    Returns True if it matches, False otherwise (including when the stored
    salt is not valid hex text).

    Raises:
        TypeError: if any argument is not text.
    """
    for name, value in (("password", password), ("salt", salt),
                        ("password_hash", password_hash)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be text, got {type(value).__name__}.")

    try:
        salt_bytes = bytes.fromhex(salt)    # hex text -> bytes
    except ValueError:
        return False                        # a broken salt can never match

    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt_bytes, ITERATIONS
    )
    # compare_digest avoids leaking information through comparison timing.
    return hmac.compare_digest(digest.hex().encode(), password_hash.encode())
def _clean_email(email) -> str:
    """Trim spaces, lowercase, and check the email looks valid.

    Raises:
        InvalidInputError: if the email is not text or does not look valid.
    """
    if not isinstance(email, str):
        raise InvalidInputError("Email must be text.")

    email = email.strip().lower()
    local, _, domain = email.partition("@")   # split at the first "@"
    if (not local or not domain or "." not in domain
            or "@" in domain or " " in email):
        raise InvalidInputError("Please enter a valid email address.")
    return email


def _check_password(password) -> None:
    """Raise InvalidInputError if the password is not text or too short."""
    if not isinstance(password, str):
        raise InvalidInputError("Password must be text.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise InvalidInputError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )


def login_or_create(email: str, password: str, db=None) -> dict:
    """Log in an existing user, or create a new account for a new email.

    Args:
        email: typed email (spaces and capital letters are ignored).
        password: typed password (at least 6 characters).
        db: the module/object providing get_user_by_email, create_user and
            DuplicateUserError. Leave it out to use the real database.py.
            Tests pass a fake database here.

    Returns:
        {"user_id": int, "email": str, "is_new": bool}

    Raises:
        InvalidInputError: bad email or password format.
        WrongPasswordError: the email exists but the password is wrong.
                            No new account is ever created in this case.
    """
    email = _clean_email(email)
    _check_password(password)

    if db is None:
        import database as db   # the real database module (imported only when needed)

    user = db.get_user_by_email(email)

    if user is None:
        # New email: create the account.
        password_hash, salt = hash_password(password)
        try:
            new_id = db.create_user(email, password_hash, salt)
            return {"user_id": new_id, "email": email, "is_new": True}
        except db.DuplicateUserError:
            # Someone created this email a moment ago. Treat it as a login.
            user = db.get_user_by_email(email)
            if user is None:
                raise

    # Existing user: the password must match. Never create a second account.
    if not verify_password(password, user["salt"], user["password_hash"]):
        raise WrongPasswordError("Incorrect password.")

    return {"user_id": user["id"], "email": email, "is_new": False}