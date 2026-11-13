"""Turns a password and its salt into the form the database stores."""
import hashlib

from config import config


def hash_password(plain_password, password_salt):
    """PBKDF2 key of the password with the account's own salt, as hexadecimal text."""
    derived = hashlib.pbkdf2_hmac(config.PASSWORD_HASH_ALGORITHM, plain_password.encode("utf-8"),
                                  password_salt.encode("utf-8"), config.PASSWORD_HASH_ROUNDS)
    return derived.hex()