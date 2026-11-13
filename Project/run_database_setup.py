"""Creates the database, its tables and the accounts people sign in with.
Safe to run again: an account already present is left exactly as it is.

Run from the Project folder:  python3 run_database_setup.py
"""
import os
import secrets

from config import config


def main():
    os.makedirs(config.DATABASE_DIRECTORY, exist_ok=True)
    from access.password_security import hash_password
    from persistence.database import database_engine, session_factory
    from persistence.models import Base, UserAccount
    from persistence.repository import fetch_user_account, insert_user_account

    Base.metadata.create_all(database_engine)
    print(f"database: {config.APPLICATION_DATABASE_FILE}")
    print(f"  tables: {', '.join(sorted(Base.metadata.tables))}")
    created, present = 0, 0
    with session_factory() as session:
        for account in config.SEED_USER_ACCOUNTS:
            if fetch_user_account(session, UserAccount.username == account["username"]):
                present += 1
                continue
            salt = secrets.token_bytes(config.PASSWORD_SALT_BYTES).hex()
            insert_user_account(session, {
                "username": account["username"], "display_name": account["display_name"],
                "role": account["role"], "password_salt": salt,
                "password_hash": hash_password(account["password"], salt)})
            created += 1
            print(f"  account created  {account['username']:<8} {account['display_name']:<14} "
                  f"{account['role']}")
        session.commit()
    print(f"  accounts created: {created}, already present: {present}")


if __name__ == "__main__":
    main()