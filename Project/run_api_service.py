"""Starts the FairShare API service.

Run from the Project folder:  python3 run_api_service.py
"""
import uvicorn

from config import config


def main():
    uvicorn.run("api.routes:service_application", host=config.API_HOST, port=config.API_PORT,
                reload=False)


if __name__ == "__main__":
    main()