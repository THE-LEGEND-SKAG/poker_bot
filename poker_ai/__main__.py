"""``python -m poker_ai`` starts the web app on http://127.0.0.1:8000."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Poker AI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("poker_ai.server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
