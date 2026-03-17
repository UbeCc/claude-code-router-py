"""Entry point — load config and start the server."""

import argparse
import logging
import os
import sys

import uvicorn

from config import load_config
from server import app, set_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code Router (Python)")
    parser.add_argument("--config", default="config.json",
                        help="Path to config.json (default: config.json)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: 1)")
    parser.add_argument("--log-level", default=None,
                        choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        sys.exit(1)

    host      = args.host      or cfg.get("HOST", "0.0.0.0")
    port      = args.port      or cfg.get("PORT", 3456)
    log_level = args.log_level or cfg.get("LOG_LEVEL", "info")
    workers   = args.workers   or cfg.get("WORKERS", 1)

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Starting Claude Code Router on {host}:{port} (workers={workers})", flush=True)

    if workers == 1:
        # Single process: load config directly into the module global.
        set_config(cfg)
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    else:
        # Multi-process: each worker loads config via lifespan from CCR_CONFIG.
        os.environ["CCR_CONFIG"] = args.config
        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
        )


if __name__ == "__main__":
    main()
