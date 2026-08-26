"""Automate creating Trello lists and cards for the COMP6002 project."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPOSITORY_ROOT / "docs" / "trello_tasks.csv"

LIST_ORDER = [
    "Done",
    "Review / Testing",
    "In Progress",
    "Ready",
    "Backlog",
]


def _trello_request(
    endpoint: str,
    params: dict[str, str],
    method: str = "POST",
) -> dict[str, object]:
    query = urllib.parse.urlencode(params)
    url = f"https://api.trello.com/1/{endpoint}?{query}"
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8")
        sys.stderr.write(f"Trello API error ({error.code}): {error_body}\n")
        raise


def setup_trello_board(
    api_key: str,
    token: str,
    board_id: str | None = None,
    board_name: str = "COMP6002 - Confidence-Gated Neural-SAT SPECK",
) -> None:
    """Populate a Trello board with the project lists and cards from CSV."""

    auth = {"key": api_key, "token": token}

    if not board_id:
        print(f"Creating new Trello board: '{board_name}'...")
        board = _trello_request("boards", {**auth, "name": board_name, "defaultLists": "false"})
        board_id = str(board["id"])
        print(f"Created board with ID: {board_id}")
    else:
        print(f"Using existing Trello board ID: {board_id}")

    # Fetch existing lists
    existing_lists = _trello_request(f"boards/{board_id}/lists", auth, method="GET")
    list_map: dict[str, str] = {str(item["name"]): str(item["id"]) for item in existing_lists}

    # Ensure all required lists exist
    for list_name in LIST_ORDER:
        if list_name not in list_map:
            print(f"Creating list: '{list_name}'...")
            created = _trello_request(
                "lists",
                {**auth, "idBoard": board_id, "name": list_name, "pos": "bottom"},
            )
            list_map[list_name] = str(created["id"])

    # Load tasks from CSV
    if not CSV_PATH.exists():
        sys.stderr.write(f"CSV file not found at {CSV_PATH}\n")
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cards = list(reader)

    print(f"\nAdding {len(cards)} cards across lists...")
    for card in cards:
        target_list = card["List"]
        list_id = list_map.get(target_list)
        if not list_id:
            print(f"Warning: list '{target_list}' not found, skipping card: {card['Card Name']}")
            continue

        desc = (
            f"**Owner:** {card['Owner']} | **Reviewer:** {card['Reviewer']}\n"
            f"**Milestone:** {card['Milestone']} | **Due Sprint:** {card['Due Sprint']}\n\n"
            f"### Description\n{card['Description']}\n"
        )

        card_params = {
            **auth,
            "idList": list_id,
            "name": card["Card Name"],
            "desc": desc,
            "pos": "bottom",
        }
        _trello_request("cards", card_params)
        print(f"  + Added [{target_list}] {card['Card Name']}")

    print("\nBoard setup complete! Open your Trello board to start tracking.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically populate a Trello board for the project."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TRELLO_API_KEY"),
        help="Trello API key (or set TRELLO_API_KEY env var)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("TRELLO_TOKEN"),
        help="Trello API token (or set TRELLO_TOKEN env var)",
    )
    parser.add_argument(
        "--board-id",
        default=os.environ.get("TRELLO_BOARD_ID"),
        help="Existing Trello Board ID (optional; creates a new board if omitted)",
    )
    parser.add_argument(
        "--board-name",
        default="COMP6002 - Confidence-Gated Neural-SAT SPECK",
        help="Name of the new board to create if --board-id is not given",
    )

    args = parser.parse_args()

    if not args.api_key or not args.token:
        print("=" * 70)
        print("TRELLO AUTOMATION SCRIPT")
        print("=" * 70)
        print("To automatically create the board, lists, and 20 detailed cards:")
        print("1. Get your API Key at: https://trello.com/power-ups/admin")
        print("2. Generate a Token on the same page.")
        print("3. Run this command:")
        print("\n   python scripts/setup_trello.py --api-key YOUR_KEY --token YOUR_TOKEN\n")
        print("Alternatively, you can import 'docs/trello_tasks.csv' directly into Trello")
        print("or copy-paste list items in bulk.")
        print("=" * 70)
        sys.exit(0)

    setup_trello_board(
        api_key=args.api_key,
        token=args.token,
        board_id=args.board_id,
        board_name=args.board_name,
    )


if __name__ == "__main__":
    main()
