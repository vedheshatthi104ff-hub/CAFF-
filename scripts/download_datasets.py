"""Report acquisition requirements without bypassing logins or dataset terms."""

from __future__ import annotations

import argparse

NOTES = {
    "gsv_cities": "The selected archive is supplied at D:\\archive in this workspace. It is joined from the official Dataframes/ and Images/ layout with scripts/prepare_gsv_manifest.py; the VPR-datasets-downloader does not include GSV-Cities. This helper does not download data.",
    "pitts250k": "The VPR-datasets-downloader README directs users to request permission from the authors. Obtain permission before downloading. No download was performed.",
    "msls": "Log in to Mapillary and review/accept the dataset license, terms, and confidentiality conditions yourself. No terms were accepted and no download was performed.",
    "amstertime": "Use the official AmsterTime project or its linked Kaggle listing; confirm current access terms and preserve official pairs. No download was performed.",
    "eynsham": "The VPR-datasets-downloader README documents a public rsync command. Check the current data terms and destination storage before using it. No download was performed.",
    "svox": "The VPR-datasets-downloader README documents a SVOX helper. Check the current data terms and destination storage before using it. No download was performed.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(NOTES), required=True)
    args = parser.parse_args()
    print(f"{args.dataset}: {NOTES[args.dataset]}")
    print("This helper is informational and never reports a download as successful.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

