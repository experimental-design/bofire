import argparse
import json
import sys
from typing import IO, Any, Dict, Iterator, Optional

from pydantic import TypeAdapter, ValidationError

from bofire.data_models.migration.api import migrate
from bofire.data_models.migration.errors import (
    MigrationError,
    UnrecoverablePayloadError,
)
from bofire.data_models.migration.version import BASELINE, SOURCE_PRE


def _validator_for(kind: str) -> Optional[TypeAdapter]:
    if kind == "surrogate":
        from bofire.data_models.surrogates.api import AnySurrogate

        return TypeAdapter(AnySurrogate)
    if kind == "strategy":
        from bofire.data_models.strategies.api import AnyStrategy

        return TypeAdapter(AnyStrategy)
    if kind == "domain":
        from bofire.data_models.domain.api import Domain

        return TypeAdapter(Domain)
    return None


def _read_record(line: str) -> dict:
    """Accept both bare JSON object lines and double-encoded JSON-string lines."""
    stripped = line.lstrip()
    if stripped.startswith('"'):
        return json.loads(json.loads(line))
    return json.loads(line)


def _iter_records(stream: IO[str]) -> Iterator[dict]:
    for line in stream:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        yield _read_record(line)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bofire.data_models.migration",
        description="Migrate legacy BoFire payloads to the current schema.",
    )
    parser.add_argument(
        "--kind",
        choices=["strategy", "surrogate", "domain"],
        required=True,
    )
    parser.add_argument("--source", default=SOURCE_PRE)
    parser.add_argument("--target", default=BASELINE)
    parser.add_argument(
        "--input", "-i", default="-", help="JSONL input (default: stdin)"
    )
    parser.add_argument(
        "--output", "-o", default="-", help="JSONL output (default: stdout)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each migrated record against the current schema.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Migrate and report but write no output."
    )
    parser.add_argument("--report", help="Write a JSON summary report to this path.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write failed records inline with an error annotation "
        "instead of routing them to a sidecar.",
    )

    args = parser.parse_args(argv)

    in_stream = sys.stdin if args.input == "-" else open(args.input, "r")
    out_stream = sys.stdout if args.output == "-" else open(args.output, "w")
    fail_stream = None
    fail_path = None
    if args.output != "-" and not args.dry_run and not args.continue_on_error:
        fail_path = args.output + ".failed.jsonl"

    validator = _validator_for(args.kind) if args.validate else None

    report: Dict[str, Any] = {
        "total": 0,
        "ok": 0,
        "unrecoverable": [],
        "validation_failed": [],
        "shape_histogram": {},
    }

    try:
        for lineno, payload in enumerate(_iter_records(in_stream), start=1):
            report["total"] += 1
            type_tag = payload.get("type", "<no-type>")
            report["shape_histogram"].setdefault(type_tag, 0)
            report["shape_histogram"][type_tag] += 1
            try:
                migrated = migrate(
                    payload, source=args.source, target=args.target, kind=args.kind
                )
                if validator is not None:
                    validator.validate_python(migrated)
                report["ok"] += 1
                if not args.dry_run:
                    out_stream.write(json.dumps(migrated) + "\n")
            except UnrecoverablePayloadError as e:
                report["unrecoverable"].append(
                    {"line": lineno, "type": e.payload_type, "reason": e.reason}
                )
                if args.continue_on_error and not args.dry_run:
                    out_stream.write(
                        json.dumps({"_migration_error": str(e), "_original": payload})
                        + "\n"
                    )
                elif fail_path and not args.dry_run:
                    if fail_stream is None:
                        fail_stream = open(fail_path, "w")
                    fail_stream.write(
                        json.dumps({"_migration_error": str(e), "_original": payload})
                        + "\n"
                    )
            except (ValidationError, MigrationError) as e:
                report["validation_failed"].append(
                    {"line": lineno, "type": type_tag, "error": str(e)[:500]}
                )
                if args.continue_on_error and not args.dry_run:
                    out_stream.write(
                        json.dumps(
                            {"_validation_error": str(e)[:500], "_original": payload}
                        )
                        + "\n"
                    )
                elif fail_path and not args.dry_run:
                    if fail_stream is None:
                        fail_stream = open(fail_path, "w")
                    fail_stream.write(
                        json.dumps(
                            {"_validation_error": str(e)[:500], "_original": payload}
                        )
                        + "\n"
                    )
    finally:
        if in_stream is not sys.stdin:
            in_stream.close()
        if out_stream is not sys.stdout:
            out_stream.close()
        if fail_stream is not None:
            fail_stream.close()

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    sys.stderr.write(
        f"migrated {report['ok']}/{report['total']} "
        f"(unrecoverable={len(report['unrecoverable'])}, "
        f"validation_failed={len(report['validation_failed'])})\n"
    )
    return 0 if len(report["validation_failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
