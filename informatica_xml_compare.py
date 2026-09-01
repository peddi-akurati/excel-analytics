#!/usr/bin/env python3
"""
Informatica XML Folder Comparator
---------------------------------
Compares two folders containing Informatica PowerCenter XML exports.

Reports:
1. Files added / removed / changed / unchanged
2. XML objects added / removed
3. Attribute changes
4. Text/value changes
5. Detailed logical object paths, using Informatica identifiers where possible

No third-party packages required.

Usage:
    python informatica_xml_compare.py \
        --old-folder /path/to/baseline \
        --new-folder /path/to/new \
        --output-folder /path/to/report

Optional:
    --ignore-attr TIMESTAMP
    --ignore-attr VERSIONNUMBER
    --ignore-text
    --case-insensitive-files
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Attributes that usually identify Informatica objects.
# The first applicable group is used to build a logical identity.
IDENTITY_GROUPS = [
    ("NAME",),
    ("FROMINSTANCE", "FROMFIELD", "TOINSTANCE", "TOFIELD"),  # CONNECTOR
    ("FROMINSTANCETYPE", "FROMINSTANCE", "FROMFIELD",
     "TOINSTANCETYPE", "TOINSTANCE", "TOFIELD"),
    ("INSTANCE_NAME",),
    ("INSTANCENAME",),
    ("FIELDNAME",),
    ("ATTRNAME",),
    ("TASKNAME",),
    ("SESSIONNAME",),
    ("WORKFLOWNAME",),
    ("REFOBJECTNAME",),
    ("DBDNAME",),
    ("SOURCENAME",),
    ("TARGETNAME",),
    ("TRANSFORMATIONNAME",),
    ("MAPPINGNAME",),
    ("WIDGETTYPE", "NAME"),
]

# Attributes that are often generated metadata and can be ignored if desired.
DEFAULT_VOLATILE_ATTRS = set()


@dataclass
class DiffRecord:
    file: str
    change_type: str
    object_type: str
    object_path: str
    attribute: str = ""
    old_value: str = ""
    new_value: str = ""
    details: str = ""


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def normalize_attr_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    return value.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def element_identity(elem: ET.Element) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    """
    Build a logical identity for an Informatica XML element.
    Uses known identifying attributes where possible.
    """
    tag = strip_namespace(elem.tag)
    attrs = elem.attrib

    for group in IDENTITY_GROUPS:
        if all(k in attrs for k in group):
            return tag, tuple((k, normalize_attr_value(attrs.get(k))) for k in group)

    # Informatica objects frequently use NAME plus TYPE/OBJECTTYPE.
    if "NAME" in attrs:
        keys = ["NAME"]
        for extra in ("TYPE", "OBJECTTYPE", "TRANSFORMATIONTYPE", "INSTANCETYPE"):
            if extra in attrs:
                keys.append(extra)
        return tag, tuple((k, normalize_attr_value(attrs.get(k))) for k in keys)

    # Stable fallback based on a small set of descriptive attributes.
    preferred = []
    for k in sorted(attrs):
        if k.upper() in {
            "TYPE", "OBJECTTYPE", "TRANSFORMATIONTYPE", "INSTANCE_TYPE",
            "DATATYPE", "PORTTYPE", "PRECISION", "SCALE"
        }:
            preferred.append((k, normalize_attr_value(attrs[k])))

    return tag, tuple(preferred)


def identity_label(elem: ET.Element) -> str:
    tag, parts = element_identity(elem)
    if parts:
        vals = ", ".join(f"{k}={v}" for k, v in parts)
        return f"{tag}[{vals}]"
    return tag


def make_child_buckets(parent: ET.Element):
    """
    Bucket children by logical identity.
    Duplicates with the same identity are retained in document order.
    """
    buckets = defaultdict(list)
    for child in list(parent):
        buckets[element_identity(child)].append(child)
    return buckets


def attrs_for_compare(elem: ET.Element, ignore_attrs: set[str]) -> Dict[str, str]:
    out = {}
    for k, v in elem.attrib.items():
        if k.upper() in ignore_attrs:
            continue
        out[k] = normalize_attr_value(v)
    return out


def node_snapshot(elem: ET.Element, ignore_attrs: set[str]) -> str:
    attrs = attrs_for_compare(elem, ignore_attrs)
    return json.dumps(
        {
            "tag": strip_namespace(elem.tag),
            "attrs": attrs,
            "text": clean_text(elem.text),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def compare_elements(
    old_elem: ET.Element,
    new_elem: ET.Element,
    file_name: str,
    path: str,
    diffs: List[DiffRecord],
    ignore_attrs: set[str],
    ignore_text: bool,
):
    old_tag = strip_namespace(old_elem.tag)
    new_tag = strip_namespace(new_elem.tag)

    if old_tag != new_tag:
        diffs.append(
            DiffRecord(
                file=file_name,
                change_type="OBJECT_TYPE_CHANGED",
                object_type=f"{old_tag}->{new_tag}",
                object_path=path,
                old_value=old_tag,
                new_value=new_tag,
            )
        )
        return

    # Compare attributes
    old_attrs = attrs_for_compare(old_elem, ignore_attrs)
    new_attrs = attrs_for_compare(new_elem, ignore_attrs)

    for attr in sorted(set(old_attrs) | set(new_attrs)):
        old_val = old_attrs.get(attr)
        new_val = new_attrs.get(attr)

        if attr not in old_attrs:
            diffs.append(
                DiffRecord(
                    file=file_name,
                    change_type="ATTRIBUTE_ADDED",
                    object_type=old_tag,
                    object_path=path,
                    attribute=attr,
                    old_value="",
                    new_value=new_val or "",
                )
            )
        elif attr not in new_attrs:
            diffs.append(
                DiffRecord(
                    file=file_name,
                    change_type="ATTRIBUTE_REMOVED",
                    object_type=old_tag,
                    object_path=path,
                    attribute=attr,
                    old_value=old_val or "",
                    new_value="",
                )
            )
        elif old_val != new_val:
            diffs.append(
                DiffRecord(
                    file=file_name,
                    change_type="ATTRIBUTE_CHANGED",
                    object_type=old_tag,
                    object_path=path,
                    attribute=attr,
                    old_value=old_val or "",
                    new_value=new_val or "",
                )
            )

    # Compare direct text
    if not ignore_text:
        old_text = clean_text(old_elem.text)
        new_text = clean_text(new_elem.text)
        if old_text != new_text:
            diffs.append(
                DiffRecord(
                    file=file_name,
                    change_type="TEXT_CHANGED",
                    object_type=old_tag,
                    object_path=path,
                    attribute="#text",
                    old_value=old_text,
                    new_value=new_text,
                )
            )

    # Compare children logically, not only positionally.
    old_buckets = make_child_buckets(old_elem)
    new_buckets = make_child_buckets(new_elem)

    all_keys = sorted(
        set(old_buckets) | set(new_buckets),
        key=lambda x: (x[0], str(x[1]))
    )

    for key in all_keys:
        old_list = old_buckets.get(key, [])
        new_list = new_buckets.get(key, [])
        max_len = max(len(old_list), len(new_list))

        for i in range(max_len):
            old_child = old_list[i] if i < len(old_list) else None
            new_child = new_list[i] if i < len(new_list) else None
            occurrence = f"#{i+1}" if max_len > 1 else ""

            reference = old_child if old_child is not None else new_child
            child_label = identity_label(reference)
            child_path = f"{path}/{child_label}{occurrence}"

            if old_child is None:
                diffs.append(
                    DiffRecord(
                        file=file_name,
                        change_type="OBJECT_ADDED",
                        object_type=strip_namespace(new_child.tag),
                        object_path=child_path,
                        new_value=node_snapshot(new_child, ignore_attrs),
                        details="Object exists only in NEW folder",
                    )
                )
                # Also capture descendants as added for better detail
                add_descendants(
                    new_child, file_name, child_path, diffs,
                    "OBJECT_ADDED", ignore_attrs
                )
            elif new_child is None:
                diffs.append(
                    DiffRecord(
                        file=file_name,
                        change_type="OBJECT_REMOVED",
                        object_type=strip_namespace(old_child.tag),
                        object_path=child_path,
                        old_value=node_snapshot(old_child, ignore_attrs),
                        details="Object exists only in OLD folder",
                    )
                )
                add_descendants(
                    old_child, file_name, child_path, diffs,
                    "OBJECT_REMOVED", ignore_attrs
                )
            else:
                compare_elements(
                    old_child,
                    new_child,
                    file_name,
                    child_path,
                    diffs,
                    ignore_attrs,
                    ignore_text,
                )


def add_descendants(
    elem: ET.Element,
    file_name: str,
    path: str,
    diffs: List[DiffRecord],
    change_type: str,
    ignore_attrs: set[str],
):
    for child in list(elem):
        child_path = f"{path}/{identity_label(child)}"
        snap = node_snapshot(child, ignore_attrs)
        if change_type == "OBJECT_ADDED":
            diffs.append(
                DiffRecord(
                    file=file_name,
                    change_type=change_type,
                    object_type=strip_namespace(child.tag),
                    object_path=child_path,
                    new_value=snap,
                    details="Descendant of added object",
                )
            )
        else:
            diffs.append(
                DiffRecord(
                    file=file_name,
                    change_type=change_type,
                    object_type=strip_namespace(child.tag),
                    object_path=child_path,
                    old_value=snap,
                    details="Descendant of removed object",
                )
            )
        add_descendants(
            child, file_name, child_path, diffs, change_type, ignore_attrs
        )


def parse_xml(path: Path) -> ET.Element:
    try:
        tree = ET.parse(path)
        return tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {path}: {e}") from e


def list_xml_files(folder: Path, case_insensitive: bool) -> Dict[str, Path]:
    files = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".xml":
            rel = p.relative_to(folder).as_posix()
            key = rel.lower() if case_insensitive else rel
            files[key] = p
    return files


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compare_folders(
    old_folder: Path,
    new_folder: Path,
    output_folder: Path,
    ignore_attrs: set[str],
    ignore_text: bool,
    case_insensitive_files: bool,
):
    output_folder.mkdir(parents=True, exist_ok=True)

    old_files = list_xml_files(old_folder, case_insensitive_files)
    new_files = list_xml_files(new_folder, case_insensitive_files)

    all_keys = sorted(set(old_files) | set(new_files))
    file_summary = []
    all_diffs: List[DiffRecord] = []

    for key in all_keys:
        old_path = old_files.get(key)
        new_path = new_files.get(key)

        display_name = (
            old_path.relative_to(old_folder).as_posix()
            if old_path else new_path.relative_to(new_folder).as_posix()
        )

        if old_path is None:
            file_summary.append({
                "file": display_name,
                "status": "FILE_ADDED",
                "change_count": 1,
                "old_sha256": "",
                "new_sha256": sha256_file(new_path),
                "error": "",
            })
            all_diffs.append(
                DiffRecord(
                    file=display_name,
                    change_type="FILE_ADDED",
                    object_type="FILE",
                    object_path=display_name,
                    new_value=str(new_path),
                    details="XML file exists only in NEW folder",
                )
            )
            continue

        if new_path is None:
            file_summary.append({
                "file": display_name,
                "status": "FILE_REMOVED",
                "change_count": 1,
                "old_sha256": sha256_file(old_path),
                "new_sha256": "",
                "error": "",
            })
            all_diffs.append(
                DiffRecord(
                    file=display_name,
                    change_type="FILE_REMOVED",
                    object_type="FILE",
                    object_path=display_name,
                    old_value=str(old_path),
                    details="XML file exists only in OLD folder",
                )
            )
            continue

        old_hash = sha256_file(old_path)
        new_hash = sha256_file(new_path)

        try:
            old_root = parse_xml(old_path)
            new_root = parse_xml(new_path)

            file_diffs: List[DiffRecord] = []
            root_path = identity_label(old_root)

            compare_elements(
                old_root,
                new_root,
                display_name,
                root_path,
                file_diffs,
                ignore_attrs,
                ignore_text,
            )

            all_diffs.extend(file_diffs)

            status = "UNCHANGED" if not file_diffs else "CHANGED"
            file_summary.append({
                "file": display_name,
                "status": status,
                "change_count": len(file_diffs),
                "old_sha256": old_hash,
                "new_sha256": new_hash,
                "error": "",
            })

        except Exception as e:
            file_summary.append({
                "file": display_name,
                "status": "ERROR",
                "change_count": 0,
                "old_sha256": old_hash,
                "new_sha256": new_hash,
                "error": str(e),
            })
            all_diffs.append(
                DiffRecord(
                    file=display_name,
                    change_type="COMPARE_ERROR",
                    object_type="FILE",
                    object_path=display_name,
                    details=str(e),
                )
            )

    # Summary by change type
    change_counts = Counter(d.change_type for d in all_diffs)
    object_counts = Counter(d.object_type for d in all_diffs)

    detailed_path = output_folder / "informatica_xml_detailed_changes.csv"
    file_summary_path = output_folder / "informatica_xml_file_summary.csv"
    change_summary_path = output_folder / "informatica_xml_change_summary.csv"
    object_summary_path = output_folder / "informatica_xml_object_summary.csv"
    json_path = output_folder / "informatica_xml_detailed_changes.json"

    diff_fields = [
        "file", "change_type", "object_type", "object_path",
        "attribute", "old_value", "new_value", "details"
    ]
    write_csv(detailed_path, (asdict(d) for d in all_diffs), diff_fields)

    write_csv(
        file_summary_path,
        file_summary,
        ["file", "status", "change_count", "old_sha256", "new_sha256", "error"],
    )

    write_csv(
        change_summary_path,
        (
            {"change_type": k, "count": v}
            for k, v in sorted(change_counts.items())
        ),
        ["change_type", "count"],
    )

    write_csv(
        object_summary_path,
        (
            {"object_type": k, "count": v}
            for k, v in sorted(object_counts.items())
        ),
        ["object_type", "count"],
    )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(d) for d in all_diffs],
            f,
            ensure_ascii=False,
            indent=2,
        )

    changed_files = sum(1 for x in file_summary if x["status"] == "CHANGED")
    added_files = sum(1 for x in file_summary if x["status"] == "FILE_ADDED")
    removed_files = sum(1 for x in file_summary if x["status"] == "FILE_REMOVED")
    unchanged_files = sum(1 for x in file_summary if x["status"] == "UNCHANGED")
    error_files = sum(1 for x in file_summary if x["status"] == "ERROR")

    print("=" * 72)
    print("INFORMATICA XML FOLDER COMPARISON")
    print("=" * 72)
    print(f"OLD folder       : {old_folder}")
    print(f"NEW folder       : {new_folder}")
    print(f"Files compared   : {len(all_keys)}")
    print(f"Changed files    : {changed_files}")
    print(f"Added files      : {added_files}")
    print(f"Removed files    : {removed_files}")
    print(f"Unchanged files  : {unchanged_files}")
    print(f"Errors           : {error_files}")
    print(f"Detailed changes : {len(all_diffs)}")
    print()
    print("Change counts:")
    for k, v in sorted(change_counts.items()):
        print(f"  {k:<24} {v}")
    print()
    print("Reports:")
    print(f"  {file_summary_path}")
    print(f"  {detailed_path}")
    print(f"  {change_summary_path}")
    print(f"  {object_summary_path}")
    print(f"  {json_path}")
    print("=" * 72)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Compare two folders of Informatica XML exports."
    )
    p.add_argument(
        "--old-folder", required=True,
        help="Baseline/old Informatica XML folder"
    )
    p.add_argument(
        "--new-folder", required=True,
        help="New Informatica XML folder"
    )
    p.add_argument(
        "--output-folder", default="./informatica_xml_compare_report",
        help="Folder for CSV/JSON reports"
    )
    p.add_argument(
        "--ignore-attr",
        action="append",
        default=[],
        help="XML attribute to ignore. Can be specified multiple times."
    )
    p.add_argument(
        "--ignore-text",
        action="store_true",
        help="Ignore direct XML text-node differences."
    )
    p.add_argument(
        "--case-insensitive-files",
        action="store_true",
        help="Match relative XML file paths case-insensitively."
    )
    return p


def main():
    args = build_arg_parser().parse_args()

    old_folder = Path(args.old_folder).expanduser().resolve()
    new_folder = Path(args.new_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()

    if not old_folder.is_dir():
        print(f"ERROR: OLD folder does not exist: {old_folder}", file=sys.stderr)
        return 2

    if not new_folder.is_dir():
        print(f"ERROR: NEW folder does not exist: {new_folder}", file=sys.stderr)
        return 2

    ignore_attrs = {x.upper() for x in DEFAULT_VOLATILE_ATTRS}
    ignore_attrs.update(x.upper() for x in args.ignore_attr)

    compare_folders(
        old_folder=old_folder,
        new_folder=new_folder,
        output_folder=output_folder,
        ignore_attrs=ignore_attrs,
        ignore_text=args.ignore_text,
        case_insensitive_files=args.case_insensitive_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
