#!/usr/bin/env python3
"""
Extraction script: loads a JSON file of unstructured elements, then:
1. Normalizes every element into a canonical internal schema
2. Builds a document outline (grouped by Title) with random section_id
3. Builds logical blocks (merged narrative/list, atomic tables, image blocks)
"""

import argparse
import asyncio
import json
import secrets
import sys
from collections import Counter
from pathlib import Path

from ai.main import generateImageCaption, generateImageSummary, generatTableSummary


# Types that start a new narrative merge
BLOCK_BREAK_TYPES = {"ListItem", "Table", "Image"}

# Types to ignore
OMIT_TYPES = {"Header", "Footer", "UncategorizedText"}


def points_to_bbox(points):
    """Convert coordinates.points (list of [x,y]) to bbox {x1, y1, x2, y2}."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


def union_bbox(bboxes: list) -> dict | None:
    """Return a single bbox that spans all given bboxes, or None if none valid."""
    valid = [b for b in bboxes if b and isinstance(b, dict)]
    if not valid:
        return None
    return {
        "x1": min(b["x1"] for b in valid),
        "y1": min(b["y1"] for b in valid),
        "x2": max(b["x2"] for b in valid),
        "y2": max(b["y2"] for b in valid),
    }


def normalize_element(raw: dict, sequence: int, file_id: str, file_name: str) -> dict:
    """Map one raw element to canonical schema."""
    meta = raw.get("metadata") or {}
    coords = meta.get("coordinates") or {}
    points = coords.get("points") or []
    page = meta.get("page_number") or meta.get("page_name") or 1
    if isinstance(page, str):
        page = 1

    return {
        "element_id": raw.get("element_id", ""),
        "type": raw.get("type", "NarrativeText"),
        "text": raw.get("text", ""),
        "page_number": page,
        "bbox": points_to_bbox(points),
        "file_id": file_id,
        "file_name": file_name,
        "sequence": sequence,
    }


def resolve_file_identifiers(elements: list, json_path: str) -> tuple[str, str]:
    """Get file_id and file_name from first element or JSON path."""
    file_name = ""
    for el in elements:
        meta = el.get("metadata") or {}
        if meta.get("filename"):
            file_name = meta["filename"]
            break
    if not file_name and json_path:
        file_name = Path(json_path).stem.replace(".json", "") or Path(json_path).name
    file_id = file_name
    return file_id, file_name


def normalize_elements(raw_elements: list, file_id: str, file_name: str) -> list[dict]:
    """Normalize all elements to canonical schema (sequence preserved)."""
    return [
        normalize_element(el, seq, file_id, file_name)
        for seq, el in enumerate(raw_elements, start=1)
    ]


def random_section_id(prefix: str = "section_") -> str:
    """Generate a random section_id starting with the given prefix."""
    return f"{prefix}{secrets.token_hex(16)}"


def build_outline(normalized: list[dict]) -> list[dict]:
    """
    Build document outline: group elements by Title.
    Each section gets a random section_id starting with 'section_'.
    Elements before the first Title go into an initial section.
    """
    sections = []
    current_section_id = random_section_id()
    current_title = "Initial Section"
    current_elements = []

    for el in normalized:
        if el["type"] == "Title":
            if current_elements:
                sections.append({
                    "section_id": current_section_id,
                    "title": current_title,
                    "elements": current_elements.copy(),
                })
            current_section_id = random_section_id()
            current_title = el["text"] or "(untitled)"
            current_elements = [el]
        else:
            current_elements.append(el)

    if current_elements:
        sections.append({
            "section_id": current_section_id,
            "title": current_title,
            "elements": current_elements,
        })

    return sections


def build_logical_blocks(sections: list[dict], raw_by_element_id: dict) -> list[dict]:
    """
    Build logical blocks per section:
    - Merge adjacent NarrativeText (stop at list/table/image)
    - Merge adjacent ListItem
    - Tables: atomic with raw HTML and summary
    - Images: metadata, base64, description, caption
    """
    total_images = sum(
        1 for sec in sections for el in sec["elements"] if el.get("type") == "Image"
    )
    total_tables = sum(
        1 for sec in sections for el in sec["elements"] if el.get("type") == "Table"
    )
    image_counter = 0
    table_counter = 0
    result_sections = []

    for sec in sections:
        blocks = []
        elements = sec["elements"]
        section_id = sec["section_id"]
        section_title = sec["title"]
        i = 0

        while i < len(elements):
            el = elements[i]
            t = el["type"]

            if t == "NarrativeText":
                merged_texts = [el["text"]]
                merged_ids = [el["element_id"]]
                merged_bboxes = [el.get("bbox")]
                j = i + 1
                while j < len(elements) and elements[j]["type"] == "NarrativeText":
                    merged_texts.append(elements[j]["text"])
                    merged_ids.append(elements[j]["element_id"])
                    merged_bboxes.append(elements[j].get("bbox"))
                    j += 1
                blocks.append({
                    "block_type": "narrative",
                    "element_ids": merged_ids,
                    "text": "\n\n".join(merged_texts),
                    "merged_count": len(merged_ids),
                    "bbox": union_bbox(merged_bboxes),
                    "section_id": section_id,
                    "section_title": section_title,
                })
                i = j

            elif t == "ListItem":
                merged_texts = [el["text"]]
                merged_ids = [el["element_id"]]
                merged_bboxes = [el.get("bbox")]
                j = i + 1
                while j < len(elements) and elements[j]["type"] == "ListItem":
                    merged_texts.append(elements[j]["text"])
                    merged_ids.append(elements[j]["element_id"])
                    merged_bboxes.append(elements[j].get("bbox"))
                    j += 1
                blocks.append({
                    "block_type": "list",
                    "element_ids": merged_ids,
                    "text": "\n".join(merged_texts),
                    "merged_count": len(merged_ids),
                    "bbox": union_bbox(merged_bboxes),
                    "section_id": section_id,
                    "section_title": section_title,
                })
                i = j

            elif t == "Table":
                raw = raw_by_element_id.get(el["element_id"]) or {}
                meta = raw.get("metadata") or {}
                raw_html = meta.get("text_as_html", "")
                summary = (el["text"] or "")[:500]
                if len(el.get("text") or "") > 500:
                    summary = summary.rstrip() + "…"
                if raw_html:
                    table_counter += 1
                    print(
                        f"Generating summary for table {table_counter}/{total_tables}",
                        file=sys.stderr,
                    )
                    try:
                        summary = asyncio.run(generatTableSummary([raw_html]))
                    except Exception:
                        pass  # keep fallback summary
                blocks.append({
                    "block_type": "table",
                    "element_id": el["element_id"],
                    "raw_html": raw_html,
                    "summary": summary,
                    "bbox": el.get("bbox"),
                    "section_id": section_id,
                    "section_title": section_title,
                })
                i += 1

            elif t == "Image":
                raw = raw_by_element_id.get(el["element_id"]) or {}
                meta = raw.get("metadata")
                base64_data = meta.get('image_base64')
                caption = el.get("text")
                description = caption or "(image)"
                if base64_data:
                    image_counter += 1
                    print(
                        f"Generating Summary & Caption for image {image_counter}/{total_images}",
                        file=sys.stderr,
                    )
                    b64_payload = base64_data.split(",", 1)[-1] if "," in base64_data else base64_data
                    try:
                        description = asyncio.run(generateImageSummary(b64_payload))
                        caption = asyncio.run(generateImageCaption(b64_payload))
                    except Exception:
                        pass  # keep fallback description/caption
                blocks.append({
                    "block_type": "image",
                    "element_id": el["element_id"],
                    "element_metadata": {k: v for k, v in el.items() if k != "text"},
                    "base64": base64_data,
                    "description": description,
                    "caption": caption,
                    "section_id": section_id,
                    "section_title": section_title,
                })
                i += 1
            elif t == "Title":
                blocks.append({
                    "block_type": "Title",
                    "element_id": el["element_id"],
                    "text": el.get("text", ""),
                    "bbox": el.get("bbox"),
                    "section_id": section_id,
                    "section_title": section_title,
                })
                i += 1
            else:
                blocks.append({
                    "block_type": "other",
                    "element_id": el["element_id"],
                    "text": el.get("text", ""),
                    "bbox": el.get("bbox"),
                    "section_id": section_id,
                    "section_title": section_title,
                })
                i += 1

        result_sections.append({
            "section_id": section_id,
            "title": section_title,
            "blocks": blocks,
        })

    return result_sections


def build_section_snippet(sec: dict, max_chars: int = 400) -> str:
    parts = [sec["title"] or ""]
    for blk in sec.get("blocks", [])[:10]: 
        bt = blk.get("block_type", "")
        if bt == "narrative" or bt == "list":
            text = (blk.get("text") or "").strip()
            if text:
                parts.append(text[:max_chars] + ("…" if len(text) > max_chars else ""))
        elif bt == "table":
            summary = (blk.get("summary") or blk.get("text") or "").strip()
            if summary:
                parts.append(summary[:300] + ("…" if len(summary) > 300 else ""))
        elif bt == "image":
            cap = (blk.get("caption") or blk.get("description") or "").strip()
            if cap:
                parts.append(cap[:200] + ("…" if len(cap) > 200 else ""))
    return "\n".join(p for p in parts if p).strip()[:800]  # cap total snippet


def add_section_snippets(sections: list[dict]) -> None:
    for sec in sections:
        sec["section_snippet"] = build_section_snippet(sec)


def run(json_path: str) -> dict:
    """Load JSON, normalize, build outline and logical blocks; return result."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
    raw_elements = json.loads(data)
    if not isinstance(raw_elements, list):
        raw_elements = [raw_elements]

    print(f"No of raw elements: {len(raw_elements)}")
    raw_elements = [el for el in raw_elements if el.get("type") not in OMIT_TYPES]
    print(f"No of raw elements after removing headers and footers: {len(raw_elements)}")

    file_id, file_name = resolve_file_identifiers(raw_elements, json_path)
    normalized = normalize_elements(raw_elements, file_id, file_name)
    print(f"No of normalized elements: {len(normalized)}")
    type_counts = Counter(el["type"] for el in normalized)
    print("Elements per type:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    outline = build_outline(normalized)
    raw_by_id = {el.get("element_id"): el for el in raw_elements if el.get("element_id")}
    sections_with_blocks = build_logical_blocks(outline, raw_by_id)
    add_section_snippets(sections_with_blocks)
    print(f"No of sections: {len(sections_with_blocks)}")

    return {
        "file_id": file_id,
        "file_name": file_name,
        "normalized_elements": normalized,
        "outline": outline,
        "sections": sections_with_blocks,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract and normalize document elements from a JSON file."
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Path to the JSON file (array of unstructured elements)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Write result to this JSON file (default: print to stdout)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent for output (default: 2)",
    )
    args = parser.parse_args()

    try:
        result = run(args.json)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    out = json.dumps(result, indent=args.indent, ensure_ascii=False)
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("output/processed/")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.json).stem
        out_path = out_dir / f"{stem}_extraction.json"
    out_path.write_text(out, encoding="utf-8")
    print(f"Output saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
