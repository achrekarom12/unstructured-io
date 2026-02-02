#!/usr/bin/env python3
"""
Vectorless RAG retrieval (PageIndex-style).

Uses the extraction output (outline + section snippets) and an LLM to reason
which sections are relevant to a query—no vectors or chunk embeddings.
Returns selected sections' blocks and the reasoning path for explainability.
Can also generate an LLM-based answer from the retrieved context.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from ai.main import asyncGenerateText


OUTLINE_PROMPT = """You are a document retrieval assistant. You have a document that was split into sections. Your job is to decide which sections are relevant to the user's question.

Document outline (each section has an id, title, and a short snippet of its content):

{outline}

User question: {query}

Respond with a JSON object only, no other text. Use this exact format:
{{"section_ids": ["section_xxx", "section_yyy", ...], "reasoning": "Brief explanation of why you chose these sections."}}

Rules:
- Include only section_ids that are likely to contain information needed to answer the question.
- If the question is broad, you may include more sections; if specific, fewer.
- reasoning must be one short sentence explaining your choices.
"""


def load_extraction(path: str | Path) -> dict:
    """Load extraction JSON from file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Extraction file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_outline_text(sections: list[dict]) -> str:
    """Build a single text outline from sections for the LLM (section_id, title, snippet)."""
    lines = []
    for i, sec in enumerate(sections, start=1):
        sid = sec.get("section_id", "")
        title = sec.get("title", "(no title)")
        snippet = sec.get("section_snippet") or sec.get("title") or ""
        if not snippet and sec.get("blocks"):
            # Fallback: first block text truncated
            first = sec["blocks"][0]
            snippet = (first.get("text") or first.get("summary") or "")[:200]
        lines.append(f"[{i}] section_id: {sid}\ntitle: {title}\nsnippet: {snippet}\n")
    return "\n".join(lines)


def parse_llm_response(response: str) -> tuple[list[str], str]:
    """Parse LLM JSON response into section_ids and reasoning. Tolerates markdown code blocks."""
    reasoning = ""
    text = response.strip()
    # Strip optional markdown code block
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
        section_ids = list(data.get("section_ids") or [])
        reasoning = (data.get("reasoning") or "").strip()
        return section_ids, reasoning
    except json.JSONDecodeError:
        # Fallback: try to find section_... ids in the text
        section_ids = re.findall(r"section_[a-f0-9]+", text)
        return list(dict.fromkeys(section_ids)), reasoning


async def retrieve_async(
    extraction: dict,
    query: str,
    *,
    model: str = "gpt-5-nano",
    max_sections: int | None = None,
) -> dict:
    """
    Reason over the document outline and return relevant sections (vectorless retrieval).

    Returns:
        {
            "query": str,
            "section_ids": list[str],
            "reasoning": str,
            "sections": list[dict],  # full section objects with blocks
            "blocks": list[dict],    # flattened blocks from selected sections
        }
    """
    sections = extraction.get("sections") or []
    if not sections:
        return {
            "query": query,
            "section_ids": [],
            "reasoning": "No sections in extraction.",
            "sections": [],
            "blocks": [],
        }

    outline_text = build_outline_text(sections)
    prompt = OUTLINE_PROMPT.format(outline=outline_text, query=query)

    response = await asyncGenerateText(
        instructions="You must respond with valid JSON only. No markdown, no explanation outside the JSON.",
        input=prompt,
        model=model,
    )

    section_ids, reasoning = parse_llm_response(response)
    id_set = {s["section_id"] for s in sections}
    section_ids = [sid for sid in section_ids if sid in id_set]
    if max_sections and len(section_ids) > max_sections:
        section_ids = section_ids[:max_sections]

    selected_sections = [s for s in sections if s["section_id"] in section_ids]
    blocks = []
    for sec in selected_sections:
        for blk in sec.get("blocks", []):
            blocks.append(blk)

    return {
        "query": query,
        "section_ids": section_ids,
        "reasoning": reasoning,
        "sections": selected_sections,
        "blocks": blocks,
    }


def retrieve(
    extraction_path: str | Path | dict,
    query: str,
    *,
    model: str = "gpt-5-nano",
    max_sections: int | None = None,
) -> dict:
    """
    Synchronous wrapper for vectorless retrieval.

    extraction_path: path to *_extraction.json or an extraction dict.
    """
    if isinstance(extraction_path, dict):
        extraction = extraction_path
    else:
        extraction = load_extraction(extraction_path)

    return asyncio.run(
        retrieve_async(extraction, query, model=model, max_sections=max_sections)
    )


def blocks_to_context_text(blocks: list[dict], max_chars_per_block: int = 8000) -> str:
    """
    Format retrieved blocks into a single context string for the answer LLM.
    Uses section_title, then block text/summary/description per block.
    """
    parts = []
    current_section = None
    for blk in blocks:
        sec_title = blk.get("section_title") or ""
        if sec_title != current_section:
            current_section = sec_title
            parts.append(f"\n## {sec_title}\n")
        bt = blk.get("block_type", "")
        if bt in ("narrative", "list", "other") or bt == "Title":
            text = (blk.get("text") or "").strip()
            if text:
                parts.append(text[:max_chars_per_block] + ("…" if len(text) > max_chars_per_block else ""))
                parts.append("\n")
        elif bt == "table":
            summary = (blk.get("summary") or blk.get("text") or "").strip()
            if summary:
                parts.append(summary[:max_chars_per_block] + ("…" if len(summary) > max_chars_per_block else ""))
                parts.append("\n")
        elif bt == "image":
            cap = (blk.get("caption") or blk.get("description") or "").strip()
            if cap:
                parts.append(f"[Image: {cap}]\n")
    return "\n".join(parts).strip() if parts else "(No content in selected sections.)"


ANSWER_PROMPT = """You are a helpful assistant that answers questions based only on the provided document context.

Rules:
- Answer using only the information in the context. Do not use external knowledge.
- If the context does not contain enough information to answer, say so clearly.
- Be concise but complete. Quote or refer to specific parts of the context when relevant.
- Do not make up facts or numbers."""


async def answer_from_context_async(
    context: str,
    query: str,
    *,
    model: str = "gpt-5-nano",
) -> str:
    """Generate an LLM answer from context and user query."""
    if not context or context == "(No content in selected sections.)":
        return "No relevant context was retrieved for this question."
    full_input = f"{ANSWER_PROMPT}\n\n---\nContext:\n{context}\n\n---\nQuestion: {query}"
    return await asyncGenerateText(
        instructions="Answer the user's question based only on the document context above.",
        input=full_input,
        model=model,
    )


def answer_from_context(context: str, query: str, *, model: str = "gpt-5-nano") -> str:
    """Synchronous wrapper for answer_from_context_async."""
    return asyncio.run(answer_from_context_async(context, query, model=model))


def retrieve_and_answer(
    extraction_path: str | Path | dict,
    query: str,
    *,
    model: str = "gpt-5-nano",
    answer_model: str | None = None,
    max_sections: int | None = None,
) -> dict:
    """
    Run vectorless retrieval then generate an LLM answer from the retrieved context.

    Returns the retrieval result dict plus "context_preview", "answer".
    """
    if isinstance(extraction_path, dict):
        extraction = extraction_path
    else:
        extraction = load_extraction(extraction_path)

    async def _run():
        ret = await retrieve_async(
            extraction, query, model=model, max_sections=max_sections
        )
        context_text = blocks_to_context_text(ret["blocks"])
        ans_model = answer_model or model
        answer = await answer_from_context_async(
            context_text, query, model=ans_model
        )
        ret["context_preview"] = context_text[:2000] + ("…" if len(context_text) > 2000 else "")
        ret["answer"] = answer
        return ret

    return asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve sections by LLM reasoning, then generate an answer from context."
    )
    parser.add_argument(
        "extraction",
        type=str,
        help="Path to extraction JSON (e.g. output/processed/..._extraction.json)",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Question to answer using the document",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="Model for retrieval and answer (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default=None,
        help="Model for answer generation only (default: same as --model)",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=None,
        help="Cap number of sections to retrieve (default: no cap)",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only retrieve sections; do not generate an answer",
    )
    parser.add_argument(
        "--blocks-only",
        action="store_true",
        help="Print only retrieved blocks as JSON (implies --retrieve-only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON (answer, reasoning, section_ids, blocks count)",
    )
    args = parser.parse_args()

    if args.blocks_only:
        result = retrieve(
            args.extraction,
            args.query,
            model=args.model,
            max_sections=args.max_sections,
        )
        print(json.dumps(result["blocks"], indent=2, ensure_ascii=False))
        return

    if args.retrieve_only:
        result = retrieve(
            args.extraction,
            args.query,
            model=args.model,
            max_sections=args.max_sections,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Reasoning:", result["reasoning"])
            print("Section IDs:", result["section_ids"])
            print("Blocks count:", len(result["blocks"]))
        return

    result = retrieve_and_answer(
        args.extraction,
        args.query,
        model=args.model,
        answer_model=args.answer_model,
        max_sections=args.max_sections,
    )

    if args.json:
        # Exclude full blocks for compact output; keep answer, reasoning, section_ids
        out = {k: v for k, v in result.items() if k != "blocks" and k != "sections"}
        out["blocks_count"] = len(result.get("blocks", []))
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("Reasoning:", result["reasoning"])
        print("Sections used:", result["section_ids"])
        print()
        print("Answer:")
        print(result["answer"])


if __name__ == "__main__":
    main()
