## Roadmap using unstructured-io

- Normalise every element into canonical internal schema something like this (sequence is important)
```
{
  "element_id": "...",
  "type": "Title | NarrativeText | Image | Table | ...",
  "text": "...",
  "page_number": 3,
  "bbox": { "x1": 12, "y1": 40, "x2": 512, "y2": 88 },
  "file_id": "policy_v3.pdf",
  "file_name": "policy_v3.pdf",
  "sequence": 42
}
```
- Build a Document Outline: Reconstruct the document and convert elements into structure. Group by 'Title'
```
{
  "section_id": "sec_03",
  "title": "Refund Policy",
  "elements": [
    NarrativeText,
    ListItem,
    Table,
    Image
  ]
}
```
- Build logical blocks:
    - Merge adjacent NarrativeText elements
    - Stop merging if:
        - list/table/image appears
    - Merge adjacent ListItem elements
    - Keep Tables as an atomic block
        - Store raw HTML format
        - Summary of the table
    - Image block should have:
        - Element metadata
        - Base64 representation
        - Generate a description and caption for the image

- Chunking strategies for different blocks
    - Text:
        - 150-300 tokens 
        - Never cross section boundaries
        - Allow overlap within section only
    - Tables:
        - Embed table summaries
        - Link table summaries with table in raw HTML
    - Images:
        - Images are referenced entities


Root -> File
Nodes -> Sections
Leaves -> Blocks