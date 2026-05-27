"""Generate the seed evaluation corpus + datasets (SPEC §9.1, epic ka-5ps).

Writes a small fixture corpus (``evaluation/datasets/seed/corpus/*.md``) and the
``dev`` / ``rotating`` / ``frozen`` JSONL splits with *correct* gold ids: ids are
read back from the same deterministic ingestion (``parse_path`` + the default
recursive chunker) the eval runner uses, so ``relevant_doc_ids`` and
``relevant_chunk_ids`` always match what retrieval will surface.

Re-run after editing ``TOPICS`` to regenerate committed artifacts::

    uv run scripts/gen_seed_dataset.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from common.schemas import GoldQuery
from knowledge_index.chunking import RecursiveChunker
from knowledge_index.ingestion.parsers import parse_path

SEED_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "seed"
CORPUS_DIR = SEED_DIR / "corpus"

# Chunker config must match configs/default.yaml index.chunker so chunk ids align.
CHUNKER = RecursiveChunker(chunk_size=500, chunk_overlap=75)


# Each topic: a distinctive fixture doc plus dev/frozen query phrasings and the
# expected answer. Vocabulary overlap with the queries lets BM25 recall the doc.
TOPICS: dict[str, dict] = {
    "mitochondria": {
        "title": "Mitochondria",
        "text": "The mitochondrion is the powerhouse of the cell. It produces ATP through "
        "cellular respiration, converting glucose and oxygen into energy.",
        "answer": "Mitochondria produce ATP via cellular respiration.",
        "dev": [
            "What does the mitochondrion do in a cell?",
            "How do mitochondria produce energy?",
            "What is the powerhouse of the cell?",
        ],
        "frozen": [
            "Which organelle generates ATP through respiration?",
            "Explain the role of mitochondria in energy production.",
            "Where is ATP produced in the cell?",
        ],
    },
    "photosynthesis": {
        "title": "Photosynthesis",
        "text": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and "
        "oxygen. It occurs in the chloroplasts of plant cells using chlorophyll.",
        "answer": "Photosynthesis turns sunlight, water, and CO2 into glucose and oxygen.",
        "dev": [
            "How does photosynthesis work?",
            "What does photosynthesis produce?",
            "Where does photosynthesis happen in plant cells?",
        ],
        "frozen": [
            "What inputs does photosynthesis convert into glucose?",
            "Which pigment drives photosynthesis in chloroplasts?",
            "Explain how plants make oxygen from sunlight.",
        ],
    },
    "everest": {
        "title": "Mount Everest",
        "text": "Mount Everest is the highest mountain above sea level, with a summit at 8849 "
        "metres. It sits in the Himalayas on the border of Nepal and Tibet.",
        "answer": "Mount Everest is the highest mountain at 8849 metres.",
        "dev": [
            "How tall is Mount Everest?",
            "What is the highest mountain on Earth?",
            "Where is Mount Everest located?",
        ],
        "frozen": [
            "What is the summit elevation of Everest in metres?",
            "Which mountain is the tallest above sea level?",
            "On which countries' border does Everest sit?",
        ],
    },
    "water": {
        "title": "Water",
        "text": "Water has the chemical formula H2O, meaning each molecule has two hydrogen "
        "atoms bonded to one oxygen atom. It is essential for all known life.",
        "answer": "Water's chemical formula is H2O.",
        "dev": [
            "What is the chemical formula of water?",
            "How many hydrogen atoms are in a water molecule?",
            "What atoms make up water?",
        ],
        "frozen": [
            "Write the molecular formula for water.",
            "What does H2O stand for?",
            "How is a water molecule structured?",
        ],
    },
    "insulin": {
        "title": "Insulin",
        "text": "Insulin is a hormone produced by the pancreas that regulates blood glucose "
        "levels. It allows cells to absorb glucose from the bloodstream for energy.",
        "answer": "Insulin is a pancreatic hormone that regulates blood glucose.",
        "dev": [
            "What does insulin regulate?",
            "Which organ produces insulin?",
            "What is the function of insulin in the body?",
        ],
        "frozen": [
            "How does insulin control blood sugar?",
            "What hormone lets cells absorb glucose?",
            "Where is insulin made?",
        ],
    },
    "tcp": {
        "title": "TCP",
        "text": "TCP, the Transmission Control Protocol, is a reliable, connection-oriented "
        "transport protocol. It guarantees ordered, error-checked delivery of bytes.",
        "answer": "TCP is a reliable, connection-oriented transport protocol.",
        "dev": [
            "What is TCP?",
            "Is TCP connection-oriented or connectionless?",
            "What does the Transmission Control Protocol guarantee?",
        ],
        "frozen": [
            "Describe the reliability guarantees of TCP.",
            "What kind of transport protocol is TCP?",
            "What does TCP stand for?",
        ],
    },
    "shakespeare": {
        "title": "William Shakespeare",
        "text": "William Shakespeare was an English playwright who wrote Hamlet, Macbeth, and "
        "Romeo and Juliet. He is widely regarded as the greatest writer in English.",
        "answer": "William Shakespeare wrote Hamlet and other plays.",
        "dev": [
            "Who wrote Hamlet?",
            "What plays did Shakespeare write?",
            "Who is William Shakespeare?",
        ],
        "frozen": [
            "Name the author of Macbeth and Romeo and Juliet.",
            "Which English playwright wrote Hamlet?",
            "What is Shakespeare best known for?",
        ],
    },
    "python": {
        "title": "Python",
        "text": "Python is a high-level, interpreted programming language known for readable "
        "syntax and dynamic typing. It is widely used for data science and web backends.",
        "answer": "Python is a high-level interpreted programming language.",
        "dev": [
            "What is Python?",
            "What is Python used for?",
            "Is Python interpreted or compiled?",
        ],
        "frozen": [
            "Describe the Python programming language.",
            "What kind of typing does Python use?",
            "Which language is popular for data science and readable syntax?",
        ],
    },
    "rome": {
        "title": "Founding of Rome",
        "text": "According to legend, Rome was founded in 753 BC by Romulus. The city grew into "
        "the capital of the Roman Empire and a centre of ancient civilisation.",
        "answer": "Rome was founded in 753 BC.",
        "dev": [
            "When was Rome founded?",
            "Who founded Rome according to legend?",
            "What year was Rome established?",
        ],
        "frozen": [
            "In which year was the city of Rome founded?",
            "Which legendary figure founded Rome?",
            "What became the capital of the Roman Empire?",
        ],
    },
    "gravity": {
        "title": "Gravity",
        "text": "Gravity is the force that attracts objects with mass toward one another. On "
        "Earth, gravity accelerates falling objects at about 9.8 metres per second squared.",
        "answer": "Gravity accelerates objects on Earth at about 9.8 m/s^2.",
        "dev": [
            "What is gravity?",
            "What is the acceleration due to gravity on Earth?",
            "What force pulls objects toward Earth?",
        ],
        "frozen": [
            "How fast do objects accelerate when falling on Earth?",
            "Define the force of gravity.",
            "What attracts masses toward one another?",
        ],
    },
    "dna": {
        "title": "DNA",
        "text": "DNA, deoxyribonucleic acid, carries genetic instructions in a double helix of "
        "nucleotide base pairs: adenine with thymine and guanine with cytosine.",
        "answer": "DNA stores genetic information as a double helix of base pairs.",
        "dev": [
            "What does DNA store?",
            "What is the structure of DNA?",
            "Which base pairs make up DNA?",
        ],
        "frozen": [
            "What molecule carries genetic instructions?",
            "Describe the double helix of DNA.",
            "Which bases pair together in DNA?",
        ],
    },
    "http": {
        "title": "HTTP",
        "text": "HTTP, the Hypertext Transfer Protocol, is the application-layer protocol used "
        "by the web. Clients send requests with methods like GET and POST to servers.",
        "answer": "HTTP is the application-layer protocol of the web.",
        "dev": [
            "What is HTTP used for?",
            "What methods does HTTP use?",
            "What does HTTP stand for?",
        ],
        "frozen": [
            "Which protocol powers requests on the web?",
            "Name two common HTTP request methods.",
            "What layer does HTTP operate at?",
        ],
    },
    "photosynthesis_light": {
        "title": "Chlorophyll",
        "text": "Chlorophyll is the green pigment in plants that absorbs light, mainly in the "
        "blue and red wavelengths, to power the light reactions of photosynthesis.",
        "answer": "Chlorophyll is the green pigment that absorbs light for photosynthesis.",
        "dev": [
            "What is chlorophyll?",
            "What colour light does chlorophyll absorb?",
            "Why are plants green?",
        ],
        "frozen": [
            "Which pigment absorbs blue and red light in plants?",
            "What gives leaves their green colour?",
            "What powers the light reactions in plants?",
        ],
    },
    "newton": {
        "title": "Newton's Laws",
        "text": "Newton's three laws of motion describe inertia, the relationship F = m a "
        "between force and acceleration, and equal and opposite reaction forces.",
        "answer": "Newton's laws describe inertia, F=ma, and action-reaction.",
        "dev": [
            "What do Newton's laws of motion describe?",
            "What is Newton's second law?",
            "How many laws of motion did Newton state?",
        ],
        "frozen": [
            "State the relationship between force, mass, and acceleration.",
            "What does the third law of motion say about forces?",
            "Whose laws explain inertia and acceleration?",
        ],
    },
    "ocean": {
        "title": "Pacific Ocean",
        "text": "The Pacific Ocean is the largest and deepest ocean on Earth, covering about a "
        "third of the surface. Its deepest point is the Mariana Trench.",
        "answer": "The Pacific is the largest ocean; its deepest point is the Mariana Trench.",
        "dev": [
            "What is the largest ocean on Earth?",
            "What is the deepest point in the Pacific Ocean?",
            "How much of Earth's surface does the Pacific cover?",
        ],
        "frozen": [
            "Which ocean is the biggest and deepest?",
            "Where is the Mariana Trench located?",
            "What fraction of the surface does the Pacific Ocean cover?",
        ],
    },
    "electricity": {
        "title": "Electric Current",
        "text": "Electric current is the flow of electric charge, measured in amperes. Ohm's law "
        "states that current equals voltage divided by resistance: I = V / R.",
        "answer": "Current is charge flow; Ohm's law is I = V / R.",
        "dev": [
            "What is electric current?",
            "What is Ohm's law?",
            "In what unit is electric current measured?",
        ],
        "frozen": [
            "State the formula relating current, voltage, and resistance.",
            "What does the ampere measure?",
            "How is the flow of electric charge described?",
        ],
    },
    "evolution": {
        "title": "Natural Selection",
        "text": "Natural selection, proposed by Charles Darwin, is the process by which "
        "organisms with advantageous heritable traits tend to survive and reproduce more.",
        "answer": "Natural selection is Darwin's mechanism of evolution by differential survival.",
        "dev": [
            "What is natural selection?",
            "Who proposed natural selection?",
            "How does natural selection drive evolution?",
        ],
        "frozen": [
            "Which scientist described evolution by natural selection?",
            "What traits tend to spread under natural selection?",
            "Explain Darwin's mechanism of evolution.",
        ],
    },
}


async def _ids_for(slug: str, text_md: str) -> tuple[str, list[str]]:
    """Parse + chunk a corpus file, returning its doc id and chunk ids."""
    path = CORPUS_DIR / f"{slug}.md"
    doc = await parse_path(str(path))
    chunks = CHUNKER.chunk(doc)
    return doc.doc_id, [c.chunk_id for c in chunks]


def _difficulty(i: int) -> str:
    return ["easy", "medium", "hard"][i % 3]


async def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Write the fixture corpus.
    for slug, t in TOPICS.items():
        (CORPUS_DIR / f"{slug}.md").write_text(f"# {t['title']}\n\n{t['text']}\n")

    # 2. Resolve deterministic gold ids per topic from the written files.
    ids: dict[str, tuple[str, list[str]]] = {}
    for slug, t in TOPICS.items():
        ids[slug] = await _ids_for(slug, t["text"])

    # 3. Build each split from its query phrasings.
    def build_split(key: str) -> list[GoldQuery]:
        out: list[GoldQuery] = []
        for slug, t in TOPICS.items():
            doc_id, chunk_ids = ids[slug]
            for i, q in enumerate(t[key]):
                out.append(
                    GoldQuery(
                        query_id=f"{key}-{slug}-{i}",
                        query=q,
                        relevant_chunk_ids=chunk_ids,
                        relevant_doc_ids=[doc_id],
                        expected_answer=t["answer"],
                        intent="lookup",
                        difficulty=_difficulty(i),
                        notes=f"seed topic: {slug}",
                    )
                )
        return out

    splits = {
        "dev": build_split("dev"),
        "frozen": build_split("frozen"),
        # rotating reuses dev phrasings (placeholder until a real rotating set exists)
        "rotating": build_split("dev"),
    }

    for name, queries in splits.items():
        path = SEED_DIR / f"{name}.jsonl"
        with path.open("w") as fh:
            for gq in queries:
                fh.write(json.dumps(gq.model_dump(), ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(SEED_DIR.parents[2])}: {len(queries)} queries")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
