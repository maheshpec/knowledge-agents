"""Generate the seed evaluation corpus + datasets at spec scale (SPEC §9.1/§11).

Writes a synthetic fixture corpus (``evaluation/datasets/seed/corpus/*.md``) and
the ``dev`` / ``rotating`` / ``frozen`` JSONL splits with *correct* gold ids: ids
are read back from the same deterministic ingestion (``parse_path`` + the default
recursive chunker) the eval runner uses, so ``relevant_doc_ids`` and
``relevant_chunk_ids`` always match what retrieval will surface.

Scale + discipline (SPEC §9/§11, Gap G1 / ka-94g):

* **Targets** — ~500 dev, ~500 rotating, ~1000 frozen. Reached by 50 synthetic
  topics × 5 probes × a paraphrase-template bank partitioned across splits.
* **Frozen hold-out (§13)** — each paraphrase template belongs to exactly one
  split (``_SPLIT_TEMPLATES``), so the frozen query *strings* are disjoint from
  dev ∪ rotating. The frozen set is never reused in evolutionary search.
* **Difficulty stratification** — every split carries easy/medium/hard queries,
  cycled deterministically within the split.
* **Synthetic / public only** — all topics are general-knowledge facts authored
  here; real user content lives gitignored under ``datasets/private/`` (§10.5).

Re-run after editing ``TOPICS`` to regenerate the committed artifacts::

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

# Paraphrase templates applied to a probe's noun-phrase subject. Each reads
# naturally with a subject like "the chemical formula of water". Templates are
# partitioned across splits below so a (subject, template) pair lands in exactly
# one split — that is what keeps the frozen query strings disjoint from the rest.
_TEMPLATES: list[str] = [
    "What is {s}?",  # 0
    "Can you explain {s}?",  # 1
    "Tell me about {s}.",  # 2
    "How would you describe {s}?",  # 3
    "What does {s} refer to?",  # 4
    "Give a brief overview of {s}.",  # 5
    "Help me understand {s}.",  # 6
    "What should I know about {s}?",  # 7
]

# Split -> the template indices it owns. Disjoint by construction; frozen gets the
# larger share to hit the ~1000 target (2× dev/rotating).
_SPLIT_TEMPLATES: dict[str, list[int]] = {
    "dev": [0, 1],
    "rotating": [2, 3],
    "frozen": [4, 5, 6, 7],
}

_DIFFICULTIES = ["easy", "medium", "hard"]


# Each topic: a distinctive fixture doc plus a set of probes. A probe is a
# (subject, answer) pair where ``subject`` is a noun phrase that reads naturally
# inside the templates and shares vocabulary with the doc (so BM25/dense recall
# can surface it). 50 topics × 5 probes drives the split sizes.
TOPICS: dict[str, dict] = {
    "mitochondria": {
        "title": "Mitochondria",
        "text": "The mitochondrion is the powerhouse of the cell. It produces ATP through "
        "cellular respiration, converting glucose and oxygen into energy. Mitochondria have "
        "their own DNA and a double membrane, and they are most abundant in muscle cells.",
        "probes": [
            ("the role of the mitochondrion in a cell", "It produces ATP, the cell's energy."),
            ("how mitochondria produce energy", "By cellular respiration of glucose and oxygen."),
            ("the molecule mitochondria produce", "ATP, adenosine triphosphate."),
            ("the membrane structure of mitochondria", "They have a double membrane."),
            (
                "why mitochondria are called the powerhouse of the cell",
                "They generate the cell's ATP energy.",
            ),
        ],
    },
    "photosynthesis": {
        "title": "Photosynthesis",
        "text": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and "
        "oxygen. It occurs in the chloroplasts of plant cells using the pigment chlorophyll. "
        "The process has light-dependent reactions and the light-independent Calvin cycle.",
        "probes": [
            (
                "the process of photosynthesis",
                "Plants turn sunlight, water, and CO2 into glucose and oxygen.",
            ),
            ("the products of photosynthesis", "Glucose and oxygen."),
            ("where photosynthesis occurs in plant cells", "In the chloroplasts."),
            (
                "the inputs photosynthesis converts into glucose",
                "Sunlight, water, and carbon dioxide.",
            ),
            (
                "the Calvin cycle in photosynthesis",
                "The light-independent reactions that fix carbon.",
            ),
        ],
    },
    "everest": {
        "title": "Mount Everest",
        "text": "Mount Everest is the highest mountain above sea level, with a summit at 8849 "
        "metres. It sits in the Himalayas on the border of Nepal and Tibet. It was first "
        "summited in 1953 by Edmund Hillary and Tenzing Norgay.",
        "probes": [
            ("the height of Mount Everest", "About 8849 metres above sea level."),
            ("the highest mountain on Earth", "Mount Everest."),
            ("the location of Mount Everest", "The Himalayas, on the Nepal–Tibet border."),
            ("the first ascent of Everest", "By Hillary and Norgay in 1953."),
            ("the mountain range containing Everest", "The Himalayas."),
        ],
    },
    "water": {
        "title": "Water",
        "text": "Water has the chemical formula H2O, meaning each molecule has two hydrogen "
        "atoms bonded to one oxygen atom. It is essential for all known life. Water boils at "
        "100 degrees Celsius and freezes at 0 degrees Celsius at sea level.",
        "probes": [
            ("the chemical formula of water", "H2O."),
            ("the atoms that make up a water molecule", "Two hydrogen atoms and one oxygen atom."),
            ("the boiling point of water at sea level", "100 degrees Celsius."),
            ("the freezing point of water", "0 degrees Celsius."),
            ("why water is essential to life", "All known life depends on it."),
        ],
    },
    "insulin": {
        "title": "Insulin",
        "text": "Insulin is a hormone produced by the pancreas that regulates blood glucose "
        "levels. It allows cells to absorb glucose from the bloodstream for energy. A lack of "
        "insulin or insulin resistance causes diabetes.",
        "probes": [
            ("the function of insulin in the body", "It regulates blood glucose."),
            ("the organ that produces insulin", "The pancreas."),
            ("how insulin lowers blood sugar", "It lets cells absorb glucose from the blood."),
            ("the disease caused by lack of insulin", "Diabetes."),
            ("the type of molecule insulin is", "A hormone."),
        ],
    },
    "tcp": {
        "title": "TCP",
        "text": "TCP, the Transmission Control Protocol, is a reliable, connection-oriented "
        "transport protocol. It guarantees ordered, error-checked delivery of bytes. TCP "
        "establishes a connection with a three-way handshake before sending data.",
        "probes": [
            (
                "the Transmission Control Protocol",
                "A reliable, connection-oriented transport protocol.",
            ),
            ("the reliability guarantees of TCP", "Ordered, error-checked byte delivery."),
            ("how TCP starts a connection", "With a three-way handshake."),
            ("whether TCP is connection-oriented", "Yes, it is connection-oriented."),
            ("what the acronym TCP stands for", "Transmission Control Protocol."),
        ],
    },
    "shakespeare": {
        "title": "William Shakespeare",
        "text": "William Shakespeare was an English playwright who wrote Hamlet, Macbeth, and "
        "Romeo and Juliet. He is widely regarded as the greatest writer in the English "
        "language. He was born in Stratford-upon-Avon in 1564.",
        "probes": [
            ("the author of Hamlet", "William Shakespeare."),
            ("the plays William Shakespeare wrote", "Hamlet, Macbeth, and Romeo and Juliet."),
            ("the birthplace of Shakespeare", "Stratford-upon-Avon."),
            ("what Shakespeare is best known for", "Being the greatest English playwright."),
            ("the nationality of Shakespeare", "English."),
        ],
    },
    "python": {
        "title": "Python",
        "text": "Python is a high-level, interpreted programming language known for readable "
        "syntax and dynamic typing. It is widely used for data science and web backends. "
        "Python was created by Guido van Rossum and released in 1991.",
        "probes": [
            (
                "the Python programming language",
                "A high-level, interpreted language with readable syntax.",
            ),
            ("what Python is used for", "Data science and web backends."),
            ("the type system of Python", "Dynamic typing."),
            ("the creator of Python", "Guido van Rossum."),
            ("whether Python is interpreted or compiled", "Interpreted."),
        ],
    },
    "rome": {
        "title": "Founding of Rome",
        "text": "According to legend, Rome was founded in 753 BC by Romulus. The city grew into "
        "the capital of the Roman Empire and a centre of ancient civilisation. Romulus and "
        "his brother Remus were said to be raised by a she-wolf.",
        "probes": [
            ("the founding year of Rome", "753 BC, by legend."),
            ("the legendary founder of Rome", "Romulus."),
            ("what Rome became the capital of", "The Roman Empire."),
            ("the legend of Romulus and Remus", "They were raised by a she-wolf."),
            ("the significance of ancient Rome", "It was a centre of ancient civilisation."),
        ],
    },
    "gravity": {
        "title": "Gravity",
        "text": "Gravity is the force that attracts objects with mass toward one another. On "
        "Earth, gravity accelerates falling objects at about 9.8 metres per second squared. "
        "Isaac Newton described gravity with his law of universal gravitation.",
        "probes": [
            ("the force of gravity", "The attraction between objects with mass."),
            ("the acceleration due to gravity on Earth", "About 9.8 metres per second squared."),
            ("who described universal gravitation", "Isaac Newton."),
            ("what gravity acts on", "Objects with mass."),
            ("the effect of gravity on falling objects", "It accelerates them downward."),
        ],
    },
    "dna": {
        "title": "DNA",
        "text": "DNA, deoxyribonucleic acid, carries genetic instructions in a double helix of "
        "nucleotide base pairs: adenine with thymine and guanine with cytosine. It was shown "
        "to be a double helix by Watson and Crick in 1953.",
        "probes": [
            ("what DNA stores", "Genetic instructions."),
            ("the structure of DNA", "A double helix of base pairs."),
            ("the base pairs in DNA", "Adenine–thymine and guanine–cytosine."),
            ("who described the DNA double helix", "Watson and Crick."),
            ("what DNA stands for", "Deoxyribonucleic acid."),
        ],
    },
    "http": {
        "title": "HTTP",
        "text": "HTTP, the Hypertext Transfer Protocol, is the application-layer protocol used "
        "by the web. Clients send requests with methods like GET and POST to servers. HTTP is "
        "stateless, so each request is independent.",
        "probes": [
            ("the Hypertext Transfer Protocol", "The application-layer protocol of the web."),
            ("common HTTP request methods", "GET and POST."),
            ("the network layer HTTP operates at", "The application layer."),
            ("why HTTP is called stateless", "Each request is independent of the others."),
            ("what HTTP is used for", "Requests and responses on the web."),
        ],
    },
    "chlorophyll": {
        "title": "Chlorophyll",
        "text": "Chlorophyll is the green pigment in plants that absorbs light, mainly in the "
        "blue and red wavelengths, to power the light reactions of photosynthesis. It reflects "
        "green light, which is why leaves look green.",
        "probes": [
            ("the pigment chlorophyll", "The green pigment that absorbs light for photosynthesis."),
            ("the light wavelengths chlorophyll absorbs", "Mainly blue and red."),
            ("why leaves look green", "Chlorophyll reflects green light."),
            ("the role of chlorophyll in photosynthesis", "It powers the light reactions."),
            ("where chlorophyll is found", "In plants."),
        ],
    },
    "newton": {
        "title": "Newton's Laws",
        "text": "Newton's three laws of motion describe inertia, the relationship F = m a "
        "between force and acceleration, and equal and opposite reaction forces. They form the "
        "foundation of classical mechanics.",
        "probes": [
            (
                "Newton's laws of motion",
                "Three laws describing inertia, F=ma, and reaction forces.",
            ),
            ("Newton's second law", "Force equals mass times acceleration."),
            ("the number of Newton's laws of motion", "Three."),
            ("Newton's third law", "Every action has an equal and opposite reaction."),
            ("what Newton's first law describes", "Inertia."),
        ],
    },
    "ocean": {
        "title": "Pacific Ocean",
        "text": "The Pacific Ocean is the largest and deepest ocean on Earth, covering about a "
        "third of the surface. Its deepest point is the Mariana Trench, nearly 11 kilometres "
        "deep. It lies between Asia, Australia, and the Americas.",
        "probes": [
            ("the largest ocean on Earth", "The Pacific Ocean."),
            ("the deepest point of the Pacific Ocean", "The Mariana Trench."),
            ("the fraction of Earth's surface the Pacific covers", "About one third."),
            ("the depth of the Mariana Trench", "Nearly 11 kilometres."),
            ("the location of the Pacific Ocean", "Between Asia, Australia, and the Americas."),
        ],
    },
    "electricity": {
        "title": "Electric Current",
        "text": "Electric current is the flow of electric charge, measured in amperes. Ohm's law "
        "states that current equals voltage divided by resistance: I = V / R. Current can be "
        "direct (DC) or alternating (AC).",
        "probes": [
            ("electric current", "The flow of electric charge."),
            ("Ohm's law", "Current equals voltage divided by resistance, I = V / R."),
            ("the unit of electric current", "The ampere."),
            ("the difference between AC and DC current", "Alternating versus direct current."),
            ("what the ampere measures", "Electric current."),
        ],
    },
    "evolution": {
        "title": "Natural Selection",
        "text": "Natural selection, proposed by Charles Darwin, is the process by which "
        "organisms with advantageous heritable traits tend to survive and reproduce more. Over "
        "generations this drives evolution and adaptation to the environment.",
        "probes": [
            ("natural selection", "Darwin's mechanism of evolution by differential survival."),
            ("who proposed natural selection", "Charles Darwin."),
            (
                "how natural selection drives evolution",
                "Advantageous traits spread over generations.",
            ),
            ("the traits favoured by natural selection", "Advantageous heritable traits."),
            ("the outcome of natural selection over time", "Adaptation to the environment."),
        ],
    },
    "moon": {
        "title": "The Moon",
        "text": "The Moon is Earth's only natural satellite, about 384000 kilometres away. Its "
        "gravity drives the ocean tides. The same side always faces Earth because the Moon is "
        "tidally locked. Humans first landed on it in 1969.",
        "probes": [
            ("the Moon's relationship to Earth", "It is Earth's only natural satellite."),
            ("the distance to the Moon", "About 384000 kilometres."),
            ("what causes ocean tides", "The Moon's gravity."),
            ("why the same side of the Moon faces Earth", "It is tidally locked."),
            ("the first human Moon landing", "It happened in 1969."),
        ],
    },
    "sun": {
        "title": "The Sun",
        "text": "The Sun is the star at the centre of the solar system, a ball of hot plasma. It "
        "produces energy by nuclear fusion of hydrogen into helium. It is about 150 million "
        "kilometres from Earth, a distance called one astronomical unit.",
        "probes": [
            ("the Sun", "The star at the centre of the solar system."),
            ("how the Sun produces energy", "By nuclear fusion of hydrogen into helium."),
            ("the distance from Earth to the Sun", "About 150 million kilometres."),
            ("what an astronomical unit measures", "The Earth–Sun distance."),
            ("what the Sun is made of", "Hot plasma, mostly hydrogen and helium."),
        ],
    },
    "heart": {
        "title": "The Human Heart",
        "text": "The human heart is a muscular organ that pumps blood through the circulatory "
        "system. It has four chambers: two atria and two ventricles. It beats about 100000 "
        "times a day, driven by an electrical pacemaker called the sinoatrial node.",
        "probes": [
            ("the function of the human heart", "It pumps blood through the body."),
            ("the number of chambers in the heart", "Four."),
            ("the chambers of the heart", "Two atria and two ventricles."),
            ("the heart's natural pacemaker", "The sinoatrial node."),
            ("how often the heart beats per day", "About 100000 times."),
        ],
    },
    "brain": {
        "title": "The Human Brain",
        "text": "The human brain is the control centre of the nervous system, containing about "
        "86 billion neurons. The cerebrum handles thought and movement, the cerebellum handles "
        "coordination, and the brainstem controls vital functions like breathing.",
        "probes": [
            ("the role of the human brain", "It is the control centre of the nervous system."),
            ("the number of neurons in the brain", "About 86 billion."),
            ("the function of the cerebrum", "Thought and voluntary movement."),
            ("the function of the cerebellum", "Coordination and balance."),
            ("what the brainstem controls", "Vital functions like breathing."),
        ],
    },
    "volcano": {
        "title": "Volcanoes",
        "text": "A volcano is a rupture in the Earth's crust where molten rock, ash, and gases "
        "escape. Magma below the surface is called lava once it erupts. Most volcanoes form at "
        "tectonic plate boundaries, such as the Pacific Ring of Fire.",
        "probes": [
            ("what a volcano is", "A crustal rupture where molten rock and gases escape."),
            ("the difference between magma and lava", "Lava is magma that has erupted."),
            ("where most volcanoes form", "At tectonic plate boundaries."),
            ("the Pacific Ring of Fire", "A zone of frequent volcanoes around the Pacific."),
            ("what erupts from a volcano", "Lava, ash, and gases."),
        ],
    },
    "earthquake": {
        "title": "Earthquakes",
        "text": "An earthquake is a sudden shaking of the ground caused by movement along faults "
        "in the Earth's crust. Its strength is measured on the moment magnitude scale. The "
        "point of origin underground is the focus, and the point above it is the epicentre.",
        "probes": [
            ("what causes an earthquake", "Sudden movement along faults in the crust."),
            ("how earthquake strength is measured", "On the moment magnitude scale."),
            ("the epicentre of an earthquake", "The surface point above the focus."),
            ("the focus of an earthquake", "The underground point of origin."),
            ("where earthquakes occur", "Along faults in the Earth's crust."),
        ],
    },
    "amazon_river": {
        "title": "The Amazon River",
        "text": "The Amazon is the largest river in the world by discharge, carrying more water "
        "than the next several rivers combined. It flows across South America, mostly through "
        "Brazil, and empties into the Atlantic Ocean.",
        "probes": [
            ("the largest river by water discharge", "The Amazon River."),
            ("the continent the Amazon flows through", "South America."),
            ("where the Amazon River empties", "Into the Atlantic Ocean."),
            ("the country containing most of the Amazon", "Brazil."),
            ("what makes the Amazon the largest river", "Its enormous water discharge."),
        ],
    },
    "great_wall": {
        "title": "The Great Wall of China",
        "text": "The Great Wall of China is a series of fortifications built over centuries to "
        "protect Chinese states from northern invasions. It stretches thousands of kilometres "
        "and was built mainly during the Ming dynasty.",
        "probes": [
            ("the purpose of the Great Wall of China", "To defend against northern invasions."),
            ("what the Great Wall is", "A long series of fortifications."),
            ("the dynasty that built most of the Great Wall", "The Ming dynasty."),
            ("the length of the Great Wall", "Thousands of kilometres."),
            ("the country where the Great Wall stands", "China."),
        ],
    },
    "periodic_table": {
        "title": "The Periodic Table",
        "text": "The periodic table arranges chemical elements by increasing atomic number into "
        "rows called periods and columns called groups. Elements in the same group share "
        "similar properties. It was devised by Dmitri Mendeleev.",
        "probes": [
            ("the periodic table", "An arrangement of elements by atomic number."),
            ("how elements are ordered in the periodic table", "By increasing atomic number."),
            ("the columns of the periodic table", "Groups of elements with similar properties."),
            ("the rows of the periodic table", "Periods."),
            ("who devised the periodic table", "Dmitri Mendeleev."),
        ],
    },
    "light_speed": {
        "title": "The Speed of Light",
        "text": "The speed of light in a vacuum is about 299792 kilometres per second, denoted "
        "c. It is the universe's ultimate speed limit. Light from the Sun takes about eight "
        "minutes to reach Earth.",
        "probes": [
            ("the speed of light in a vacuum", "About 299792 kilometres per second."),
            ("the symbol for the speed of light", "The letter c."),
            ("the cosmic speed limit", "The speed of light."),
            ("how long sunlight takes to reach Earth", "About eight minutes."),
            (
                "why the speed of light matters in physics",
                "It is the universe's ultimate speed limit.",
            ),
        ],
    },
    "atom": {
        "title": "The Atom",
        "text": "An atom is the basic unit of matter, made of a nucleus of protons and neutrons "
        "orbited by electrons. The number of protons determines the element. Atoms are "
        "incredibly small, with most of their mass in the nucleus.",
        "probes": [
            ("what an atom is", "The basic unit of matter."),
            ("the particles in an atomic nucleus", "Protons and neutrons."),
            ("what determines which element an atom is", "Its number of protons."),
            ("the particles that orbit the nucleus", "Electrons."),
            ("where most of an atom's mass is", "In the nucleus."),
        ],
    },
    "vaccine": {
        "title": "Vaccines",
        "text": "A vaccine trains the immune system to recognise a pathogen by exposing it to a "
        "harmless piece or weakened form of the microbe. This builds immunity without causing "
        "disease. Widespread vaccination can produce herd immunity.",
        "probes": [
            ("how a vaccine works", "It trains the immune system to recognise a pathogen."),
            ("what a vaccine contains", "A harmless or weakened piece of a microbe."),
            ("the benefit of a vaccine", "Immunity without getting the disease."),
            ("what herd immunity is", "Protection from widespread vaccination."),
            ("the system a vaccine trains", "The immune system."),
        ],
    },
    "antibiotics": {
        "title": "Antibiotics",
        "text": "Antibiotics are medicines that kill or stop the growth of bacteria. They do not "
        "work against viruses. Overuse of antibiotics can lead to antibiotic resistance, where "
        "bacteria evolve to survive the drugs. Penicillin was the first antibiotic.",
        "probes": [
            ("what antibiotics do", "Kill or stop the growth of bacteria."),
            ("why antibiotics do not treat viral infections", "They only act on bacteria."),
            ("the cause of antibiotic resistance", "Overuse letting bacteria evolve."),
            ("the first antibiotic discovered", "Penicillin."),
            ("the type of microbe antibiotics target", "Bacteria."),
        ],
    },
    "internet": {
        "title": "The Internet",
        "text": "The Internet is a global network of interconnected computers that communicate "
        "using the Internet Protocol. Data is broken into packets and routed between networks. "
        "The World Wide Web is a service that runs on top of the Internet.",
        "probes": [
            ("what the Internet is", "A global network of interconnected computers."),
            (
                "how data travels across the Internet",
                "Broken into packets and routed between networks.",
            ),
            ("the protocol the Internet uses", "The Internet Protocol."),
            (
                "the difference between the Internet and the Web",
                "The Web is a service on the Internet.",
            ),
            ("the units data is split into on the Internet", "Packets."),
        ],
    },
    "solar_system": {
        "title": "The Solar System",
        "text": "The solar system consists of the Sun and the objects bound to it by gravity, "
        "including eight planets. The four inner planets are rocky and the four outer planets "
        "are gas or ice giants. Jupiter is the largest planet.",
        "probes": [
            ("what the solar system contains", "The Sun and objects bound to it by gravity."),
            ("the number of planets in the solar system", "Eight."),
            ("the nature of the inner planets", "They are rocky."),
            ("the nature of the outer planets", "They are gas or ice giants."),
            ("the largest planet in the solar system", "Jupiter."),
        ],
    },
    "mars": {
        "title": "Mars",
        "text": "Mars is the fourth planet from the Sun, known as the Red Planet for its iron "
        "oxide surface. It has two small moons, Phobos and Deimos, and the tallest volcano in "
        "the solar system, Olympus Mons.",
        "probes": [
            ("the planet known as the Red Planet", "Mars."),
            ("why Mars looks red", "Its surface is rich in iron oxide."),
            ("the moons of Mars", "Phobos and Deimos."),
            ("the tallest volcano in the solar system", "Olympus Mons, on Mars."),
            ("the position of Mars from the Sun", "The fourth planet."),
        ],
    },
    "blood": {
        "title": "Blood",
        "text": "Blood is the fluid that circulates through the body carrying oxygen, nutrients, "
        "and waste. Red blood cells carry oxygen using haemoglobin, white blood cells fight "
        "infection, and platelets help clotting.",
        "probes": [
            ("the function of blood", "It carries oxygen, nutrients, and waste."),
            ("what red blood cells carry", "Oxygen, using haemoglobin."),
            ("the role of white blood cells", "They fight infection."),
            ("the role of platelets", "They help blood clot."),
            ("the protein that carries oxygen in blood", "Haemoglobin."),
        ],
    },
    "lungs": {
        "title": "The Lungs",
        "text": "The lungs are the organs of respiration that exchange oxygen and carbon dioxide "
        "with the blood. Air travels through the trachea into bronchi and then to tiny sacs "
        "called alveoli where gas exchange happens.",
        "probes": [
            (
                "the function of the lungs",
                "They exchange oxygen and carbon dioxide with the blood.",
            ),
            ("where gas exchange happens in the lungs", "In tiny sacs called alveoli."),
            ("the airway leading into the lungs", "The trachea."),
            ("the gases the lungs exchange", "Oxygen and carbon dioxide."),
            ("the branches of the airway in the lungs", "The bronchi."),
        ],
    },
    "kidneys": {
        "title": "The Kidneys",
        "text": "The kidneys are two bean-shaped organs that filter waste and excess water from "
        "the blood to make urine. They also help regulate blood pressure and salt balance. "
        "Each kidney contains about a million filtering units called nephrons.",
        "probes": [
            ("the function of the kidneys", "They filter waste and excess water from the blood."),
            ("what the kidneys produce", "Urine."),
            ("the filtering units of the kidney", "Nephrons."),
            ("what the kidneys help regulate", "Blood pressure and salt balance."),
            ("the shape of the kidneys", "Bean-shaped."),
        ],
    },
    "nile": {
        "title": "The Nile River",
        "text": "The Nile is one of the longest rivers in the world, flowing northward through "
        "northeastern Africa into the Mediterranean Sea. It was the lifeline of ancient Egypt, "
        "whose civilisation depended on its annual floods.",
        "probes": [
            ("the length status of the Nile River", "One of the longest rivers in the world."),
            ("the direction the Nile flows", "Northward."),
            ("where the Nile empties", "Into the Mediterranean Sea."),
            ("the civilisation that depended on the Nile", "Ancient Egypt."),
            ("why the Nile mattered to ancient Egypt", "Its annual floods sustained farming."),
        ],
    },
    "sahara": {
        "title": "The Sahara Desert",
        "text": "The Sahara is the largest hot desert in the world, covering much of North "
        "Africa. It is roughly the size of the United States. Despite extreme daytime heat, "
        "temperatures can drop sharply at night.",
        "probes": [
            ("the largest hot desert in the world", "The Sahara."),
            ("the region the Sahara covers", "North Africa."),
            ("the size of the Sahara", "Roughly the size of the United States."),
            ("the temperature swing in the Sahara", "Hot days and sharply colder nights."),
            ("the continent of the Sahara Desert", "Africa."),
        ],
    },
    "rainforest": {
        "title": "The Amazon Rainforest",
        "text": "The Amazon rainforest is the largest tropical rainforest on Earth, often "
        "called the lungs of the planet for the oxygen it produces. It holds enormous "
        "biodiversity and plays a key role in regulating the global climate.",
        "probes": [
            ("the largest tropical rainforest", "The Amazon rainforest."),
            (
                "why the Amazon is called the lungs of the planet",
                "It produces large amounts of oxygen.",
            ),
            ("what the Amazon rainforest is known for", "Its enormous biodiversity."),
            ("the climate role of the Amazon rainforest", "It helps regulate the global climate."),
            ("the type of forest the Amazon is", "A tropical rainforest."),
        ],
    },
    "democracy": {
        "title": "Democracy",
        "text": "Democracy is a system of government in which power comes from the people, "
        "usually through free and fair elections. In a representative democracy, citizens elect "
        "officials to make decisions on their behalf. It originated in ancient Athens.",
        "probes": [
            ("what democracy is", "A system where power comes from the people."),
            ("how power is expressed in a democracy", "Through free and fair elections."),
            ("what a representative democracy is", "Citizens elect officials to decide for them."),
            ("where democracy originated", "In ancient Athens."),
            ("the source of authority in a democracy", "The people."),
        ],
    },
    "enzymes": {
        "title": "Enzymes",
        "text": "Enzymes are biological catalysts, usually proteins, that speed up chemical "
        "reactions in living things without being consumed. Each enzyme acts on a specific "
        "substrate, fitting it like a lock and key.",
        "probes": [
            ("what enzymes are", "Biological catalysts that speed up reactions."),
            ("the molecules most enzymes are made of", "Proteins."),
            ("how an enzyme matches its substrate", "Like a lock and key."),
            ("whether enzymes are used up in reactions", "No, they are not consumed."),
            ("the effect of enzymes on chemical reactions", "They speed them up."),
        ],
    },
    "proteins": {
        "title": "Proteins",
        "text": "Proteins are large molecules built from chains of amino acids folded into "
        "specific shapes. They do most of the work in cells, acting as enzymes, structural "
        "components, and signals. There are 20 standard amino acids.",
        "probes": [
            ("what proteins are made of", "Chains of amino acids."),
            ("the role of proteins in cells", "They do most of the cell's work."),
            ("the number of standard amino acids", "Twenty."),
            ("what determines a protein's function", "Its folded shape."),
            ("examples of what proteins act as", "Enzymes, structures, and signals."),
        ],
    },
    "carbon_cycle": {
        "title": "The Carbon Cycle",
        "text": "The carbon cycle is the movement of carbon between the atmosphere, living "
        "things, oceans, and rocks. Plants take in carbon dioxide during photosynthesis, and "
        "respiration and burning fuels release it back.",
        "probes": [
            ("the carbon cycle", "The movement of carbon among air, life, oceans, and rocks."),
            ("how plants take in carbon", "Through photosynthesis."),
            ("how carbon returns to the atmosphere", "Through respiration and burning fuels."),
            ("the reservoirs of the carbon cycle", "Atmosphere, living things, oceans, and rocks."),
            ("the gas plants absorb in the carbon cycle", "Carbon dioxide."),
        ],
    },
    "greenhouse_effect": {
        "title": "The Greenhouse Effect",
        "text": "The greenhouse effect is the warming of a planet when gases like carbon dioxide "
        "trap heat radiated from the surface. It keeps Earth warm enough for life, but extra "
        "greenhouse gases intensify it and drive climate change.",
        "probes": [
            ("the greenhouse effect", "Warming when gases trap heat near a planet's surface."),
            ("the gases that cause the greenhouse effect", "Greenhouse gases like carbon dioxide."),
            (
                "the benefit of the natural greenhouse effect",
                "It keeps Earth warm enough for life.",
            ),
            (
                "how extra greenhouse gases affect climate",
                "They intensify warming and drive climate change.",
            ),
            ("what greenhouse gases trap", "Heat radiated from the surface."),
        ],
    },
    "plate_tectonics": {
        "title": "Plate Tectonics",
        "text": "Plate tectonics is the theory that Earth's outer shell is divided into moving "
        "plates. Their interactions build mountains, cause earthquakes, and form volcanoes. "
        "The plates float on the semi-fluid mantle beneath.",
        "probes": [
            ("the theory of plate tectonics", "Earth's outer shell is divided into moving plates."),
            ("what plate interactions produce", "Mountains, earthquakes, and volcanoes."),
            ("what the tectonic plates float on", "The semi-fluid mantle."),
            ("Earth's outer shell in plate tectonics", "It is broken into plates."),
            ("a geological result of moving plates", "Mountain building."),
        ],
    },
    "magnetism": {
        "title": "Magnetism",
        "text": "Magnetism is a force produced by moving electric charges. Magnets have north "
        "and south poles, and like poles repel while opposite poles attract. Earth itself acts "
        "like a giant magnet, which is why compasses point north.",
        "probes": [
            ("what produces magnetism", "Moving electric charges."),
            ("the poles of a magnet", "North and south."),
            ("how magnetic poles interact", "Like poles repel; opposite poles attract."),
            ("why a compass points north", "Earth acts like a giant magnet."),
            ("the force magnetism describes", "The force between magnetic poles and charges."),
        ],
    },
    "sound": {
        "title": "Sound",
        "text": "Sound is a vibration that travels as a wave through a medium such as air, water, "
        "or solids. It cannot travel through a vacuum. Its pitch depends on frequency and its "
        "loudness on amplitude. In air it travels about 343 metres per second.",
        "probes": [
            ("what sound is", "A vibration that travels as a wave through a medium."),
            (
                "why sound cannot travel through a vacuum",
                "It needs a medium to carry the vibration.",
            ),
            ("what determines the pitch of a sound", "Its frequency."),
            ("what determines the loudness of a sound", "Its amplitude."),
            ("the speed of sound in air", "About 343 metres per second."),
        ],
    },
    "optics": {
        "title": "Light and Optics",
        "text": "Light is electromagnetic radiation visible to the eye. It travels in straight "
        "lines, reflects off surfaces, and bends, or refracts, when passing between materials. "
        "A prism splits white light into the colours of the spectrum.",
        "probes": [
            ("what light is", "Electromagnetic radiation visible to the eye."),
            ("what happens when light bounces off a surface", "It reflects."),
            ("the bending of light between materials", "Refraction."),
            ("what a prism does to white light", "Splits it into the spectrum of colours."),
            ("how light travels through a uniform medium", "In straight lines."),
        ],
    },
    "relativity": {
        "title": "Relativity",
        "text": "Einstein's theory of relativity reshaped physics. Special relativity says the "
        "speed of light is constant and time slows for fast-moving observers. General "
        "relativity describes gravity as the curving of spacetime by mass.",
        "probes": [
            ("who developed the theory of relativity", "Albert Einstein."),
            ("what special relativity says about light", "Its speed in a vacuum is constant."),
            ("the time effect in special relativity", "Time slows for fast-moving observers."),
            ("how general relativity describes gravity", "As the curving of spacetime by mass."),
            ("the two parts of relativity", "Special and general relativity."),
        ],
    },
    "quantum": {
        "title": "Quantum Mechanics",
        "text": "Quantum mechanics describes nature at the scale of atoms and particles, where "
        "energy comes in discrete packets called quanta. Particles can behave like waves, and "
        "the uncertainty principle limits how precisely position and momentum are known.",
        "probes": [
            ("what quantum mechanics describes", "Nature at the scale of atoms and particles."),
            ("the discrete packets of energy in quantum mechanics", "Quanta."),
            ("the wave-like behaviour of particles", "Particles can act like waves."),
            ("the uncertainty principle", "A limit on knowing position and momentum together."),
            ("the scale quantum mechanics applies to", "Atoms and subatomic particles."),
        ],
    },
}


async def _ids_for(slug: str) -> tuple[str, list[str]]:
    """Parse + chunk a corpus file, returning its doc id and chunk ids."""
    path = CORPUS_DIR / f"{slug}.md"
    doc = await parse_path(str(path))
    chunks = CHUNKER.chunk(doc)
    return doc.doc_id, [c.chunk_id for c in chunks]


def _build_split(key: str, ids: dict[str, tuple[str, list[str]]]) -> list[GoldQuery]:
    """All queries for one split: each topic × probe × the split's templates.

    Difficulty is cycled across the split's emission order so easy/medium/hard
    are all represented (SPEC §11 stratification). ``query_id`` encodes the
    topic, probe, and template so ids are unique and the provenance is legible.
    """
    template_idxs = _SPLIT_TEMPLATES[key]
    out: list[GoldQuery] = []
    diff_counter = 0
    for slug, topic in TOPICS.items():
        doc_id, chunk_ids = ids[slug]
        for p_idx, (subject, answer) in enumerate(topic["probes"]):
            for t_idx in template_idxs:
                query = _TEMPLATES[t_idx].format(s=subject)
                out.append(
                    GoldQuery(
                        query_id=f"{key}-{slug}-{p_idx}-{t_idx}",
                        query=query,
                        relevant_chunk_ids=chunk_ids,
                        relevant_doc_ids=[doc_id],
                        expected_answer=answer,
                        intent="lookup",
                        difficulty=_DIFFICULTIES[diff_counter % 3],  # type: ignore[arg-type]
                        notes=f"seed topic: {slug}",
                    )
                )
                diff_counter += 1
    return out


async def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Write the fixture corpus.
    for slug, t in TOPICS.items():
        (CORPUS_DIR / f"{slug}.md").write_text(f"# {t['title']}\n\n{t['text']}\n")

    # 2. Resolve deterministic gold ids per topic from the written files.
    ids: dict[str, tuple[str, list[str]]] = {}
    for slug in TOPICS:
        ids[slug] = await _ids_for(slug)

    # 3. Build each split from the disjoint template partition.
    splits = {name: _build_split(name, ids) for name in ("dev", "rotating", "frozen")}

    for name, queries in splits.items():
        path = SEED_DIR / f"{name}.jsonl"
        with path.open("w") as fh:
            for gq in queries:
                fh.write(json.dumps(gq.model_dump(), ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(SEED_DIR.parents[2])}: {len(queries)} queries")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
