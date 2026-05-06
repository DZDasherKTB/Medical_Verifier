from verification.reasoning_verifier import ReasoningVerifier
from verification.proposition_verifier import PropositionVerifier
import json

reasoning_trace = """
1. The urogenital diaphragm is a structure located in the pelvic region, specifically within the perineum. It is important to understand its composition to determine which structures contribute to its formation.

2. The urogenital diaphragm is primarily composed of the deep transverse perineal muscle and the perineal membrane. These structures are located in the deep perineal
pouch, which is a part of the pelvic floor.

3. The perineal membrane is a layer of fascia that provides support and is part of the urogenital diaphragm. It is important to distinguish between the deep fascia, which contributes to the diaphragm, and the superficial fascia, which does not.

4. Colle's fascia is a type of superficial fascia found in the perineal region. It is continuous with the superficial fascia of the abdominal wall and is located more superficially than the structures forming the urogenital diaphragm.

5. Since the urogenital diaphragm is composed of deeper structures, such as the deep transverse perineal muscle and the perineal membrane, superficial fascia like Colle's fascia does not contribute to its formation.
"""

propositions = """
1. The urogenital diaphragm is a structure located in the pelvic region, specifically within the perineum. It is important to understand its composition to determine which structures contribute to its formation.

2. The urogenital diaphragm is primarily composed of the deep transverse perineal muscle and the perineal membrane. These structures are located in the deep perineal
pouch, which is a part of the pelvic floor.

3. The perineal membrane is a layer of fascia that provides support and is part of the urogenital diaphragm. It is important to distinguish between the deep fascia, which contributes to the diaphragm, and the superficial fascia, which does not.

4. Colle's fascia is a type of superficial fascia found in the perineal region. It is continuous with the superficial fascia of the abdominal wall and is located more superficially than the structures forming the urogenital diaphragm.

5. Since the urogenital diaphragm is composed of deeper structures, such as the deep transverse perineal muscle and the perineal membrane, superficial fascia like Colle's fascia does not contribute to its formation.
"""

hypotheses = [
    "H01: Deep transverse perineal muscle is a part of urogenital diaphragm",
    "H02: Colle's fascia is a part of urogenital diaphragm",
    "H03: Sphincter urethrae is a part of urogenital diaphragm"
]

verifier = ReasoningVerifier()

result = verifier.verify(
    reasoning_trace=reasoning_trace,
    hypotheses=hypotheses
)

print("Reas Verifier: ",result)

verifier = PropositionVerifier()

result = verifier.verify(
    propositions=propositions,
    hypotheses=hypotheses
)

print("Prop Verifier: ",result)