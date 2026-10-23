SYSTEM = (
    "You are the planner for a tabletop robot arm. Images show a table from above. "
    "Objects are marked with numbered white circles. Bins are labelled A and B. "
    "Always answer with a single JSON object and nothing else."
)

TEACH = """These {n} frames were taken in order while a person sorted objects into bins.
Work out the sorting RULE they used (e.g. by category, colour, material, size).
Return JSON: {{"rule": "<one sentence>", "bins": {{"A": "<what goes in A>", "B": "<what goes in B>"}}, "say": "<short sentence to speak aloud>"}}"""

PLAN = """Sorting rule: {rule}
Bin A: {bin_a}. Bin B: {bin_b}.
Numbered objects still on the table: {ids}.
Pick ONE object to move next and choose its bin.
Return JSON: {{"object": <number>, "label": "<what it is>", "bin": "A" or "B", "reason": "<few words>"}}
If no numbered object fits either bin, return {{"object": null, "say": "<why>"}}"""

CORRECT = """Current rule: {rule}. Bin A: {bin_a}. Bin B: {bin_b}.
The user said: "{text}"
Update the rule. Return JSON: {{"rule": "...", "bins": {{"A": "...", "B": "..."}}, "say": "<confirm in one sentence>"}}"""
