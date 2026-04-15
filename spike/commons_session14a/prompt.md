# Compression Task

You are being asked to compress a corpus of episodes from an AI agent's memory
system. The corpus is in the attached file `corpus_for_agents.md` — 30 episodes
numbered `[1]` through `[30]`, each with a type (finding/decision/observation/
connection) and a date. Read the corpus carefully before answering.

Your job is to produce a short structured compression of the corpus. There is
no hidden agenda, no adversarial game, no review task — just an honest attempt
to distill what these episodes collectively say and which episodes belong
together.

## Output Format (STRICT — follow exactly)

Your response MUST contain these three sections with these exact headers, in
this order, and nothing else. No preamble, no closing remarks, no meta-commentary
about the task.

### SUMMARY

A 150-300 word prose compression of what these episodes collectively say.
Narrative, no bullet points, no citation markers. Write it as if you were
handing a continuity note to someone who will pick up this work tomorrow.

### CITED CONNECTIONS

A list of episode pairs that you judge belong together, in this exact format:

    (a, b): one-line reason these belong together

One pair per line. Use `(smaller_idx, larger_idx)` ordering (e.g. `(3, 17)`,
not `(17, 3)`). Your `why` should be concrete — a specific semantic or
structural reason, not a generic "both about X" gloss.

Cite as many pairs as your judgment says. There is no minimum and no maximum.
Do not pad. Do not withhold. If two episodes genuinely belong together, cite
the pair. If they don't, don't.

Only cite pairs where BOTH episodes carry real connective weight. A pair you
cite should be one where understanding one episode is materially sharpened by
reading the other.

### TOP PATTERNS

3 to 5 cross-episode patterns, one sentence each, numbered:

    1. Pattern sentence
    2. Pattern sentence
    3. Pattern sentence

A pattern is a compact claim that holds across multiple episodes — something
the corpus is collectively showing that no single episode shows alone.

## Reminders

- Use the bracketed indices `[1]..[30]` to refer to episodes.
- Do NOT invent episode indices outside `[1]..[30]`.
- Do NOT include any section other than the three above.
- Write the compression from your own priors — do not try to guess what
  someone else would write. The whole point of this exercise is independent
  judgment.
