# Ethics

## The genuine harm vector
The realistic way this project could hurt someone is if wearable-derived cycle estimates were read as
**contraception or fertility guidance**. They are not reliable enough for that, and using them that way
is unsafe. This is handled in three places, not by hopeful disclaimers:
- the LLM explanation layer **hard-refuses** contraception, diagnosis, and treatment questions
  (`src/infradian/llm/guard.py`), verified by a red-team eval;
- the app surfaces "not a diagnostic device" **structurally**, at the same weight as content;
- the model card's out-of-scope section states it plainly.

## Data governance
- **mcPHASES is not redistributed.** No raw rows, no per-participant figures, no mcPHASES-trained
  checkpoint. Only a loader, file checksums, salted-hash split manifest, and aggregate metrics with a
  **minimum cell size ≥ 5** suppression rule. A pre-commit hook blocks any data commit.
- The salted-hash split manifest cannot be inverted usefully; non-redistribution — not obscurity — is
  what protects privacy.
- NHANES-derived tables carry the NCHS no-re-identification obligation forward in their card.

## Representation
The clinical cohort is 42 Canadian adults aged 18–29. Publishing a benchmark on such a narrow group
risks implying broader validity. We state the scope prominently everywhere (README, app footer, methods
page, LIMITATIONS.md) and refuse to extrapolate to menopause or PCOS-typical populations.

## Honesty over hype
The real-data primary endpoint is a **null**, and we report it as the headline rather than a
cherry-picked positive cell. A benchmark that only publishes wins is not measuring anything.

## Licensing choice
We considered OpenRAIL for the weights and rejected it (non-OSI, unenforceable, blocks legitimate
reuse). Misuse is addressed by design, not by license text. Weights are Apache-2.0.
