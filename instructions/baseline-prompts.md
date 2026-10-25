# Prompts of the LLM-as-a-judge baselines

The organisers' LLM baselines (the Gemma 4, Qwen 3.6, GPT-5.6 and Gemini 3.6 Flash runs in the
`baselines` config) follow a two-stage GEMBA-ESA protocol. This file transcribes, word for word,
Appendix E of the findings paper (Lavie et al., 2026, *Proceedings of the Eleventh Conference on
Machine Translation (WMT)*, pages 1006-1009): the shared system prompt and decoding settings, the
Stage 1 prompt (Figure 7), the Stage 2 prompt (Figure 8), the domain requirements (Table 63) and
the reference-type labels (Table 64). Only the Markdown layout and the lines in square brackets
that say they are notes are ours.

The paper typesets quotation marks as curly quotes and writes the triple backticks of the
templates as three typewriter backticks. Where a template is copied into code, check that the
quotation marks are the plain ASCII ones that the model should see.

## Shared system prompt and decoding (Appendix E)

> The LLM-based baselines follow the two-stage GEMBA-ESA prompting strategy. In Stage 1 (Task 1),
> the model identifies and classifies translation errors using MQM-style categories. In Stage 2
> (Task 2), the model assigns a quality score on a 0–100 scale given the source, translation,
> and the error annotations produced by Stage 1. Both stages use a shared system prompt:

~~~~text
Your task is to identify machine translation errors and assess the quality of the translation.
~~~~

> Each stage is issued as a single-turn chat message (system + user). Stage 1 uses
> chain-of-thought decoding when enabled (temperature 0.6, top-p 0.95, max 8192 tokens);
> otherwise greedy decoding (temperature 0). Stage 2 always uses greedy decoding with a 64-token
> limit and thinking disabled.

## Stage 1: error annotation, Task 1 (Appendix E.1, Figure 7)

> Figure 7 shows the Stage 1 prompt template. When a reference translation is available
> (-with-ref), the shaded reference block is inserted between the translation and the annotation
> instructions; otherwise it is omitted. The reference header adapts to the reference type
> (Table 64).

**Figure 7: Stage 1 Prompt Template (User Turn)**

~~~~text
{src_name} source:
```{src_text}```
{tgt_name} translation:
```{hyp_text}```
~~~~

Block inserted only when `-with-ref` is active:

~~~~text
{ref_label}:
```{ref_text}```
~~~~

Instructions:

~~~~text
Based on the source segment{, machine translation, and reference translation | and machine translation} surrounded by triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
Each error is classified as one of two categories: major or minor. Major errors disrupt the flow and make the understandability of the text difficult or impossible. Minor errors are errors that do not disrupt the flow significantly, and what the text is trying to say is still understandable.
Return only the annotations in this format:
Major:
category/subcategory - “error span”
Minor:
category/subcategory - “error span”
Use one error per line and write no-error when a section is empty. Quote spans from the translation; for omissions, quote the omitted source span.
Domain requirements: {domain_req}
~~~~

> Figure 7: Stage 1 (Task 1) error annotation prompt template. Placeholders in braces are filled
> per-segment. The gray inner box is included only in the reference-augmented variant. The
> instruction wording adapts accordingly (gray alternation text).

[Note: the paper prints the two alternatives inside the braces in grey. The first, ", machine
translation, and reference translation", is the wording with a reference, and the second, " and
machine translation", the wording without one.]

## Stage 2: quality scoring, Task 2 (Appendix E.2, Figure 8)

> Figure 8 shows the Stage 2 prompt template. The {error_spans} placeholder is filled with the
> parsed Stage 1 output. As with Stage 1, the reference block is included only when -with-ref is
> active.

**Figure 8: Stage 2 Prompt Template (User Turn)**

~~~~text
Given the translation from {src_name} to {tgt_name} and the annotated error spans, assign a score on a continuous scale from 0 to 100. The scale has the following reference points: 0=“No meaning preserved”, 33=“Some meaning preserved”, 66=“Most meaning preserved and few grammar mistakes”, up to 100=“Perfect meaning and grammar”.
Domain requirements: {domain_req}
Score the following translation:
{src_name} source:
```{src_text}```
{tgt_name} translation:
```{hyp_text}```
~~~~

Block inserted only when `-with-ref` is active:

~~~~text
{ref_label}:
```{ref_text}```
~~~~

Then:

~~~~text
Annotated error spans:
```{error_spans}```
Respond with ONLY a valid JSON object and nothing else: {“score”: N}
where N is an integer from 0 to 100.
~~~~

> Figure 8: Stage 2 (Task 2) quality scoring prompt template. The {error_spans} field contains
> the Stage 1 annotation output. The gray inner box is included only in the reference-augmented
> variant.

## Domain requirements (Appendix E.3, Table 63)

> Each segment carries a domain label extracted from its item_id. The domain-specific instruction
> text appended to both Stage 1 and Stage 2 prompts is shown in Table 63.

**Table 63: Domain requirement text inserted into both Stage 1 and Stage 2 prompts. The news and
factchecking domains share identical requirements.**

| Domain | Requirement text |
| --- | --- |
| news / factchecking | The source segment is from a news article. The translation should use a formal register consistent with journalistic standards and preserve the source HTML formatting. |
| speech | The source segment is a transcript of spoken content from a video. The translation should preserve the speaker’s flow and colloquial style. It should omit non-linguistic sounds, such as laughter, groans, and hesitation sounds, while retaining interjections. Interrupted words should be completed when they can be inferred from context; otherwise, they should be omitted. Foreign words should remain unchanged. Each sentence should be placed on a separate line. |
| social | The source segment is user-generated content from a social media platform. Source spelling mistakes should not be reproduced. Meaningful expressiveness, such as capitalization or elongation, should be reproduced naturally in the target language. URLs and user handles should be copied unchanged, while hashtags should be translated when appropriate. Source punctuation should be followed as closely as possible, with additional punctuation only when needed to prevent serious loss of comprehension. The translation should use an informal style, like close friends talking, even if this changes the original tone, and preserve the source HTML formatting. |
| software | The source segment contains software data from a JSON. Only JSON content or values should be translated; keys and placeholders should be copied unchanged. The translation should contain only valid JSON content matching the input format. |
| edu | The source segment consists of biology, chemistry, and geography exercises from an educational web portal for children aged 9–16. The translation should be suitable for this educational context and age range, and preserve the source HTML formatting. |
| general | The translation should be accurate and fluent. |

## Reference type labels (Appendix E.4, Table 64)

> When the reference-augmented variant is used, the reference block header adapts to the type of
> reference available for each segment, as shown in Table 64.

**Table 64: Reference type labels used in prompt headers.**

| Type | Prompt label |
| --- | --- |
| human | Human reference translation |
| postedit | Post-edited reference translation |
| pseudo | High-quality machine translation reference |
