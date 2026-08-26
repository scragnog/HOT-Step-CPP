# Writing style

Rules for prose a human reads: GitHub issue replies, commit messages, PR bodies,
release notes, README and docs text, in-app copy. Code identifiers and API names
are exempt, they must match the codebase.

Anything published under the project's name follows this. That includes replies
written by an agent on a maintainer's behalf.

## No emojis

No emojis anywhere in prose. Not in headings, not as bullet markers, not as a
sign-off, not as a reaction. This includes the decorative ones that look
harmless in a heading.

The one exception is quoting something that already contained an emoji, or
describing UI copy that ships with one.

## No AI tells

The full checklist lives in the `unslop` skill at `~/.claude/skills/unslop/`.
The ones that matter most here:

No em dashes. Use a comma or end the sentence. Reaching for parentheses or an
en dash instead just swaps one tell for another.

No "not just X, but Y". Say the thing directly.

No rule-of-three padding. If there are two reasons, give two.

No inline-header bullets where a bold label and colon restate the line that
follows. Write it as prose.

No opening flattery. Skip "Great catch", "Great question", "You're absolutely
right". Answer the question.

No closing filler. Skip "Hope this helps", "Let me know if you have any
questions", "Feel free to reach out". Stop when the answer is finished.

No puffery vocabulary: crucial, delve, robust, seamless, leverage, utilize,
comprehensive, landscape, tapestry, testament, underscore, showcase, pivotal,
enhance. Plain words instead. "Use", not "leverage". "Help", not "facilitate".

Sentence case in headings, not Title Case.

Straight quotes, not curly.

## Tone for issue replies

Say what was found, what changed, and what the reporter should do next. In that
order. Short is fine.

Be honest about confidence. If a cause is a hypothesis, say it is a hypothesis.
If a fix is unverified on the reporter's platform, say so. Never claim something
is fixed when only the code changed and nothing was tested.

Own mistakes plainly. "This is our bug" beats a paragraph of apology.

Credit people who diagnosed the problem or supplied the fix, by handle.

Give the version or commit the fix lands in, so the reporter knows when to
expect it.

## Examples

Bad:

> Great catch! 🎯 This is definitely a crucial issue — we've now implemented a
> comprehensive fix that not only resolves the crash but also enhances the
> overall robustness of the download pipeline. Let me know if you have any
> other questions!

Good:

> Confirmed. The `minimax-music3-balanced` entry in the model catalogue was
> missing its `fileIds` array, so the Model Manager threw on `undefined.map`
> and the whole page failed to render. Fixed in a1b2c3d, plus a guard so one
> malformed entry can no longer blank the page.
>
> Thanks to @dabuliukas for finding the exact field.
