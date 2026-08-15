import { useState } from "react";

// Plain-language walkthrough of the whole project, written so a first-time reader (no
// stats or ML background assumed) can follow how the training data was built (Phase 1),
// how the detector actually measures text (Phase 2), and what the numbers on this page
// mean. Sourced directly from PHASE1_PROCESS.md and PHASE2_PROCESS.md - the technical
// versions of this same story, with every experiment, bug, and number - this is the
// short, friendly retelling, not a replacement for those.
export default function HowItWorks() {
  const [open, setOpen] = useState(true);

  return (
    <section className="card howitworks-card">
      <button
        type="button"
        className="howitworks-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span>How this tool actually works (read this first)</span>
        <span className="howitworks-chevron">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="howitworks-body">
          <p className="howitworks-lede">
            This tool doesn't ask another AI "does this look AI-written?" and repeat its
            guess back to you — that would be a black box, unreliable, and impossible to
            explain. Instead, it counts real, measurable facts about the writing itself
            (like how predictable the word choices are) and compares those numbers to
            what thousands of real human and AI sentences actually look like. Here's the
            whole story, in plain English.
          </p>

          <h3 className="howitworks-h">
            Step 1 — Building a practice set of essays it could learn from
          </h3>
          <p>
            Before this tool could measure anything, it needed real examples to compare
            against — like a student needing practice essays before an exam, not just a
            textbook rule. 300 essays were collected, in three groups:
          </p>
          <ul className="howitworks-list">
            <li><strong>100 real essays</strong>, written entirely by real students.</li>
            <li><strong>100 essays written entirely by AI</strong>, start to finish.</li>
            <li>
              <strong>100 "hybrid" essays</strong> — a real student's essay where a
              portion of the sentences or paragraphs were rewritten/polished by AI, and
              the rest left untouched. This is the tricky, realistic case: not all-human
              or all-AI, but mixed, like one swapped ingredient hidden in a real recipe.
              Every single sentence in these essays was individually labeled — untouched
              original, or AI-edited — by directly comparing each hybrid against the
              student's original draft.
            </li>
          </ul>
          <p>
            This set was then cleaned up: broken or duplicate essays were removed, and
            essay lengths were capped at a comparable size across all three groups — so
            length alone couldn't give away which group an essay belonged to. Finally, it
            was split into a "study set" (used to build and tune everything below) and a
            locked-away "test set" that was only opened once, at the very end, to check
            honestly how well the finished tool actually performs on essays it had never
            seen.
          </p>

          <h3 className="howitworks-h">
            Step 2 — Teaching it to actually notice the difference
          </h3>
          <p>
            This is where GPT-2 comes in — an older, freely available AI language model.
            Importantly, it's <strong>never asked "is this AI-written?"</strong> It's used
            only as a measuring instrument, like a thermometer: it reads a sentence and
            reports how surprised it was by each word choice, nothing more.
          </p>
          <p>From that, seven separate measurements ("clues") are computed for every sentence:</p>
          <ul className="howitworks-list">
            <li>
              <strong>Word predictability</strong> — feed a sentence in, and for every
              word GPT-2 says "here's how likely I thought that word was, given
              everything before it." Averaged across the sentence, this is called
              <em> perplexity</em>: how surprising the wording was overall.
            </li>
            <li>
              <strong>Predictability swings</strong> — how much that word-by-word
              surprise level jumps around within one sentence, instead of staying flat.
            </li>
            <li>
              <strong>Sentence-length variation</strong>, <strong>grammatical-complexity
              variation</strong>, and <strong>word-choice variation</strong> — measured
              across the whole essay. Real writers naturally speed up, slow down, write a
              short punchy line, then a long complicated one — this unevenness is called
              "burstiness." Very machine-smooth writing can be unusually even.
            </li>
            <li>
              <strong>Phrase repetition</strong> and <strong>transition-word
              repetition</strong> — how often the same short phrase, or the same
              connector word ("however," "moreover"), gets reused across the essay.
            </li>
          </ul>
          <p>
            Each of these seven raw numbers is then converted into "how far from typical
            human writing is this?" — the same idea as converting a height in
            centimeters into "taller than most people." This is called a{" "}
            <strong>z-score</strong>: 0 means exactly average for human essays, a bigger
            positive or negative number means further from typical.
          </p>
          <p>
            Finally, instead of a person guessing which of the seven clues matter most,
            the computer was shown thousands of real, labeled sentences (human vs. AI)
            and <strong>learned a formula on its own</strong> — giving more weight to
            whichever clues actually separated human from AI in the practice data, and
            little or none to the clues that turned out not to matter. This method is
            called <em>logistic regression</em>, and — importantly — the weights it
            learned are fully visible, not a hidden black box. That's what makes it
            possible to show you, per sentence, exactly which clues pushed the score up
            or down, and by how much.
          </p>

          <h3 className="howitworks-h">
            Why there are two separate scores, not one
          </h3>
          <p>
            One honest complication was discovered along the way: five of the seven
            clues above (everything except word predictability and its swings) only
            really make sense averaged across a <em>whole</em> essay — they describe a
            property of the whole piece of writing, not one sentence. That's fine for
            judging an entire essay, but it creates a blind spot: if just one paragraph
            out of five was AI-polished, the four genuinely-human paragraphs can "outvote"
            it in the average, and the AI-touched part can hide.
          </p>
          <p>So this tool shows two separate signals, not one:</p>
          <ul className="howitworks-list">
            <li>
              <strong>The main, whole-essay check</strong> (all 7 clues) — the big
              percentage at the top of the results. Well-tested and reliable: it catches
              fully AI-written essays almost every time, and does a solid job on genuine
              human essays.
            </li>
            <li>
              <strong>A second, weaker, sentence-only check</strong> — built specifically
              to try to catch one AI-edited sentence hiding inside an otherwise human
              essay, using only the two clues that genuinely work one sentence at a time.
              It's shown honestly as a secondary, lower-confidence signal (its real,
              tested error rate is stated directly in the app) — never the headline
              number.
            </li>
          </ul>

          <h3 className="howitworks-h">What you're actually looking at on this page</h3>
          <ul className="howitworks-list">
            <li>The big percentage at the top — the main, whole-essay check.</li>
            <li>
              A sentence gets a solid highlight when its own contribution to that
              whole-essay check is high enough, with the exact percentage shown right
              next to it.
            </li>
            <li>
              A thin underline under a sentence — the second, weaker, sentence-only
              check.
            </li>
            <li>
              A percentage above each paragraph — that paragraph's own average of the
              main check.
            </li>
            <li>
              An occasional <strong>⚠ "statistically unusual paragraph"</strong> badge —
              a newer, separate, experimental check that compares one paragraph's writing
              style only against the <em>other paragraphs in that same essay</em> (not
              against any training data). It flags "this looks different from the rest of
              this essay," never "this is the AI part" — click it to see why, including a
              plain-worded reminder that "different" isn't proven to mean "AI."
            </li>
            <li>
              Click the ⓘ on any sentence to see all seven clues individually, each with
              a plain-English description, a 0-10 "how unusual" score, and the real
              measured value behind it.
            </li>
          </ul>

          <h3 className="howitworks-h">Being honest about the limits</h3>
          <ul className="howitworks-list">
            <li>Very strong on essays that are entirely AI-written.</li>
            <li>Solid, not perfect, on genuine human essays — it gets roughly 1 in 5 wrong.</li>
            <li>
              Noticeably weaker at spotting a few AI-polished sentences hidden inside an
              otherwise human essay — that's the hardest, most realistic case, and it's
              disclosed here rather than hidden.
            </li>
            <li>
              Measurably less accurate for writers who learned English as a second
              language, because the underlying model was trained mostly on fluent
              native-English text, so unusual (but completely legitimate) phrasing can
              register as "surprising." This is flagged directly in the fairness note
              above, not swept under the rug.
            </li>
          </ul>
          <p className="howitworks-footer">
            None of this is a verdict — it's a set of statistical clues, shown with their
            real, tested track record, so you can make an informed judgment instead of
            being handed someone else's. The full technical write-up — every experiment,
            every bug found and fixed, every real number — lives in this project's{" "}
            <code>PHASE1_PROCESS.md</code> and <code>PHASE2_PROCESS.md</code> files, if
            you want the whole story.
          </p>
        </div>
      )}
    </section>
  );
}
