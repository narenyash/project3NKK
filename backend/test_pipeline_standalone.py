"""
Step 2: standalone test of analyze_essay() (pipeline.py) on fresh, non-dataset essays -
imports and calls the function directly in-process, unlike integration_test.py which
sends real HTTP requests to a separately-running server. Useful for quickly checking the
pipeline's behavior on hand-written test text without needing the FastAPI server up.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import analyze_essay

HUMAN_ESSAY = """
My grandmother's kitchen smelled like burnt garlic and cardamom, and I hated it until I was about fourteen.
I used to sit on the cold tile floor doing homework while she yelled at the stove in a language I only half understood.
One Tuesday she handed me a wooden spoon and told me to stir, and I burned the onions immediately.
She laughed so hard she had to sit down. I don't really know why that moment stuck with me more than the ones where
things actually went well. Maybe it's because it was the first time I saw her as someone who found things funny instead
of someone who was just always working. After that I started showing up on Tuesdays on purpose, not because my parents
made me, and not because I was particularly good at cooking, because I wasn't. I just liked being yelled at in a kitchen
that smelled like garlic. It took me years to realize that's basically what I mean when I say I miss her.
"""

AI_ESSAY = """
In today's rapidly evolving technological landscape, artificial intelligence has emerged as a transformative force
across numerous industries. From healthcare to finance, organizations are increasingly leveraging machine learning
algorithms to optimize their operations and enhance decision-making processes. This paradigm shift represents a
significant departure from traditional methodologies, offering unprecedented opportunities for innovation and growth.
Furthermore, the integration of artificial intelligence into everyday applications has fundamentally reshaped how
individuals interact with technology. As these systems continue to advance, it becomes increasingly important for
stakeholders to consider the ethical implications of widespread AI adoption. Moreover, organizations must carefully
balance the pursuit of efficiency with the need for transparency and accountability. In conclusion, artificial
intelligence represents both tremendous opportunity and significant responsibility for society moving forward.
"""

MIXED_ESSAY = """
When I was twelve my dad taught me to fix the lawnmower in our garage every Saturday morning, mostly because he
didn't trust anyone else to do it right. Artificial intelligence has since transformed numerous industries by
enabling more efficient data-driven decision making across diverse sectors of the economy. I still remember the
smell of gasoline and the way his hands were always stained black no matter how many times he washed them. The
integration of these advanced technologies into everyday workflows represents a significant paradigm shift for
organizations seeking to maintain competitive advantage. He never once explained what he was doing, just handed
me tools and expected me to figure it out, which used to make me furious.
"""


def run(name, text):
    print("=" * 70)
    print(name)
    print("=" * 70)
    result = analyze_essay(text)
    print(f"Essay-level AI-likelihood: {result['essay_level']['ai_likelihood_score']}")
    print(f"Sentence count: {result['sentence_count']}")
    for s in result["sentences"]:
        print(f"  [{s['index']}] sentence_score={s['sentence_level_score']}, "
              f"essay_model_score={round(s['essay_model_sentence_score'], 4) if s['essay_model_sentence_score'] else None}, "
              f"top_features={s['top_features']}")
        print(f"       text: {s['text'][:80]!r}")
    print()


def main():
    run("HUMAN ESSAY (fresh, hand-written for this test)", HUMAN_ESSAY)
    run("AI ESSAY (fresh, hand-written to sound generic-AI)", AI_ESSAY)
    run("MIXED ESSAY (alternating human/AI-style paragraphs)", MIXED_ESSAY)


if __name__ == "__main__":
    main()
