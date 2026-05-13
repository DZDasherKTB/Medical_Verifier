
# 🧠 How the LLM Actually Thinks: Triplet Evaluation Insights

Hey team! I just finished diving into the Triplet Method evaluation data, and the results gave us a really fascinating peek into the AI's "psychology."

First, the raw numbers are solid. But the real story is in *how* the model fails when it gets confused.

## 📊 The Bottom Line Numbers

* **Total Triplet Comparisons:** `2,955` evaluated with an impressive **`95.97%`** accuracy.
* **Question-Level Consistency:** Out of `985` total questions, the model got `917` completely correct (**`93.10%`**).

> *Note: A question only counts as "fully correct" if the model didn't stumble on any of the 3 triplet comparisons for that specific question.*



## 🚩 The Big Takeaway: The AI is a People-Pleaser, Not an Expert

The biggest flaw we found with this Triplet method is how easily the model gets swayed by what's right in front of it.

If I had to summarize the AI's behavior in one sentence, it would be:

> *"I'm comparing these options, but if you put a really tricky wrong answer in front of me, I'll completely forget my original correct answer—and then write a long essay to defend my bad choice."*

Here is exactly how that plays out in the data:

### 1. The "Company It Keeps" Problem

We found **59 questions (about 6%)** where the model got 1 or 2 of the triplets right, but completely failed the others for the exact same question.

Why is this a big deal? Because if the AI truly *knew* the answer was "C", it should pick "C" every single time, no matter what distractors we put next to it (`C+A+B`, `C+A+D`, etc.). But it doesn't.

If we introduce a particularly clever "trick" option into the pair, the model gets distracted, forgets the actual right answer, and picks the trick option. **Takeaway:** The AI doesn't have "absolute knowledge." Its decisions are highly relative and depend entirely on how confusing the current options are.

### 2. Falling for the "A vs. B" Trap

When the model *did* get confused in those edge cases, it made the vast majority of its mistakes specifically when comparing **Option A vs. Option B**.

This makes total sense when you think about it. Test-makers usually put the most deceptively similar, tricky statements in options A and B. Because the Triplet method forces the AI to look at options in isolated pairs, it gets severe **tunnel vision**. It can't see the whole question at once to gauge the context, so it falls right into the semantic trap set between A and B.

### 3. It Rambles When It's Lying

This is a classic LLM behavioral quirk, and the data proved it mathematically!

When the model confidently picks the right answer, it keeps its explanation short and sweet. But when it gets confused and picks a *wrong* distractor, it suddenly starts generating significantly longer text justifications. Basically, when the AI is wrong, it hallucinates longer explanations to try and convince itself (and us) that its bad guess was actually brilliant.