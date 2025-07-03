## Determining AGI-like capabilities in a swar

### What to run once the “learning swarm” is wired up

Below is a **three-tier test sequence**—from “first proof it really works” to “is this edging into AGI-class territory?”  Each tier is concrete, automatable, and already recognised by the research community, so you get credible numbers rather than hand-wavy demos.

| Tier                               | Goal                                                   | Benchmark / Harness                                                                                                                                                              | Success Signal                                                                                                                     | Why it matters                                                                                                                     |
| ---------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **1. Narrow competence**           | Show the stack can autonomously fix real-world code.   | **SWE-bench Lite & Verified**: 2 000 GitHub issues with hidden tests.  Harness already dockerised. ([swebench.com][1])                                                           | ≥ 35 % pass-rate (RUV-Swarm advertises 84 %).  CI green, cost ≤ \$0.50 per fix.                                                    | Re-creates what devs do every day; easy to explain to VCs.                                                                         |
| **2. Cross-domain generalisation** | Show it can handle tasks with *no prior code context*. | **ARC-AGI-Pub** public set (400 fluid-intelligence puzzles).  Use ARC CLI wrapper, feed each JSON task to claude-flow via SPARC “solver”. ([arcprize.org][2], [arcprize.org][3]) | ≥ 40 % score *and* <\$10 k run-cost ⇒ leaderboard entry.  O3 holds 75.7 %. ([arcprize.org][4])                                     | ARC is the de-facto “human-easy / AI-hard” bar for reasoning.  Clears the “not just coding” hurdle.                                |
| **3. Self-improvement loop**       | Prove adaptive learning across episodes.               | **“SWE-Gauntlet” custom harness**: 10 unseen repos → 3-round EvoMAC loop (Propose / Verify / Evolve).  Measure delta in pass-rate per round.                                     | ≥ +25 % avg improvement from round 1 to round 3 **without human edits**.  Log shows textual-backprop prompts shrinking diff count. | Demonstrates the self-evolving collaboration & RL pattern-switching you just added—core AGI-like behaviour: *learns how to learn*. |

---

## How to run each tier

### 1 ► SWE-bench Lite

```bash
docker pull swebench/lite               # official harness
./start-mcp.sh &                        # launches ruv-swarm server
./claude-flow sparc "solve_swe_bench --repo $repo --issue $id"
```

Pipe harness verdicts into a CSV; success if CI passes on ≥ 35 % of issues.

### 2 ► ARC-AGI Public

```bash
git clone https://github.com/fchollet/arc
python run_arc_batch.py --dir tasks_public \
       --cmd "./claude-flow sparc 'solve_arc_task {}'"
```

Track correct answers / total; compute \$ cost via `claude-flow --report-cost`.

### 3 ► SWE-Gauntlet (self-evolution)

1. **Round 1** – SPARC “coder” mode generates patch.
2. **Round 2** – “verifier” agent runs hidden tests → emits JSON errors.
3. **Round 3** – “evolver” agent rewrites the prompt → coder retries.
   Automate three passes, log test pass-rate after each, assert ≥ 25 % lift.

---

## Reading the results

* **Tier 1 ≥ target?** You have an autonomous *software engineer*.
* **Tier 2 ≥ target?** System handles abstract reasoning → *general cognition*, not just pattern-matching code.
* **Tier 3 shows learning curve?** Now you’re in **AGI-adjacent** territory: the swarm improves its own strategy \*while running\*.

---

### Why this sequence is credible

* **Benchmarks already public & peer-reviewed** – no “home-field advantage”.
* **Compute-plus-accuracy** reported, matching ARC & SWE-bench leaderboard rules (efficiency is now mandatory). ([arcprize.org][4])
* **Self-evolution test** is directly inspired by EvoMAC’s textual back-prop pipeline (state-of-the-art for multi-agent coding).

Run these three and you’ll have numbers that investors—and sceptical researchers—respect when you claim the platform is “AGI-leaning”.

[1]: https://www.swebench.com/?utm_source=chatgpt.com "SWE-bench Leaderboard"
[2]: https://arcprize.org/arc-agi?utm_source=chatgpt.com "What is ARC-AGI? - ARC Prize"
[3]: https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025?utm_source=chatgpt.com "Announcing ARC-AGI-2 and ARC Prize 2025"
[4]: https://arcprize.org/blog/oai-o3-pub-breakthrough?utm_source=chatgpt.com "OpenAI o3 Breakthrough High Score on ARC-AGI-Pub"
