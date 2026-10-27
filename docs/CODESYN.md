# sweeplib@5: code syntax roles

Bonsai-8B is also a model of programming languages, whose syntax is explicit and deterministic to parse: an if
statement's condition, a call's arguments, a list literal, a dictionary's keys. sweeplib@5 asks whether single poles
track those roles. It uses the grammar pass's engine (docs/GRAMMAR.md), with a Pygments lex in place of spaCy.

Code: `bonsai_interp/codesyn/` (`lex.py` lexes the windows and assigns roles, `roles.py` defines conditions and
batteries, `run5.py` runs the stages). Protocol: sweeplib@5 (graph/auto/bonsai1.sweeplib5.*). Data: the
`sweeplib5-data` release.

## The lex

The corpus's code domain is mostly Python: 2,181 of the selection split's 2,427 code windows, and 1,718 of the test
split's 1,895.
- **Python test:** at least 3 lines open with a Python statement keyword, and they outnumber the lines ending in `;`,
  `{` or `}`.
- **Lexing:** each window is lexed whole by Pygments' `PythonLexer`. A bracket stack and a per-statement state machine
  then give every token its roles, its innermost context and the start of its logical line.
- **Mapping:** every model token takes the attributes of the Pygments token that holds its first non-space byte.
- **Visibility:** both snippets of a pair must show the token's whole logical line (its start at most 46 tokens back),
  so the role can be decided from the 48 tokens the model reads.

## The conditions (39)

| role | the last token is… |
|---|---|
| cond | in the condition of an if, elif or while |
| for_target, for_iter | a for loop's variable, or its iterable |
| return, import, decorator | in a return statement, an import, a decorator line |
| def_name, class_name | the name being defined |
| def_sig | in a function definition's parameter list |
| call_arg, kwarg_name | in a call's arguments; a keyword argument's name |
| subscript, list, dict_key, dict_value, comprehension | in a subscript, a list literal, a dict key or value, a comprehension's `for … in` |
| assign_lhs, assign_rhs | left or right of a top-level `=` |
| string, docstring, comment | in a string, a docstring, a comment |
| nested | two or more brackets deep, not one |

- **`role:R`**: the same token in role R against that token elsewhere. For a code role, both tokens are code (neither
  is in a string, docstring or comment).
- **`any:R`** (17 roles): role R whatever the token, against tokens of the same coarse type (name, keyword, number…)
  and corpus-frequency band outside it.

## Results (2026-10-27)

| | |
|---|---|
| Protocol | 0840d9a0 |
| Tests pre-registered | 047b60bc |
| Results | fecbdd7c |
| Selection time | 16 min on the GPU |
| Scoring time | 2.7 min on the GPU |

| condition | tests | held |
|---|---:|---:|
| role:for_target | 77 | 21 |
| role:class_name | 114 | 18 |
| any:for_target | 57 | 10 |
| role:def_name | 56 | 8 |
| any:kwarg_name | 33 | 3 |
| role:def_sig, role:assign_lhs | 30, 10 | 2, 2 |
| role:kwarg_name | 17 | 1 |
| 13 other conditions | 48 | 0 |
| **all** | **442** | **65 (54 poles)** |
| placebo | 442 | **0** |

- **By depth** (held / tests): L0–5 18/114, L6–11 0/21, L12–17 8/51, L18–23 8/64, L24–29 10/53, L30–35 21/139.
  Code-syntax poles appear at every depth except 6–11, and most often late.
- **Coverage:**
  - The share of the pole's 32 strongest activations that fall in code windows and have the role: median 0.16.
  - 22 holds on 20 poles cover at least half and are assignment-grade, for example:
    - L26/N1135− (class name, 94%);
    - L32/N8525− (for-loop variable, 89%);
    - L17/N7128− (class name, 84%);
    - L2/N6207+ (assignment target, 86%);
    - L17/N174− (def name, 74%);
    - L25/N377+ (any keyword-argument name, 71%).
- **What selected nothing or failed:** if-conditions, call arguments, return statements, list literals, dict values,
  comprehensions, strings, docstrings, comments and nesting depth.

## Reading

The roles that hold are **binding sites**: the name a class, def or for loop introduces, and the name of a keyword
argument. They are where a program defines a name that later code will refer to. Roles that only say where a token sits
(inside a call, a list or a condition) produced no holds.

That parallels sweeplib@3's relative-pronoun and possessive results: what the model marks with dedicated poles is
structure that later tokens depend on. Unlike part of speech (docs/POS.md), these poles run through the deep layers
(21 in L30–35), consistent with predicting the next token of a definition.

**FINDINGS paragraph.** sweeplib@5 screened all 884,736 poles for 22 Python syntax roles (39 conditions) on a Pygments
lex of the corpus's code windows. 442 poles were selected and 65 tests held on 54 poles, against 0 of 442 placebo tests;
20 poles are assignment-grade. The held tests are binding sites: a for loop's variable (31), the name in a class (18) or
def (8) statement, and keyword-argument names (4). They run from layer 0 to 35, most often in layers 30–35. Positional roles
(inside a call, list, condition, string or comment, or nesting depth) held nothing.
