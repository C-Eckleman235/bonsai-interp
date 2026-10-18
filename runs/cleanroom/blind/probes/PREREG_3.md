# Pre-registration 3: circuits and induce (written before running neighbors/induce)

Question: is the + pole one representation read out on several word families, or several jobs merged
into one channel? And is the − pole one job?

Neighbors (`neighbors_request.json` in each probe dir):
- P3a (+ pole, merge vs unified): the strongest upstream feeder of the nerve branch (expected L8/N7756, which
  quick.py found at 0.76 drop on the top examples) gives < 0.2 mean drop on the strong branch, and the top-3
  feeders by drop of the strong branch and the nerve branch share at most 1 unit. If this holds, the + pole
  is several jobs merged. If one feeder gives ≥ 0.3 drop on all three branches, the + pole is closer to one job.
- P3b (+ flip, pain context silences "nerves"): group-zeroing the top 10 disambiguators restores ≥ 30% of the gap.
  Among the top 8 restorers there is at least one unit whose corpus peaks are pain/sensation words.
- P3c (− pole): the pain-word feeder L8/N3858 gives ≥ 0.15 drop on pain-mech and < 0.10 on withdrawal. The
  withdrawal branch relies on different feeders, so the − pole is a "somatosensory" job assembled from more
  than one input. It is still one interpretable job.
- P3d (− flip "flinched" vs "smiled"): alone ≥ 0.3, flipped < 0.15 (the reflex cue is the verb).

Induce (exploratory; weak predictions only, because layer 9 of 36 is far from the output):
- P3e: KL from base stays < 0.05 at |c| ≤ 2. Positive steering of the − pole raises p(pain/sensation tokens),
  and positive steering of the + pole raises p(nerve/strong tokens), each by > 1.5× at c = 10.
