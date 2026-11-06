Communication scheduling plays an important part in pipeline parallelism execution.
In ELF, communications are executed asynchronously. A dedicated stream (and communicator) is used for each pair of rank and each direction (forward or backward). This means each communication is independant.
A communication can be started, which means that its NCCL kernel will be enqueued and begin on its dedicated stream. It will finish when the matching kernel (the other side of the communication) is also started and the time needed for the communication passed.
Computations depend on communications, and communications on computation. Forward (resp. Backward Inputs) needs its corresponding RECV Forward (resp. RECV Backward) to be finished before being able to start. SEND Forward (resp. SEND Backward) needs its corresponding Forward (resp. Backward Inputs) to be finished before being able to start.
Additionally, overlapping computations and communications has a cost in performance. Computations are slowed down when SEND of RECV operations are executing at the same time on other streams. This slow down is exponential.

A key parameter to take into account is the number of channels (CTAs) allocated to each communicator. More communicators mean that time needed to transfer data is lower, but the slowdown caused by overlap is higher. Conversely, fewer communicators slow down the transfer but allow the communication to be overlapped with a smaller cost.

The problem of communication scheduling is about when to schedule the start of communications given a fixed order of computations.

There are two major challenges to solve:
1. Avoid deadlocks.
2. Maximize performance, in terms of end-to-end iteration time.
The first one is absolutely needed ; the second one is the objective to optimize.

A deadlock can happen when two ranks are waiting for each other without being able to perform the work that would free the other. The simplest pattern is:
0 | Send to 1 - Receive from 1
1 | Send to 0 - Receive from 0
If communications are blocking, then both ranks are stuck on their first communication.
In our case communications are *not* blocking, but they can be if a cuda synchronization is triggered. In general, any computation can trigger a synchronization. Therefore, if there is a computation between the communications on any of those ranks, it *can* deadlock.

There is a fundamental tradeoff between overlap and latency. Posting a communication in advance means that it may last for longer while waiting for its counterpart, slowing down all computations that happen in the meanwhile. Posting it later risks delaying the communication happening on the other side, and consequently the computations that depend on it.



## The `timed-overlap` scheduler

`safe-overlap` posts every RECV at the earliest slot that provably cannot deadlock, and `overlap`
posts it about one computation before the one that consumes it. Both are answers to the question
"how early?", and neither of them can be right everywhere, because the right answer is not a
constant: it is "at the moment the matching SEND is posted".

That is the whole idea. A transfer cannot start before both of its sides have been posted, so:

- posting a RECV *before* its SEND buys nothing. The transfer still starts when the SEND is posted.
  All the extra time does is keep a communication open, and an open communication slows down every
  computation running next to it.
- posting a RECV *after* its SEND delays the transfer, the computation waiting for it, and
  everything behind that. This is why a rendezvous placement such as `simulation` collapses as the
  pipeline gets wider: it loses 25 % to 47 % at 16 ranks.

So the target is the slot the receiving rank is executing when the SEND goes out. Finding it needs
to know how long operations take, which is why `timed-overlap` reads the `StageProfile` of each
stage. The pipeline measures those once at setup (`elf.profiling.profile_pipeline`, exposed as
`pipeline.profiles` and switched off with `profiling=False`) and hands them to every comm
scheduler as its second argument. Since where the RECVs go changes the very timeline the target is
read off, the two are iterated to a fixed point, starting from the `safe-overlap` placement:

1. time the schedule — a longest path over the computations, exact for a live placement;
2. read off the slot each RECV should aim for;
3. place them there, or as close as the liveness interval of `safe-overlap` allows.

Every round is a placement `safe-overlap` would accept, so `timed-overlap` is deadlock-free by
construction, and it keeps the cheapest round rather than the last, so it never does worse than
the `safe-overlap` placement it starts from. Schedules converge in two to ten rounds. Without
profiles there is no timeline to read, and a guessed one (every computation the same length)
measures *worse* than not trying, so `timed-overlap` then falls back to `safe-overlap`.

The placement itself is not computed at setup: it needs the number of micro-batches, so it happens
on the first `step()`, and again whenever that number changes.

### Measured

Simulated with `local/simulator_with_comms.py` over every greedy schedule of
`~/experiments/comm-scheduling` at 1 and 2 waves (n4/n8/n16, cyclic and mirror, 4 waves left out),
in both CTA regimes, for 198 runs. Results are in `results/{1w,2w}_*_{1ctas,defaultctas}.csv`; the
group files next to them are what reproduces it. Each number is the mean, then the worst, slowdown
against the best scheduler for that exact case, so 1.000 means "always the best". The last column
is how often it *is* the best.

| scheduler       | 1 CTA          | 8 CTAs (default) | best on |
| --------------- | -------------- | ---------------- | ------- |
| `timed-overlap` | 1.0010 / 1.010 | 1.0029 / 1.017   | 136/198 |
| `overlap`       | 1.009 / 1.016  | 1.017 / 1.026    | 0/198   |
| `toporecv`      | 1.070 / 1.222  | 1.028 / 1.056    | 0/198   |
| `safe-overlap`  | 1.037 / 1.082  | 1.033 / 1.050    | 0/198   |
| `simulation`    | 1.139 / 1.397  | 1.031 / 1.116    | 57/198  |
| `topo`          | 1.155 / 1.446  | 1.038 / 1.121    | 5/198   |

`timed-overlap` is the best scheduler on 69 % of the runs and never more than 1.7 % off it,
against 2.6 % for `overlap` — which deadlocks — and 8.2 % for `safe-overlap`. The margin grows with
the pipeline: at 16 ranks it is first on every run, 0.2 % to 1.2 % ahead of `overlap` and 6 % to
38 % ahead of `simulation`.

Where it is not first is 4 ranks, in both regimes, where a strict rendezvous wins by 0.3 % to
0.6 % at one CTA and 0.7 % to 1.3 % at the default: the pipeline has enough bubbles there that
delaying a transfer costs nothing, and `timed-overlap` will not delay a transfer.

It costs a few rounds of the `safe-overlap` placement, which is the dominant term: 0.02 s at
n4 m8, 2 s at n16 m32, about a minute at n16 m128. That is paid on the first `step()`, not spread
over the iterations.

### How many CTAs

The simulator charges overlap as one fixed slowdown per unit of overlapped time, which cannot
represent what `~/experiments/comm-comp-overlap-overhead` actually measured: with the default
number of CTAs the slowdown is roughly linear up to about 8 communications in flight and then runs
away (+50 % at 8, +98 % at 10, and the computation is starved past 16), while with one CTA it stays
around 5 % however many are in flight. Replacing that term with the measured curve, indexed by the
number of communications actually in flight at each instant, says to **keep the default number of
CTAs**:

| case      | 1 CTA | 8 CTAs |
| --------- | ----- | ------ |
| n4 m8     | 1039  | 1032   |
| n8 m64    | 4007  | 3873   |
| n16 m32   | 1129  | 1068   |
| n16 m128  | 4543  | 3907   |

Cutting a communicator to one channel makes every transfer 4.1x longer (1.52 ms against 0.37 ms
here) to buy a flatter overlap penalty, and that trade only pays if the schedule really does keep
many communications in flight at once. Under `timed-overlap` it does not: measured over the same
runs at n16 m128, the mean number of communications in flight per rank is 0.50, against 0.85 under
`overlap` and 1.06 under `safe-overlap`, and the peak never exceeds 3 — nowhere near the region
where the default CTA count misbehaves. Posting just in time is what makes the fast communicator
safe to use, so the two choices go together.

### Open: eating the slack

A RECV whose consumer is not ready anyway could be held back further than the SEND, shortening the
window for free. `timed-overlap` does not do this, and on an earlier and less trustworthy dataset
holding it back measured clearly worse. On the current data it measures mildly *better* — about
0.3 % to 0.6 % at a quarter of the available slack, across almost every case. The difference is
most likely that the earlier schedules had been solved for a different model than the profile they
were timed with, so the timeline the slack was read off was wrong. Worth a proper sweep before
acting on: the failure mode, when it fails, is that eating the slack of one transfer moves the next
onto the critical path and the delay cascades.
