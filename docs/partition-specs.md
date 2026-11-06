## Partitioning an arbitrary computational graph into stages for pipeline-parallel training

We are given a computational DAG, where each node is a computation that:
- Has duration $t$
- Produces output memory $m$
- Saves stored memory $s$
- Has dependencies $[d_1, d_2, .., d_n]$ whose outputs are needed to compute it

Output and stored memory are different ; output is the result of the computation, saved is the activations saved for backward.
The goal is to group these nodes into $n$ groups, where $n$ is a given number of stages, such as:
- Total computation time of each group is balanced
- Volume of memory transfers between groups is minimized
- Saved memory is balanced, *following a formula that depends on the schedule*: e.g., 1F1B with 4 ranks stores 4 activations for rank 0, 3 for rank 1, etc. Ideally we want $4s_0 = 3s_1 = 2s_2 = 1s_3$.

Special cases to keep in mind: produced and saved memory are not purely additive: if two nodes need to save the same tensor for their backward, then the total size of saved memory is only 1x that tensor. However, if they're dispatched onto 2 different groups, it is 1x on *each rank*, meaning memory is duplicated.
Memory transfer should also not count between groups that live on the same rank according to the stage assignment.


Additional technical constraints:
- Some nodes should be forced into a group (input nodes, so that the first stage's inputs are the original model's)
- Some edges should not be allowed to be cut through (i.e. two nodes linked by this edge are forced into the same group). That's to avoid non-tensor communications, that are not handled by ELF for now.