The goal of rematerialization is to delete some tensors to free memory and recompute them later.
The two criteria for a good implementation are:
- No tensor is kept in memory if it is not absolutely needed for later
- No computation is performed if not strictly needed.

A tensor and its gradients are too different things ; sometimes the gradient is needed but not the tensor data. In that case the data should be deleted.

There are two kinds of recomputation possible:
- forward recomputation is when activations are deleted after the forward pass F, and recomputed during the recomputation RF, which is before the backward with respect to inputs B.
- backward recomputation is when tensors are deleted after B, and recomputed before the backward with respect to parameters W. In that case there are 2 possibilies:
    - activations are deleted. In that case only forward computation is needed before W.
    - activations & gradients are deleted. In that case both a forward and a backward are needed to recompute everything.


For forward recomputation, the decision to keep or delete tensors is driven by the RematStrategy option. It tells us if a module is recomputed or not. If a module is recomputed, all of its intermediate activations, including its outputs, are deleted. Its input however is saved for the recomputation, UNLESS this input is also recomputed. Note that the module can be any submodule in the tree, including the root module itself.
For backward recomputation, only full recomputation (the root module) is needed for now. Only the input to the root module should be kept. For the gradients, if they are recomputed then only the output gradient should be kept.