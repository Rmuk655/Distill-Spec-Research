from node import *

"""
Creates a tree verifier over a draft tree by taking in:
    - q_paths: 2D list of i.i.d. draft token IDs, rows are paths
    - q_prefix: list of all distinct string representations of path prefixes, e.g. "12,34"
    - q_probs_dict: maps path prefixes to corresponding next-token draft distributions
    - p_probs_dict: maps path prefixes to corresponding next-token target distributions

The tree verifier structure consists of a list of Nodes (see node.py), which represent distinct prefixes on the draft tree.
Any verification method probabilistically outputs a Node in the tree, plus an additional int (extra token).

This class implements a wrapper to call a verification method from a string descriptor.
The currently available verification methods are shown below.
    - {descriptor}  {appearing in}              {setting}       {OT-based}
    - naive:        Speculative Decoding        single-path     YES
    - bv:           Block Verification          single-path     NO
    - nss:          Naive Speculative Sampling  multi-path      YES
    - spectr:       SpecTr                      multi-path      YES
    - specinfer:    SpecInfer                   multi-path      YES
    - khisti:       Canonical Decomposition     multi-path      YES
    - max:          N/A, combines the above     multi-path      YES
    - traversal:    Traversal Verification      multi-path      NO
    - gbv:          Greedy Block Verification   multi-path      NO
NOTE: the bulk of OT-based methods such as spectr are implemented at a token level in the Node class
NOTE: single-path methods, while intended for K=1, can be used for higher K by only taking the first path
NOTE: spectr, specinfer, khisti, and max are equivalent to naive if K=1
NOTE: traversal and gbv are equivalent to bv if K=1
"""
class TreeVerifier:
    def __init__(
        self,
        q_paths: List[List[int]],
        q_prefixes: List[str],
        q_probs_dict: Dict[str, torch.Tensor],
        p_probs_dict: Dict[str, torch.Tensor],
    ):
        self.K = len(q_paths)
        self.L = len(q_paths[0]) - 1

        self.q_paths = q_paths
        self.q_prefixes = q_prefixes
        self.p_probs_dict = p_probs_dict
        self.q_probs_dict = q_probs_dict

        # Initialize Nodes in the tree, without edges.
        self.nodes = []
        for idx, q_prefix in enumerate(self.q_prefixes):
            depth = len(q_prefix.split(",")) - 1        # Depth 0 for root node, which is a single token (pending context)
            token = int(q_prefix.split(",")[-1])        # Last token in the string representation
            node = Node(idx, q_prefix, token, depth)
            self.nodes.append(node)

        # Add edges from a Node (e.g. "12,23,124") to all child Nodes (e.g. "12,23,124,1").
        for q_path in self.q_paths:
            prefix = str(q_path[0])
            for i in range(1, self.L + 1):
                node_idx = self.q_prefixes.index(prefix)
                prefix += "," + str(q_path[i])
                child_node_idx = self.q_prefixes.index(prefix)
                self.add_edge(node_idx, child_node_idx)



    """
    Helper method to build tree structure.
    """
    def add_edge(self, node_idx1: int, node_idx2: int):
        assert node_idx1 < len(self.nodes)
        assert node_idx2 < len(self.nodes)
        self.nodes[node_idx1].children.append(self.nodes[node_idx2])
        self.nodes[node_idx2].parent = self.nodes[node_idx1]



    """
    General wrapper to call a verification method from a string descriptor.
    """
    def verify(self, method: str) -> Tuple[Node, int]:
        if method == "naive":
            return self.naive_verify()
        elif method == "bv":
            return self.bv_verify()
        elif method == "nss":
            return self.nss_verify()
        elif method == "specinfer":
            return self.specinfer_verify()
        elif method == "spectr":
            return self.spectr_verify()
        elif method == "khisti":
            return self.khisti_verify()
        elif method == "max":
            return self.max_verify()
        elif method == "traversal":
            return self.traversal_verify()
        elif method == "gbv":
            return self.gbv_verify()
        else:
            raise Exception(f"{method} is not a supported verifier. Choose from [naive, bv, nss, specinfer, spectr, khisti, max, traversal, gbv].")



    """
    OT-based tree verification methods include NSS, SpecInfer, SpecTr, and standard Speculative Decoding.
    Each requires its own OTLP solver, which maps a triple (draft node, target dist, draft dist) to a single token.
        - The assumption is that the children of the draft node are formed by appending i.i.d. tokens from the draft dist.
        - Under this assumption, the OTLP solver output must follow the target dist (lossless property of decoding).
    The process is then to iteratively progress from the root node of the tree to child nodes, by appending the OTLP solver output.
    When the OTLP solver output takes us off the tree, we terminate.
    Otherwise, if we reach a leaf node of the tree, we sample an additional token from the target distribution.
    The output is a pair (verified draft tree node, additional correction token).
    """
    def otlp_verify(self, otlp_solver: Callable[Tuple[Node, torch.Tensor, torch.Tensor], int]) -> Tuple[Node, int]:
        node = self.nodes[0]

        # Use the OTLP solver (returns a token) to progress to child nodes.
        while node.depth < self.L:
            p = self.p_probs_dict[node.rep]
            q = self.q_probs_dict[node.rep]
            next_token = otlp_solver(node, p, q)
            next_node_rep = node.rep + "," + str(next_token)
            for child_node in node.children:
                if child_node.rep == next_node_rep:
                    node = child_node
                    break

            # Stop when we first land off the draft tree.
            if next_node_rep not in self.q_prefixes:
                return node, next_token

        # If we progressed all the way to a leaf node, sample another token directly from p.
        p = self.p_probs_dict[node.rep]
        return node, torch.multinomial(p, num_samples=1).item()

    def naive_verify(self) -> Tuple[Node, int]:
        return self.otlp_verify(lambda node, p, q : node.naive_otlp_solver(p, q))

    def nss_verify(self) -> Tuple[Node, int]:
        return self.otlp_verify(lambda node, p, q : node.nss_otlp_solver(p, q))

    def specinfer_verify(self) -> Tuple[Node, int]:
        return self.otlp_verify(lambda node, p, q : node.specinfer_otlp_solver(p, q))

    def spectr_verify(self) -> Tuple[Node, int]:
        return self.otlp_verify(lambda node, p, q : node.spectr_otlp_solver(p, q))

    def khisti_verify(self) -> Tuple[Node, int]:
        return self.otlp_verify(lambda node, p, q : node.khisti_otlp_solver(p, q))

    def max_verify(self) -> Tuple[Node, int]:
        return self.otlp_verify(lambda node, p, q : node.max_otlp_solver(p, q))


    """
    Expected returned node.depth for otlp_verify methods.
    Uses cutoffs D = L-trunc, ..., L, where verification terminates once it reaches depth >= D.
    Returns a list with ith entry being E[returned_depth with cutoff D = (L-trunc) + i]
    Uses nss_otlp_branch once per node to cache on-tree transition probabilities to disticnt child tokens.
    """
    def expected_otlp_depths(self, otlp_branch: Callable[[Node, torch.Tensor, torch.Tensor], Dict[int, float]], trunc: int) -> List[float]:
        n = len(self.nodes)
        trunc = min(trunc, self.L)

        # Iterate through nodes in decreasing depth, i.e. ancestors appear later in the order.
        order_desc = sorted(self.nodes, key=lambda u: u.depth, reverse=True)

        # For each node, cache tuples (child_idx, P(append that child's token)) over distinct children, and P(append a token that lands off-tree).
        edges = [[] for _ in range(n)]
        stop = [0.0 for _ in range(n)]

        # Iterate through nodes to update edges and stop lists.
        for node in self.nodes:
            if node.depth >= self.L or node.children == []:
                stop[node.idx] = 1.0
                continue

            # Create mapping of distinct child tokens to indices, overwriting duplicates.
            token_to_child_idx = {}
            for child in node.children:
                token_to_child_idx[child.token] = child.idx

            # Compute branching probability dictionary and update edge and stop lists.
            branch_dict = otlp_branch(node, self.p_probs_dict[node.rep], self.q_probs_dict[node.rep])
            continue_prob = 0.0
            for next_token, next_token_prob in branch_dict.items():
                edges[node.idx].append((token_to_child_idx[next_token], next_token_prob))
                continue_prob += next_token_prob
            stop[node.idx] = max(0.0, 1.0 - continue_prob)

        # For each truncating depth, compute expected OTLP depth on truncated tree from root using DP on edge and stop lists.
        root_idx = self.nodes[0].idx
        out = []
        for end_depth in range(self.L - trunc, self.L + 1):
            expected_depth = [0.0 for i in range(n)]
            for node in order_desc:
                if node.depth >= end_depth:
                    expected_depth[node.idx] = float(end_depth)
                else:
                    expected_depth[node.idx] += stop[node.idx] * node.depth
                    for child_idx, next_token_prob in edges[node.idx]:
                        expected_depth[node.idx] += next_token_prob * expected_depth[child_idx]
            out.append(expected_depth[root_idx])
        return out

    def expected_naive_depths(self, trunc: int) -> List[float]:
        return self.expected_otlp_depths(lambda node, p, q : node.naive_otlp_branch(p, q), trunc)

    def expected_nss_depths(self, trunc: int) -> List[float]:
        return self.expected_otlp_depths(lambda node, p, q : node.nss_otlp_branch(p, q), trunc)

    def expected_specinfer_depths(self, trunc: int) -> List[float]:
        return self.expected_otlp_depths(lambda node, p, q : node.specinfer_otlp_branch(p, q), trunc)

    def expected_spectr_depths(self, trunc: int) -> List[float]:
        return self.expected_otlp_depths(lambda node, p, q : node.spectr_otlp_branch(p, q), trunc)

    def expected_khisti_depths(self, trunc: int) -> List[float]:
        return self.expected_otlp_depths(lambda node, p, q : node.khisti_otlp_branch(p, q), trunc)


    """
    Block verification (https://arxiv.org/pdf/2403.10444) is not an OT-based verification method.
    This is a single-path method, so when the draft tree contains multiple paths, only the FIRST is used.
    First, it recursively computes node weights p_i and block acceptances h_i as in Algorithm 2 of the paper.
    Then, it randomly accepts (prob h_i) or rejects each block, and selects the last accepted block of length tau.
    If the block is the full path (tau = L), an extra token is sampled from the target, otherwise a residual is used.
    """
    def bv_verify(self) -> Tuple[Node, int]:
        node = self.nodes[0]
        node.weight = 1.0
        p = self.p_probs_dict[node.rep]
        q = self.q_probs_dict[node.rep]

        # We will keep track of the last verified node.
        last_ver_node = node

        # Recursively computes node weights and block acceptance probabilities to nodes along the path.
        tau = 0
        for i in range(1, self.L + 1):
            p = self.p_probs_dict[node.rep]
            q = self.q_probs_dict[node.rep]
            weight = node.weight

            # Move to first child node (always stay on the first path), and compute its weight.
            node = node.children[0]
            denom = torch.clamp(q[node.token], min=1e-8)
            node.weight = min(1.0, weight * (p[node.token] / denom).item())

            # Compute block acceptance for the child node.
            h_block = node.weight
            if i < self.L:
                p = self.p_probs_dict[node.rep]
                q = self.q_probs_dict[node.rep]
                h_block = F.relu(node.weight * p - q).sum()
                h_block = h_block / (h_block + 1.0 - node.weight + 1e-10)

            # Set tau to block length if accepted, and verify the block's last node.
            if random.random() <= h_block:
                tau = i
                last_ver_node = node

        # If the whole path was accepted, sample an extra token from the target.
        p = self.p_probs_dict[last_ver_node.rep]
        if tau == self.L:
            return last_ver_node, torch.multinomial(p, num_samples=1).item()

        # Otherwise, sample the extra token from the block residual.
        q = self.q_probs_dict[last_ver_node.rep]
        p_res = F.relu(last_ver_node.weight * p - q)
        if p_res.sum() <= 0:
            p_res += 1e-4
        return last_ver_node, torch.multinomial(p_res, num_samples=1).item()



    """
    Greedy block verification helper to compute the skewed draft distribution at a node with draft distribution q (tensor).
    Assumes that q_joint (scalar), q_joint_cdf (scalar), q_cdf (tensor) fields have been populated.
    We use the following shorthand notation:
        - A = q_joint_cdf - q_joint     scalar      prob of sampling path < current node
        - x = q_joint                   scalar      prob of sampling path == current node
        - y = q_joint * (q_cdf - q)     tensor      prob of sampling path < [current_node, x] over x
        - z = q_joint * q               tensor      dist of sampling path = [current_node, x] over x
    The exact formula for the skew distribution is:
        [(A + y + z)^K - (A + y)^K] / [(A + x)^K - A^K]
    This is not numerically stable, so we factorize this with difference of Kth powers:
        (z / x) * [(A + y + z)^(K-1) + (A + y + z)^(K-2) * (A + y) + ... + (A + y)^(K-1)] / [(A + x)^(K-1) + (A + x)^(K-2) * A + ... + A^(K-1)]
    Simplifying z / x and adjusting scalar constants gives a stable form for reasonably large A:
        q * [((A + y + z) / A)^(K-1) + ((A + y + z) / A)^(K-2) * ((A + y) / A) + ... + ((A + y) / A)^(K-1)]
    This is the formula we use, and we normalize to make it a valid distribution.
    When A is especially small, we replace division by A with division by max(A, 1e-8) for stability.
    NOTE: It is not recommended to use this function (and GBV) with K > 4 due to numerical instability.
    """
    def compute_skew(self, node: Node) -> torch.Tensor:
        q = self.q_probs_dict[node.rep]

        # The skew distribution reduces to q when the multinomial sum has K - 1 = 0 terms.
        if self.K == 1:
            return q

        # Compute relevant quantities for skew computation.
        A = node.q_joint_cdf - node.q_joint
        y = node.q_joint * (node.q_cdf - q)
        z = node.q_joint * q
        A_denom = torch.clamp(A, min=1e-8)
        A_stable = A / A_denom
        y_stable = y / A_denom
        z_stable = z / A_denom

        # Compute the sum of multinomial terms, multiply by q, and renormalize.
        q_skew = torch.zeros_like(q)
        for i in range(self.K):
            q_skew += ((A_stable + y_stable + z_stable) ** i) * ((A_stable + y_stable) ** (self.K - 1 - i))
        q_skew *= q

        # Guard against precision issues and ensure a valid distribution is returned.
        q_skew = F.relu(q_skew)
        s = q_skew.sum()
        if (s <= 0) or (not torch.isfinite(s)):
            return q
        q_skew /= s
        return q_skew



    """
    Greedy block verification (our method) is not an OT-based verification method.
    This is a multi-path method, which reduces to block verification when K=1 (single-path).
    First, it selects the path with highest rank according to lexicographic ordering by p()/q().
    Along the way, computes the skewed draft distribution q_skew at all draft tree nodes.
    Finally, it runs block verification on this single path with q replaced q_skew.
    """
    def gbv_verify(self) -> Tuple[Node, int]:
        q_skew_probs_dict = {}

        # Travel from root to a leaf, by stepping to the child node with highest next-token p / q.
        node = self.nodes[0]
        q0 = self.q_probs_dict[node.rep]
        node.q_cdf = None                       # Full CDF distribution of the next token under p / q ordering.
        node.q_joint = q0.new_tensor(1.0)       # Joint probability value of sampling the current node context.
        node.q_joint_cdf = q0.new_tensor(1.0)   # Joint CDF value of sampling <= to the current node context, under lexicographic p / q ordering.
        for _ in range(self.L):
            p_probs = self.p_probs_dict[node.rep]
            q_probs = self.q_probs_dict[node.rep]
            ratio = p_probs / (q_probs + torch.finfo(p_probs.dtype).eps)

            # Compute the current node's q CDF under the vocab ordering of p / q increasing.
            order = torch.argsort(ratio, stable=True)
            node.q_cdf = torch.empty_like(q_probs)
            node.q_cdf[order] = q_probs[order].cumsum(dim=-1)

            # For non-root nodes, recursively compute the current node's q_joint and q_joint_cdf.
            if node.parent is not None:
                # Multiply parent's joint by current next-token prob to get current joint.
                node.q_joint = node.parent.q_joint * self.q_probs_dict[node.parent.rep][node.token]

                # Compute CDF as total mass across paths which match the parent and deviate at the last token, or are ranked lower than the parent.
                node.q_joint_cdf = node.parent.q_joint_cdf - node.parent.q_joint            # Paths whose start ranks lower than the parent.
                node.q_joint_cdf += node.parent.q_joint * node.parent.q_cdf[node.token]     # Paths that start with the parent and rank lower at next token.

            # Compute skewed draft at this node and update the dictionary.
            q_skew_probs_dict[node.rep] = self.compute_skew(node)

            # Move the current node to the child with highest p / q ranking.
            max_child, max_ratio = None, -1
            for child in node.children:
                if ratio[child.token] > max_ratio:
                    max_child, max_ratio = child, ratio[child.token]
            node = max_child

        # Only select the path we have traveled along.
        chosen_path_idx = min(i for i, q_path in enumerate(self.q_paths) if ",".join(str(x) for x in q_path) == node.rep)
        chosen_path = self.q_paths[chosen_path_idx]

        # Create a new tree verifier instance and run BV.
        bv_tree = TreeVerifier([chosen_path], self.q_prefixes, q_skew_probs_dict, self.p_probs_dict)
        return bv_tree.bv_verify()



    """
    Traversal verification (https://arxiv.org/pdf/2505.12398) is not an OT-based verification method.
    This is a multi-path method, which reduces to block verification when K=1 (single-path).
    First, it recursively computes node weights p_alpha, similar to block verification.
    Then, it iteratively selects the first leaf by DFS ordering, and follows one of two paths:
        (1) accept the leaf and return it, sampling an additional token from p.
        (2) reject the leaf, remove it from the tree, and update weights of descendants.
    The tree size decrements at each iteration, until step (1) is reached (always holds at the root node).
    """
    def traversal_verify(self) -> Tuple[Node, int]:
        # Initialize node weights and list of leaf indices.
        node = self.nodes[0]
        leaves = []
        for node in self.nodes:
            node.weight = 1.0
            if node.parent is not None:
                p_parent = self.p_probs_dict[node.parent.rep]
                q_parent = self.q_probs_dict[node.parent.rep]
                qtok = q_parent[node.token].item()
                node.weight = min(1.0, node.parent.weight * (p_parent[node.token].item() / qtok)) if qtok > 0.0 else 0.0
            if node.children == []:
                leaves.append(node.idx)

        # While there are still leaves remaining, continue.
        while leaves != []:
            # Select the first leaf by DFS ordering.
            first_leaf = self.nodes[min(leaves)]

            # Accept the leaf with probability equal to its weight (or accept it if isolated, i.e. the only node left).
            if first_leaf.parent is None or random.random() <= first_leaf.weight:
                return first_leaf, torch.multinomial(self.p_probs_dict[first_leaf.rep], num_samples=1).item()

            # If we reject the leaf, remove it from the tree, and update connections and leaves.
            leaf_parent = first_leaf.parent
            leaves.remove(first_leaf.idx)
            leaf_parent.children = [child for child in leaf_parent.children if child.rep != first_leaf.rep]
            if leaf_parent.children == []:
                leaves.append(leaf_parent.idx)
            first_leaf.parent = None

            # Update parent weight and target distributions.
            p = self.p_probs_dict[leaf_parent.rep]
            q = self.q_probs_dict[leaf_parent.rep]
            leaf_parent.weight = min(leaf_parent.weight, 1.0)  # guard fp drift above 1
            p_prime = F.relu(leaf_parent.weight * p - q)
            p_prime_sum = p_prime.sum().item()
            if p_prime_sum > 0:
                leaf_parent.weight = p_prime_sum / (p_prime_sum + 1.0 - leaf_parent.weight)
                self.p_probs_dict[leaf_parent.rep] = p_prime / p_prime_sum
            else:
                leaf_parent.weight = 0.0
                self.p_probs_dict[leaf_parent.rep] = p

            # Update parent draft distribution.
            q = self.q_probs_dict[leaf_parent.rep]
            q_rem_sum = q.sum() - q[first_leaf.token]
            if q_rem_sum > 0:
                q_new = q / q_rem_sum
                q_new[first_leaf.token] = 0
                self.q_probs_dict[leaf_parent.rep] = q_new
            else:
                q[first_leaf.token] = 0
                self.q_probs_dict[leaf_parent.rep] = q

            # Update weights for all descendants of the parent.
            for desc in self.nodes:
                if desc.rep.startswith(leaf_parent.rep + ",") and desc.parent is not None:
                    p_parent = self.p_probs_dict[desc.parent.rep]
                    q_parent = self.q_probs_dict[desc.parent.rep]
                    qtok = q_parent[desc.token].item()
                    desc.weight = min(1.0, desc.parent.weight * p_parent[desc.token].item() / qtok) if qtok > 0.0 else 0.0

        return first_leaf, torch.multinomial(self.p_probs_dict[first_leaf.rep], num_samples=1).item()
