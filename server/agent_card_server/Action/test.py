import math
import random
import re
import uuid
import json

# ========================================================
# 1. Utilities for scoring decomposition & improvement
# ========================================================

def decompose_score(total_score):
    """
    Distribute 'total_score' (integer) across 5 sections by fixed percentages:
      1) Intro & Context -> 15%
      2) Summ. Approach  -> 25%
      3) Comprehensiveness -> 30%
      4) Discussion & Future -> 20%
      5) Writing & Referencing -> 10%
    All must be integer and sum to total_score.

    We'll do a largest-remainder approach:
      partial_i = floor(total_score * percentage_i)
      leftover = total_score - sum(partials)
      distribute leftover 1-by-1 based on largest fractional remainder.
    """
    weights = [15, 25, 30, 20, 10]  # sum=100
    N = len(weights)

    if total_score <= 0:
        return [0, 0, 0, 0, 0]

    # 1) initial floor
    partials = []
    remainders = []
    for w in weights:
        exact = total_score * (w / 100.0)
        flo = math.floor(exact)
        partials.append(flo)
        remainders.append(exact - flo)

    assigned = sum(partials)
    leftover = total_score - assigned

    # 2) distribute leftover by largest remainder
    while leftover > 0:
        idx = max(range(N), key=lambda i: remainders[i])
        partials[idx] += 1
        remainders[idx] = 0
        leftover -= 1

    return partials


def sample_improvement(interaction_type, model):
    """
    Return an integer improvement for an interaction,
    with different min..max ranges for "gpt-4o" vs "llama".
    """
    gpt4o_ranges = {
        "path":        (0, 5),    # mean ~1.64
        "chat":        (0, 12),   # mean ~4
        "instruction": (5, 9)     # mean ~7.46
    }
    llama_ranges = {
        "path":        (0, 3),
        "chat":        (0, 7),
        "instruction": (0, 6)
    }

    if model == "gpt-4o":
        low, high = gpt4o_ranges[interaction_type]
    else:
        # assume any other string = "llama"-like model
        low, high = llama_ranges[interaction_type]

    return random.randint(low, high)


def distribute_improvement(improvement, interaction_type):
    """
    Distribute 'improvement' integer across the 5 sections 
    depending on the interaction type. 
    E.g. "path" mostly improves comprehensiveness, etc.

    Return a list of 5 int increments that sum to 'improvement'.
    """
    # We'll define weights for [Intro, Summ, Compr, Disc, Writing]
    # Example patterns:
    if interaction_type == "path":
        # "path" => mostly about comprehensiveness 
        # + some effect on discussion
        weights = [0, 0, 80, 20, 0]
    elif interaction_type == "chat":
        # "chat" => roughly same comprehensiveness, summarization, discussion
        # let's do Summ=30, Compr=40, Disc=30
        weights = [0, 30, 40, 30, 0]
    elif interaction_type == "instruction":
        # "instruction" => more about comprehensiveness & summarization
        # let's do Summ=40, Compr=60
        weights = [0, 40, 60, 0, 0]
    else:
        # fallback => distribute evenly
        weights = [20, 20, 20, 20, 20]

    total_w = sum(weights)
    if total_w == 0:
        # if no weights, fallback to everything in comprehensiveness
        weights = [0, 0, 100, 0, 0]
        total_w = 100

    if improvement <= 0:
        return [0,0,0,0,0]

    partials = []
    remainders = []
    for w in weights:
        exact = improvement * (w / total_w)
        flo = math.floor(exact)
        partials.append(flo)
        remainders.append(exact - flo)

    assigned = sum(partials)
    leftover = improvement - assigned

    while leftover > 0:
        idx = max(range(5), key=lambda i: remainders[i])
        partials[idx] += 1
        remainders[idx] = 0
        leftover -= 1

    return partials


# ============================================================
# 2. Identify parent->child branches and triggers from commits
# ============================================================
def find_branch_parent_child(sim_data):
    """
    For a single simulation output, parse:
      - projectId, uid, user_study_id
      - agents -> each agent has a model, multiple branches
    Build:
      {
        "projectId": ...,
        "uid": ...,
        "user_study_id": ...,
        "model_per_branch": {branch_id: model_str},
        "parents": { child_branch_id: parent_branch_id },
        "trigger": { child_branch_id: interaction_action }, # "chat"|"path"|"instruction"
        "all_branches": [branch_ids...]
      }

    We do this by:
      1. Gather commits in each branch as a set of (index, action).
      2. A is a parent of B if commits(A) is a strict subset of commits(B).
      3. If exactly one parent -> define parents[B] = A.
      4. The "new commits" = commits(B) - commits(A). The earliest of them that is in 
         {"chat","path","instruction"} is the reason for branching -> "trigger[B]".

    If a branch has no parent, it's a root branch (the "very first" branch).
    """
    if len(sim_data.keys()) != 1:
        return None

    study_id = list(sim_data.keys())[0]
    top_obj = sim_data[study_id]
    project_id = top_obj["projectId"]
    uid = top_obj["uid"]
    user_study_id = top_obj["user_study_id"]
    agents = top_obj["agents"]

    # We'll unify across all agents
    model_per_branch = {}
    commits_per_branch = {}

    commit_pattern = re.compile(r"commit-(\d+)\s+\(([^)]+)\)")

    for agent_name, agent_obj in agents.items():
        model = agent_obj["model"]
        branches = agent_obj["branches"]

        for branch_id, branch_data in branches.items():
            cset = set()
            for ck in branch_data["commits"].keys():
                m = commit_pattern.match(ck)
                if m:
                    idx = int(m.group(1))
                    act = m.group(2).strip()
                    cset.add((idx, act))
            commits_per_branch[branch_id] = cset
            model_per_branch[branch_id] = model

    all_branches = list(commits_per_branch.keys())

    parents_map = {}
    trigger_map = {}

    # Find parent->child
    for branch_b in all_branches:
        cset_b = commits_per_branch[branch_b]
        # find possible parents
        possible_parents = []
        for branch_a in all_branches:
            if branch_a == branch_b:
                continue
            cset_a = commits_per_branch[branch_a]
            if cset_a.issubset(cset_b) and (len(cset_a) < len(cset_b)):
                possible_parents.append(branch_a)
        if len(possible_parents) == 1:
            parent_id = possible_parents[0]
            parents_map[branch_b] = parent_id
            new_commits = cset_b - commits_per_branch[parent_id]
            # sort new by index
            sorted_new = sorted(list(new_commits), key=lambda x: x[0])
            # find earliest new commit that is chat/path/instruction
            cause = None
            for (idx, act) in sorted_new:
                if act in ("chat","path","instruction"):
                    cause = act
                    break
            if cause:
                trigger_map[branch_b] = cause

    return {
        "projectId": project_id,
        "uid": uid,
        "user_study_id": user_study_id,
        "model_per_branch": model_per_branch,
        "parents": parents_map,
        "trigger": trigger_map,
        "all_branches": all_branches
    }


# ============================================================
# 3. Score propagation in topological order
# ============================================================

def compute_branch_scores(parent_child_info, project_score_map, final_user_study_scores):
    """
    Given the parent->child relationships + triggers for a single simulation,
    compute 5-section scores for each branch in a topological order.

    We'll store:
       scores_map[branch_id] = [s1, s2, s3, s4, s5]  (the final after-scores for that branch)

    Algorithm:
     - A root branch has no parent => we do the baseline decomposition from project_score_map[projectId].
       That means if baseline=40, we decompose 40 into [6,10,12,8,4], etc.
     - A child branch is triggered by "interaction_type" => we take parent's final scores, 
       compute total_before= sum(parent_scores). 
       Then sample improvement for that interaction. 
       Distribute improvement across the 5 sections. 
       Add them to parent's final scores => child's final scores. 
       Then clamp the sum by final_user_study_scores[user_study_id].
     - If no recognized trigger => assume 0 improvement.

    Return:
      scores_map: { branch_id -> [int,int,int,int,int] (after-scores) }
    """
    project_id = parent_child_info["projectId"]
    user_study_id = parent_child_info["user_study_id"]
    model_per_branch = parent_child_info["model_per_branch"]
    parents_map = parent_child_info["parents"]
    trigger_map = parent_child_info["trigger"]
    all_branches = parent_child_info["all_branches"]

    baseline = project_score_map.get(project_id, 0)
    final_max = final_user_study_scores.get(user_study_id, 100)

    # Build adjacency (parent->list_of_children)
    children_map = {}
    for b in all_branches:
        children_map[b] = []
    for child_b, parent_a in parents_map.items():
        children_map[parent_a].append(child_b)

    # Find roots = those not in parents_map.keys()
    roots = [b for b in all_branches if b not in parents_map]

    scores_map = {}  # branch_id -> final array of 5 ints

    # For root branches => just decompose baseline
    for r in roots:
        scores_map[r] = decompose_score(baseline)

    # BFS or DFS
    queue = list(roots)
    while queue:
        current = queue.pop(0)
        current_scores = scores_map[current]  # parent's final distribution
        parent_total = sum(current_scores)

        # push children
        for child in children_map[current]:
            # child's before-scores = parent's after-scores
            model = model_per_branch[child]
            interaction = trigger_map.get(child, None)
            if interaction is None:
                # no recognized trigger => no improvement
                child_scores = current_scores[:]
            else:
                # sample improvement
                imp = sample_improvement(interaction, model)
                # distribute improvement among the 5 sections
                increments = distribute_improvement(imp, interaction)

                # apply
                child_raw = [p + inc for p, inc in zip(current_scores, increments)]
                # now clamp the sum by final_max
                child_sum = sum(child_raw)
                if child_sum > final_max:
                    diff = child_sum - final_max
                    # reduce 'diff' from the largest section, or distribute in some way
                    idx_max = max(range(5), key=lambda i: child_raw[i])
                    child_raw[idx_max] -= diff
                    child_sum = final_max

                child_scores = child_raw

            scores_map[child] = child_scores
            queue.append(child)

    return scores_map


# ============================================================
# 4. Build final "before-after" records for each child branch
# ============================================================
def build_before_after_records(parent_child_info, scores_map):
    """
    For each child branch that has a parent AND a recognized trigger,
    produce a record:

      {
        "interaction_type": ...,
        "projectID": ...,
        "uid": ...,
        "user_study_id": ...,
        "model": ...,
        "branch-before": parentID,
        "branch-after":  childID,
        "score-before": sum_of_parent_scores,
        "score-after":  sum_of_child_scores,
        "detail-before": parent_scores_list,
        "detail-after":  child_scores_list
      }
    """
    project_id   = parent_child_info["projectId"]
    uid          = parent_child_info["uid"]
    user_study_id = parent_child_info["user_study_id"]
    model_map    = parent_child_info["model_per_branch"]
    parents_map  = parent_child_info["parents"]
    trigger_map  = parent_child_info["trigger"]

    records = []
    for child in parents_map:
        if child in trigger_map:
            interaction = trigger_map[child]
            parent = parents_map[child]

            parent_scores = scores_map[parent]
            child_scores  = scores_map[child]

            record = {
                "interaction_type": interaction,
                "projectID": project_id,
                "uid": uid,
                "user_study_id": user_study_id,
                "model": model_map[child],
                "branch-before": parent,
                "branch-after":  child,
                "score-before": sum(parent_scores),
                "score-after":  sum(child_scores),
                "detail-before": parent_scores,
                "detail-after":  child_scores
            }
            records.append(record)
    return records


# ============================================================
# 5. Put everything together
# ============================================================
def create_detailed_before_after(
    all_simulations,
    project_score_map,
    final_user_study_scores
):
    """
    all_simulations: list of data items (each from your generator).
    project_score_map: { projectId -> baseline (0..100) }
    final_user_study_scores: { user_study_id -> final max score (0..100) }

    For each simulation:
      - parse parent->child structure
      - compute 5-section scores in topological order 
        (root branch uses baseline, child branches add improvement)
      - build a list of "before-after" records for each child that was triggered by an interaction

    Return a single combined list of all such records from all simulations.
    """
    all_records = []
    for sim_data in all_simulations:
        pc_info = find_branch_parent_child(sim_data)
        if pc_info is None:
            continue
        
        scores_map = compute_branch_scores(pc_info, project_score_map, final_user_study_scores)
        recs = build_before_after_records(pc_info, scores_map)
        all_records.extend(recs)

    return all_records


# ============================================================
# DEMO
# ============================================================
# if __name__ == "__main__":
    # Example "project -> baseline" scores:
    # project_score_map = {
    #     "proj-1": 25,
    #     "proj-2": 40,
    #     "proj-3": 70,
    # }
    # # Example "user_study -> final" max scores:
    # final_user_study_scores = {
    #     "study-1": 60,
    #     "study-2": 95,
    #     "study-3": 100
    # }

    # # We'll build a small "fake" simulation data 
    # # with multiple branches that illustrate how new branches 
    # # are triggered by interactions:
    
    # # Root branch has (commit-0 (retrieve))
    # # Child branch from root triggered by (commit-1 (path)), (commit-2 (reflect))
    # # Another child from that triggered by (commit-3 (chat)), (commit-4 (reflect))
    # # Another from that triggered by (commit-5 (instruction))
    # # 
    # # All commits from the parent are inherited by the child.

    # root_id = str(uuid.uuid4())
    # child1_id = str(uuid.uuid4())
    # child2_id = str(uuid.uuid4())
    # child3_id = str(uuid.uuid4())

    # sim_obj_1 = {
    #     str(uuid.uuid4()): {
    #         "projectId": "proj-2",
    #         "uid": "user-1",
    #         "user_study_id": "study-1",
    #         "agents": {
    #             "card-agent-0": {
    #                 "model": "gpt-4o",
    #                 "branches": {
    #                     root_id: {
    #                         "commits": {
    #                             "commit-0 (retrieve)": {
    #                                 "blob": {"id":"..."}
    #                             }
    #                         }
    #                     },
    #                     child1_id: {
    #                         "commits": {
    #                             "commit-0 (retrieve)": {"blob": {"id":"..."}},
    #                             "commit-1 (path)":     {"blob": {"id":"..."}},
    #                             "commit-2 (reflect)":  {"blob": {"id":"..."}}
    #                         }
    #                     },
    #                     child2_id: {
    #                         "commits": {
    #                             "commit-0 (retrieve)": {"blob": {"id":"..."}},
    #                             "commit-1 (path)":     {"blob": {"id":"..."}},
    #                             "commit-2 (reflect)":  {"blob": {"id":"..."}},
    #                             "commit-3 (chat)":     {"blob": {"id":"..."}},
    #                             "commit-4 (reflect)":  {"blob": {"id":"..."}}
    #                         }
    #                     },
    #                     child3_id: {
    #                         "commits": {
    #                             "commit-0 (retrieve)": {"blob": {"id":"..."}},
    #                             "commit-1 (path)":     {"blob": {"id":"..."}},
    #                             "commit-2 (reflect)":  {"blob": {"id":"..."}},
    #                             "commit-3 (chat)":     {"blob": {"id":"..."}},
    #                             "commit-4 (reflect)":  {"blob": {"id":"..."}},
    #                             "commit-5 (instruction)": {"blob": {"id":"..."}}
    #                         }
    #                     }
    #                 }
    #             }
    #         }
    #     }
    # }

    # # We can feed multiple sims. Let's just do one for demonstration:
    # all_sims = [sim_obj_1]

    final_records = create_detailed_before_after(
        all_simulations=all_sims,
        project_score_map=project_score_map,
        final_user_study_scores=final_user_study_scores
    )

    # # Print the final results
    # print(json.dumps(final_records, indent=2))
