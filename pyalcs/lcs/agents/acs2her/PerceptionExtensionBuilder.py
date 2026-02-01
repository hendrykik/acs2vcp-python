from lcs import Perception


def empty_goal_to_state(state: Perception, classifier_wildcard: str) -> Perception:
    # 11111101
    # return Perception(tuple(state) + tuple(['1','1','1','1','1','1','0','1']))
    return Perception(tuple(state) + tuple([classifier_wildcard] * len(state)))


def goal_to_state(state: Perception, goal: Perception) -> Perception:
    return Perception(tuple(state) + tuple(goal))
