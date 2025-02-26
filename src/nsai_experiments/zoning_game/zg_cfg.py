import logging
from collections import namedtuple
from collections.abc import Iterable

from nltk.tokenize import wordpunct_tokenize
from nltk.parse.recursivedescent import RecursiveDescentParser
from nltk import CFG, PCFG, Nonterminal
from nltk.tree import Tree
import numpy as np
import scipy.ndimage

from .zg_gym import Tile, Location, calc_distance_to_tile, calc_distance_to_location

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_NUM = 6  # TODO do not hardcode
ZONING_GAME_GRAMMAR_STRING = f"""
    Policy -> Rule ";" Policy [0.8] | "" [0.2]
    Rule -> Subject "must" Constraint [1.0]
    Constraint -> "(" Constraint "or" Constraint ")" [0.15]
    Constraint -> "(" Constraint "and" Constraint ")" [0.15]
    Constraint -> "(" "not" Constraint ")" [0.25]
    Constraint -> DistanceConstraint [0.15]
    Constraint -> ClusterCountConstraint [0.15]
    Constraint -> ClusterSizeConstraint [0.15]
    DistanceConstraint -> "be_within" Number "tiles_of" Object [1.0]
    ClusterCountConstraint -> "form_fewer_than" Number "separate_clusters" [1.0]
    ClusterSizeConstraint -> "form_cluster_with_fewer_than" Number "tiles" [1.0]
    Subject -> Tile [1.0]
    Object -> Tile [0.5] | Location [0.5]
    Tile -> {" | ".join([f"\"{x.name}\" [{1/(len(Tile)-1):.4f}]" for x in Tile if x is not Tile.EMPTY])}
    Location -> {" | ".join([f"\"{x.name}\" [{1/(len(Tile)-1):.4f}]" for x in Location])}
    Number -> {" | ".join([f"\"{x}\" [{1/(MAX_NUM-1):.4f}]" for x in range(1, MAX_NUM)])}
    """
ZONING_GAME_GRAMMAR = PCFG.fromstring(ZONING_GAME_GRAMMAR_STRING)

RuleNT = namedtuple("Rule", ["subject", "constraint"])
OrNT = namedtuple("Or", ["sub1", "sub2"])
AndNT = namedtuple("And", ["sub1", "sub2"])
NotNT = namedtuple("Not", ["sub"])
DistanceConstraintNT = namedtuple("DistanceConstraint", ["distance", "object"])
ClusterCountConstraintNT = namedtuple("ClusterCountConstraint", ["count"])
ClusterSizeConstraintNT = namedtuple("ClusterSizeConstraint", ["size"])

def _generate_one_probabilistic_helper(pcfg, current_nonterminal, rng, recursion_limit):
    if recursion_limit < 0:
        return None
    if current_nonterminal is None: current_nonterminal = pcfg.start()
    current_prods = list(pcfg.productions(lhs = current_nonterminal))
    selected_prod = (np.random if rng is None else rng).choice(current_prods, p = [prod.prob() for prod in current_prods])
    result = []
    for fragment in selected_prod.rhs():
        next_token = _generate_one_probabilistic_helper(
            pcfg, fragment, rng=rng, recursion_limit=recursion_limit-1) if isinstance(fragment, Nonterminal) else [fragment]
        if next_token is None: return None
        result += next_token
    return result

def generate_one_probabilistic(pcfg: PCFG, current_nonterminal=None, seed=None, rng=None, max_tokens=250, recursion_limit=25):
    # NOTE max_tokens and recursion_limit exist to try to avoid overly complicated results
    # that the parser runs into a RecursionError processing, but there may be better approaches
    assert seed is None or rng is None
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
        seed = None
    result = _generate_one_probabilistic_helper(pcfg, current_nonterminal, rng, recursion_limit=recursion_limit)
    if (result is not None) and (len(result) <= max_tokens):
        return result
    else:  # If we hit the recursion limit or it's too long, try again
        # NOTE this outer recursive step could get pretty long if we're not careful
        return generate_one_probabilistic(pcfg, current_nonterminal, seed, rng, max_tokens)

def format_ruleset(ruleset):
    return " ".join(ruleset).replace("; ", ";\n")

ZONING_GAME_PARSER = RecursiveDescentParser(ZONING_GAME_GRAMMAR)
def parse_to_ast(formatted_ruleset):
    root, = ZONING_GAME_PARSER.parse(wordpunct_tokenize(formatted_ruleset)+[""])
    return root

def _process_subject_object(subject_object_ast):
    so_type = subject_object_ast.label()
    so_id, = subject_object_ast
    match so_type:
        case "Tile":
            return Tile[so_id]
        case "Location":
            return Location[so_id]
        case _:
            raise ValueError()

def _process_base_constraint(base_constraint_ast):
    match base_constraint_ast.label():
        case "DistanceConstraint":
            _, dist, _, obj = base_constraint_ast
            dist, = dist
            obj, = obj
            return DistanceConstraintNT(int(dist), _process_subject_object(obj))
        case "ClusterCountConstraint":
            _, count, _ = base_constraint_ast
            count, = count
            return ClusterCountConstraintNT(int(count))
        case "ClusterSizeConstraint":
            _, size, _ = base_constraint_ast
            size, = size
            return ClusterSizeConstraintNT(int(size))
        case _:
            raise ValueError()

def _process_constraint(constraint_ast):
    match len(constraint_ast):
        case 1:
            return _process_base_constraint(constraint_ast[0])
        case 4:
            _, not_str, sub, _ = constraint_ast
            assert not_str == "not"
            return NotNT(_process_constraint(sub))
        case 5:
            _, sub1, op_str, sub2, _ = constraint_ast
            match op_str:
                case "and":
                    op = AndNT
                case "or":
                    op = OrNT
                case _:
                    raise ValueError()
            return op(_process_constraint(sub1), _process_constraint(sub2))
        case _:
            raise ValueError()

def _process_rule(rule_ast):
    subject, _, constraint = rule_ast
    subject, = subject
    subject = _process_subject_object(subject)
    constraint = _process_constraint(constraint)
    return RuleNT(subject, constraint)

def parse_to_nt(ruleset):
    """
    A parser of sorts that turns a ruleset into a NamedTuple-based intermediate
    representation that is easier to work with. The input can be an `nltk.tree.Tree` from
    `parse_to_ast` or a string, in which case `parse_to_ast` is called first.
    """
    if not isinstance(ruleset, Tree):
        ruleset = parse_to_ast(ruleset)
    if len(ruleset) == 3:  # Handles `Policy -> Rule ";" Policy`
        rule, _, rest = ruleset
        return [_process_rule(rule)] + parse_to_nt(rest)
    else:  # Handles `Policy -> ""`
        assert list(ruleset) == [""]
        return []

def _calc_shortest_distance(tile_grid, from_row, from_col, to_object):
    match to_object:
        case Tile():
            return calc_distance_to_tile(tile_grid, from_row, from_col, to_object)
        case Location():
            return calc_distance_to_location(tile_grid, from_row, from_col, to_object)
        case _:
            raise ValueError(f"Cannot calculate shortest distance to {to_object}")
    return 0

def _calc_cluster_count(tile_grid, from_row, from_col):
    _, num_features = scipy.ndimage.label(tile_grid == tile_grid[from_row, from_col])
    return num_features

def _calc_cluster_size(tile_grid, from_row, from_col):
    clustered, _ = scipy.ndimage.label(tile_grid == tile_grid[from_row, from_col])
    return np.count_nonzero(clustered == clustered[from_row, from_col])

def _interpret_one_constraint(constraint, tile_grid, my_row, my_col, depth = 0):
    logger.debug(f"  {'  '*depth}Interpreting {constraint} on {(my_row, my_col)}")
    # Fix some stuff for ease of recursion:
    subinterpret = lambda subconstraint: _interpret_one_constraint(subconstraint, tile_grid, my_row, my_col, depth = depth+1)
    # TODO rewrite in object-oriented style
    match constraint:
        case OrNT():
            # Using `|` instead of `or` to avoid short circuiting for now
            return subinterpret(constraint.sub1) | subinterpret(constraint.sub2)
        case AndNT():
            return subinterpret(constraint.sub1) & subinterpret(constraint.sub2)
        case NotNT():
            return not subinterpret(constraint.sub)
        case DistanceConstraintNT():
            shortest_distance = _calc_shortest_distance(tile_grid, my_row, my_col, constraint.object)
            logger.debug(f"  {'  '*depth}  - found shortest distance {shortest_distance} vs. criterion {constraint.distance}")
            return shortest_distance <= constraint.distance
        case ClusterCountConstraintNT():
            cluster_size = _calc_cluster_count(tile_grid, my_row, my_col)
            logger.debug(f"  {'  '*depth}  - found cluster count {cluster_size} vs. criterion {constraint.count}")
            return cluster_size <= constraint.count
        case ClusterSizeConstraintNT():
            cluster_size = _calc_cluster_size(tile_grid, my_row, my_col)
            logger.debug(f"  {'  '*depth}  - found cluster size {cluster_size} vs. criterion {constraint.size}")
            return cluster_size <= constraint.size
        case _:
            raise ValueError(f"Failed to evaluate constraint {constraint}")

def _interpret_indiv(rule_list, tile_grid, my_row, my_col):
    "`interpret_indiv` helper that can only operate on a `NamedTuple` representation"
    logger.info(f"Interpreting on {Tile(tile_grid[my_row, my_col])} at {(my_row, my_col)}")
    my_tile = Tile(tile_grid[my_row, my_col])
    my_rules = list(filter(lambda rule: rule.subject == my_tile, rule_list))
    my_results = [_interpret_one_constraint(rule.constraint, tile_grid, my_row, my_col) for rule in my_rules]
    total_result = all(my_results)
    if total_result: logger.info(f"  - tile complies!")
    else: logger.info(f" - tile violates rules: {[rule for (rule, result) in zip(my_rules, my_results) if not result]}")
    return total_result

def _parse_if_necessary(ruleset):
    if isinstance(ruleset, Iterable) and all([isinstance(r, RuleNT) for r in ruleset]):
        return ruleset
    return parse_to_nt(ruleset)

def interpret_indiv(ruleset, tile_grid, my_row, my_col):
    """
    Check the given tile of the tile grid against the ruleset and return whether it
    complies. If we view sentences from the zoning game language as programs that describe
    which tile configurations are allowed, this is an interpreter of such programs.
    `ruleset` can be the output of `parse_to_nt` or a valid input to that function, in which
    case it is called first.
    """
    rule_list = _parse_if_necessary(ruleset)
    return _interpret_indiv(rule_list, tile_grid, my_row, my_col)

def interpret_grid(ruleset, tile_grid):
    """
    Like `interpret_indiv` but returns a Boolean array of per-tile results across an entire
    tile grid. `ruleset` can be the output of `parse_to_nt` or a valid input to that
    function, in which case it is called first.
    """
    ruleset = _parse_if_necessary(ruleset)
    result = np.zeros(tile_grid.shape, dtype=np.bool)
    for row, col in np.ndindex(tile_grid.shape):
        result[row][col] = _interpret_indiv(ruleset, tile_grid, row, col)
    return result

def interpret_valid_moves(ruleset, tile_grid, next_tile):
    """
    Given a ruleset, tile grid, and next tile, return a Boolean array of where that tile may
    be placed in compliance with the ruleset.
    """
    # TODO test
    
    ruleset = _parse_if_necessary(ruleset)
    tile_grid = tile_grid.copy()
    result = np.zeros(tile_grid.shape, dtype=np.bool)

    for row, col in np.ndindex(tile_grid.shape):
        prev_tile = tile_grid[row, col]
        tile_grid[row, col] = next_tile
        result[row, col] = _interpret_indiv(ruleset, tile_grid, row, col)
        tile_grid[row, col] = prev_tile

    return result
