"""Category-specific prompts for the LLM formulator.

Each problem category gets its own BUILD prompt that tells the LLM what to
focus on and what common mistakes to avoid (analogous to OptiMind's
class-specific error hints).
"""

CLASSIFY_SYSTEM_PROMPT = """\
You are an expert in optimization and experimental design. Your task is to
classify a natural language problem description into one of these categories:

- "initial_design": The user wants to design experiments to explore a parameter
  space or fit a surrogate model. They have controllable factors (inputs) and
  measurable responses (outputs). No prior data is available or the user
  explicitly asks for an initial experimental design.

- "single_objective": The user wants to optimize exactly ONE measurable outcome
  (e.g., maximize yield, minimize cost). They have controllable inputs and a
  single objective. They may or may not have prior experimental data.

- "multi_objective": The user has MULTIPLE competing objectives that cannot be
  trivially combined (e.g., maximize strength AND minimize weight). The
  trade-off between objectives matters.

- "space_filling": The user wants uniform coverage of a parameter space without
  a specific optimization target. The goal is exploration or sampling, not
  optimization toward a specific outcome.

- "not_experimental": The problem is a mathematical program (LP, MILP, scheduling,
  assignment, routing, etc.) that does NOT involve physical experiments or
  measurements. It can be solved analytically or with a solver. This is NOT
  suitable for experimental design.

Classify based on the NATURE of the problem, not the user's exact wording.
If uncertain between initial_design and single/multi_objective, prefer the
objective-based category when a clear optimization target is stated.
"""

BUILD_SYSTEM_PROMPT = """\
You are an expert in experimental design and optimization. Your task is to
translate a natural language problem description into a structured domain
specification.

You must identify:
1. INPUT FEATURES: The controllable variables (factors) the experimenter can set.
   - Continuous: has a range [lower_bound, upper_bound]
   - Discrete: has specific allowed numerical values
   - Categorical: has named categories/levels

2. OUTPUT FEATURES: The measurable responses/KPIs to optimize.
   - Each output needs an optimization direction (maximize, minimize, or close_to_target)

3. CONSTRAINTS: Mathematical relationships between inputs that must hold.
   - Linear equality: coefficients * inputs = rhs (e.g., mixture fractions sum to 1)
   - Linear inequality: coefficients * inputs <= rhs (e.g., budget limits)

IMPORTANT RULES:
- Use short, descriptive snake_case names for all features (e.g., "temperature", "pressure", "catalyst_type")
- Every continuous input MUST have both lower_bound and upper_bound
- Every discrete input MUST have a values list
- Every categorical input MUST have a categories list
- Bounds should be physically meaningful (not arbitrary)
- Include units where known
- Constraint feature names must exactly match input names
"""

# Category-specific hints appended to BUILD_SYSTEM_PROMPT
CATEGORY_HINTS: dict[str, str] = {
    "initial_design": """\

CATEGORY: Initial Experimental Design
You are building a domain for designing initial experiments (Design of Experiments).

FOCUS:
- Identify ALL controllable factors and their physical ranges
- Identify what will be MEASURED (outputs) and whether higher or lower is better
- Look for mixture constraints (fractions summing to 1)
- Look for physical constraints (e.g., total volume, budget)

COMMON MISTAKES TO AVOID:
- Forgetting to include all factors mentioned in the problem
- Setting bounds too narrow (missing exploration potential) or too wide (physically impossible)
- Missing mixture/sum constraints when ingredients must sum to a total
- Confusing categorical factors (catalyst type) with continuous ones (catalyst loading)
- Not including outputs — even for DoE, you must specify what you'll measure
""",
    "single_objective": """\

CATEGORY: Single-Objective Optimization
You are building a domain for optimizing exactly ONE output.

FOCUS:
- Identify the SINGLE primary objective clearly
- Determine if it should be maximized or minimized
- If there are secondary metrics, still include them as outputs but note the primary one

COMMON MISTAKES TO AVOID:
- Including multiple outputs without objectives (every output needs an objective direction)
- Confusing constraints with objectives (a budget LIMIT is a constraint, not an output to minimize)
- Missing input features that the user can actually control
- Setting bounds that don't match the user's actual experimental capabilities
""",
    "multi_objective": """\

CATEGORY: Multi-Objective Optimization
You are building a domain with MULTIPLE competing objectives.

FOCUS:
- Each competing objective becomes its own output with its own direction
- Do NOT try to combine objectives into one (no scalarization)
- Constraints are NOT objectives — they are hard limits that must always be satisfied

COMMON MISTAKES TO AVOID:
- Combining objectives into one (e.g., "maximize yield/cost ratio" — keep them separate)
- Forgetting to assign an objective direction to each output
- Confusing constraints (hard limits) with objectives (things to optimize)
- Having only one output (that's single_objective, not multi_objective)
""",
    "space_filling": """\

CATEGORY: Space-Filling Design
You are building a domain for uniform exploration without optimization targets.

FOCUS:
- The primary goal is coverage, not optimization
- Still identify outputs (what will be measured) but the objective direction matters less
- Focus on getting input ranges right for good space coverage

COMMON MISTAKES TO AVOID:
- Not specifying any outputs (you still need to say what you'll measure)
- Setting bounds too tight for exploration
- Overcomplicating with too many constraints
""",
    "not_experimental": """\

CATEGORY: Not Experimental (Mathematical Programming)
This problem is NOT suitable for experimental design in BoFire.

DO NOT attempt to formulate this as a Domain. Instead, provide a minimal
specification that documents the problem structure, even though it won't be
converted to a usable Domain.

The problem is likely: LP, MILP, scheduling, assignment, routing, bin packing,
or similar combinatorial/mathematical optimization that should be solved with
a mathematical programming solver, not experimental design.
""",
}


def get_build_prompt(category: str) -> str:
    """Get the full build system prompt for a given category."""
    hints = CATEGORY_HINTS.get(category, "")
    return BUILD_SYSTEM_PROMPT + hints
