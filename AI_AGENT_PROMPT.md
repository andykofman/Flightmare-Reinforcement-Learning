# AI Agent Work Instructions

## Context
You are helping refactor `flightrl_modern` into `flightrl_v2` following the detailed task list in `REFACTORING_TASKS.md`. This is a migration that maintains 100% backward compatibility while introducing cleaner architecture.

## Your Role
Review and validate completed tasks in the refactoring document, fixing any issues found.

## Workflow

### For Each Task:

1. **Read the task requirements** from `REFACTORING_TASKS.md`
2. **Check the implementation** in `flightrl_v2/` 
3. **Verify**:
   - Files exist at correct locations
   - Imports are correct (change `flightrl_modern` → `flightrl_v2`)
   - No typos or syntax errors
   - Functionality preserved (don't change behavior)
   - Follows task specifications exactly
4. **Fix any issues** found
5. **Mark task as complete** if all verification items pass

### What to Check:

**Imports:**
- Update: `from flightrl_modern.X import Y` → `from flightrl_v2.X import Y`
- Relative imports: `from .module import something`

**Common Issues:**
- Typos in variable names
- Wrong function names (e.g., `make_flightmare_env_for_sb3` should be `make_flight_env_for_sb3`)
- Indentation errors
- Missing files
- Wrong inheritance (check class hierarchies)

**DO NOT:**
- Change functionality or behavior
- Add features not in the task
- Refactor working code
- Add comments unless asked

**DO:**
- Fix typos
- Correct import paths
- Fix syntax errors
- Add clarifying comments when explicitly requested

## Example Review Process

```
Task 3.1: Copy and Adapt train_sac.py
1. Read: Task says copy file, update imports, add log_dir logic
2. Check: flightrl_v2/flightrl_v2/algorithms/train_sac.py exists
3. Verify:
   - ✓ File exists
   - ✗ Import: `make_flightmare_env_for_sb3` (WRONG NAME)
   - ✓ log_dir logic present
   - ✗ Typo: `chceckpoint_callback`
4. Fix:
   - Change to `make_flight_env_for_sb3`
   - Fix typo → `checkpoint_callback`
5. Mark: [x] COMPLETED in REFACTORING_TASKS.md
```

## Response Format

Keep responses SHORT:
- List issues found
- Show fixes applied
- Confirm task completion

Example:
```
✅ Task 3.1 COMPLETED

Fixed:
1. Import: make_flightmare_env_for_sb3 → make_flight_env_for_sb3
2. Typo: chceckpoint_callback → checkpoint_callback

Verified: All checks pass
```

## Critical Rules

1. **Preserve behavior** - Don't change what works
2. **Follow the document** - REFACTORING_TASKS.md is the spec
3. **Answer shortly** - No unnecessary explanations
4. **Fix, don't refactor** - Migration first, improvements later
5. **Verify thoroughly** - Check all verification items

## Starting Point

Begin with: "Review task X.X in REFACTORING_TASKS.md"

The agent will check the task, verify implementation, fix issues, and mark complete.




