# Lean Environment for Marin RL

This environment provides reinforcement learning training for Lean theorem proving, particularly focused on problems from the Lean Game curriculum (Natural Number Game, etc.).

## Overview

The Lean environment (`lean_env.py`) allows RL agents to:
1. Sample theorem proving problems from a curriculum
2. Generate proof attempts using an inference endpoint
3. Validate proofs using a Lean 4 server
4. Receive rewards based on proof correctness and progress

## Prerequisites

### 1. Install Lean 4

```bash
# On macOS
brew install lean

# On Ubuntu/Debian
wget https://github.com/leanprover/lean4/releases/download/v4.3.0/lean-4.3.0-linux.tar.gz
tar -xzf lean-4.3.0-linux.tar.gz
export PATH=$PATH:$(pwd)/lean-4.3.0-linux/bin

# Verify installation
lean --version
```

### 2. Install Mathlib (Optional, for advanced problems)

```bash
# Create a new Lean project
mkdir lean_workspace && cd lean_workspace
lake init myproject
cd myproject

# Add Mathlib to your project
echo 'require mathlib from git "https://github.com/leanprover-community/mathlib4.git"' >> lakefile.lean
lake update
lake build
```

### 3. Download a Pre-trained Model

For testing, you'll need access to an inference endpoint. Options include:

a) **Local Model (Recommended for testing)**:
```bash
# Example using a small model with vLLM
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-mini-4k-instruct \
    --port 8000
```

b) **OpenAI API**:
```bash
export OPENAI_API_KEY="your-api-key"
```

c) **Custom Marin Inference Server**:
Follow the Marin documentation to set up your inference endpoint.

## Quick Start

### 1. Basic Usage

```python
import asyncio
from marin.rl.envs.lean_env import LeanEnv, LeanEnvConfig, LeanGameCurriculum
from marin.rl.types import InferenceEndpoint

# Create inference endpoint
inference = InferenceEndpoint(
    address="http://localhost:8000",  # Your inference server
    model="gpt-3.5-turbo"  # Or your model
)

# Create rollout sink (callback for completed rollouts)
def rollout_sink(rollouts):
    for rollout_group in rollouts:
        print(f"Received rollout: {rollout_group.id}")
        for rollout in rollout_group.rollouts:
            print(f"  Solved: {rollout.metadata.get('solved', False)}")
            print(f"  Reward: {rollout.metadata.get('best_reward', 0.0)}")

# Create and run environment
async def main():
    env = LeanEnv(
        inference=inference,
        rollout_sink=rollout_sink,
        curriculum_path="lean_problems.json",  # Optional custom problems
        lean_path="lean",  # Path to Lean executable
        workspace_dir="/tmp/lean_workspace",
        max_level=3,  # Start with easier problems
        max_attempts=3,
        seed=42
    )
    
    # Run for a few iterations
    task = asyncio.create_task(env.run())
    await asyncio.sleep(30)  # Run for 30 seconds
    await env.stop()
    await task

# Run the environment
asyncio.run(main())
```

### 2. Using with Ray (Production)

```python
import ray
from marin.rl.envs.lean_env import LeanEnvConfig
from marin.rl.types import InferenceEndpoint

ray.init()

# Configure the environment
config = LeanEnvConfig(
    curriculum_path="lean_problems.json",
    lean_path="lean",
    workspace_dir="/tmp/lean_workspace",
    max_level=5,
    max_attempts=3,
    num_replicas=4  # Run 4 parallel environments
)

# Create inference endpoint
inference = InferenceEndpoint(
    address="http://localhost:8000",
    model="your-model"
)

# Build the environment actor
def rollout_callback(rollouts):
    # Process rollouts (e.g., send to training)
    pass

actor = config.build(
    inference=inference,
    rollout_sink=rollout_callback,
    seed=42
)

# The environment is now running asynchronously
# Stop it when done:
# ray.get(actor.stop.remote())
```

## Testing the Environment

### 1. Unit Test

Create a test file `test_lean_env.py`:

```python
import asyncio
import pytest
from marin.rl.envs.lean_env import LeanServer, LeanProblem, LeanGameCurriculum

@pytest.mark.asyncio
async def test_lean_server():
    """Test Lean server proof checking."""
    server = LeanServer()
    await server.start()
    
    # Test a simple proof
    problem = LeanProblem(
        id="test_refl",
        level=1,
        category="test",
        statement="∀ x : Nat, x = x",
        context=""
    )
    
    # This should succeed
    result = await server.check_proof(problem, "intro x; rfl")
    assert result["valid"] == True
    assert len(result["errors"]) == 0
    
    # This should fail
    result = await server.check_proof(problem, "sorry")
    assert result["valid"] == False
    
    await server.stop()

def test_curriculum():
    """Test problem curriculum loading."""
    curriculum = LeanGameCurriculum("lean_problems.json")
    assert len(curriculum.problems) > 0
    
    # Test getting problems by level
    easy_problem = curriculum.get_problem_by_level(1)
    assert easy_problem.level == 1
    
    # Test getting specific problem
    problem = curriculum.get_problem_by_id("tutorial_world_level_1")
    assert problem is not None
    assert problem.statement == "∀ x : Nat, x = x"

# Run tests
if __name__ == "__main__":
    asyncio.run(test_lean_server())
    test_curriculum()
    print("All tests passed!")
```

### 2. Integration Test

```bash
# Start a dummy inference server (for testing)
python -c "
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/v1/completions', methods=['POST'])
def complete():
    # Return a simple proof attempt
    return jsonify({
        'choices': [{
            'text': 'intro x\\nrfl'
        }]
    })

app.run(port=8000)
" &

# Run the environment test
python -c "
import asyncio
from marin.rl.envs.lean_env import LeanEnv
from marin.rl.types import InferenceEndpoint

async def test():
    inference = InferenceEndpoint('http://localhost:8000', 'test')
    rollouts_received = []
    
    def sink(rollouts):
        rollouts_received.extend(rollouts)
    
    env = LeanEnv(
        inference=inference,
        rollout_sink=sink,
        max_level=1,
        max_attempts=1
    )
    
    task = asyncio.create_task(env.run())
    await asyncio.sleep(5)
    await env.stop()
    
    print(f'Received {len(rollouts_received)} rollouts')
    for r in rollouts_received:
        print(f'  - {r.id}: {r.metadata}')

asyncio.run(test())
"
```

## Customizing the Environment

### Adding New Problems

Edit `lean_problems.json` to add new problems:

```json
{
  "id": "custom_theorem_1",
  "level": 3,
  "category": "custom",
  "statement": "∀ n : Nat, n * 2 = n + n",
  "hint": "Use induction and properties of multiplication",
  "solution": "intro n; induction n; simp; simp [Nat.mul_succ, Nat.add_succ, *]",
  "context": "import Mathlib.Data.Nat.Basic"
}
```

### Adjusting Rewards

Modify the `_calculate_reward` method in `LeanEnv`:

```python
def _calculate_reward(self, result, attempt):
    if result["valid"]:
        # Bonus for quick solutions
        time_bonus = 1.0 - (0.1 * attempt)
        # Bonus for short proofs
        length_bonus = 0.1 if len(proof) < 50 else 0
        return time_bonus + length_bonus
    else:
        # Partial credit for progress
        return 0.5 * (1 - result["goals_remaining"] / 10)
```

### Using Different Lean Versions

Specify the Lean path in the configuration:

```python
config = LeanEnvConfig(
    lean_path="/path/to/lean4/bin/lean",
    # ... other config
)
```

## Troubleshooting

### Common Issues

1. **Lean not found**: 
   - Ensure Lean is installed and in PATH
   - Specify full path in `lean_path` parameter

2. **Mathlib import errors**:
   - Build Mathlib in your workspace: `lake build`
   - Use simpler problems without Mathlib imports

3. **Inference timeout**:
   - Increase timeout in inference configuration
   - Use simpler problems or reduce `max_attempts`

4. **Memory issues with Ray**:
   - Reduce `num_replicas` in configuration
   - Increase Ray object store memory: `ray.init(object_store_memory=2_000_000_000)`

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check Lean server output:
```python
# In lean_env.py, modify check_proof to print output:
print(f"Lean stdout: {result.stdout}")
print(f"Lean stderr: {result.stderr}")
```

## Performance Tips

1. **Batch Processing**: Process multiple problems in parallel using Ray
2. **Caching**: The environment caches Lean file compilations
3. **Problem Selection**: Start with easier problems (lower levels) for faster iteration
4. **Proof Timeout**: Adjust timeout in `LeanServer.check_proof()` based on problem complexity

## References

- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Natural Number Game](https://www.ma.imperial.ac.uk/~buzzard/xena/natural_number_game/)
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Lean Game Server](https://github.com/leanprover-community/lean4game)

## Next Steps

1. Integrate with actual Marin training pipeline
2. Add support for Lean Language Server Protocol for better error messages
3. Implement curriculum learning with automatic difficulty adjustment
4. Add support for multi-file Lean projects
5. Integrate with LeanDojo for more sophisticated proof search