"""Lean theorem proving environment for Marin RL.

This environment provides an RL interface for training agents to solve
Lean theorem proving problems, particularly those from the Lean Game
curriculum. The agent interacts with a Lean server to check proofs and
receive feedback.

The environment:
1. Samples Lean problems from a curriculum (e.g., Natural Number Game)
2. Sends problems to the inference endpoint to get proof attempts
3. Validates proofs using a Lean server
4. Returns rewards based on proof correctness and progress
"""

import asyncio
import json
import logging
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
from levanter.utils.ray_utils import RayResources
from ray.actor import ActorHandle

from ..config import AbstractEnvConfig
from ..env import AbstractMarinEnv
from ..types import (
    InferenceEndpoint,
    Rollout,
    RolloutGroup,
    RolloutSink,
    Turn,
)

logger = logging.getLogger(__name__)


@dataclass
class LeanProblem:
    """A single Lean theorem proving problem."""
    
    id: str
    level: int  # Difficulty level (1-10)
    category: str  # e.g., "natural_numbers", "logic", "sets"
    statement: str  # The theorem statement to prove
    hint: Optional[str] = None  # Optional hint for the problem
    solution: Optional[str] = None  # Known solution (for validation)
    context: Optional[str] = None  # Lean imports and context


class LeanServer:
    """Interface for interacting with a Lean 4 server."""
    
    def __init__(self, lean_path: str = "lean", workspace_dir: Optional[str] = None):
        """Initialize the Lean server interface.
        
        Args:
            lean_path: Path to the Lean executable
            workspace_dir: Directory containing Lean project files
        """
        self.lean_path = lean_path
        self.workspace_dir = workspace_dir or "/tmp/lean_workspace"
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        self.process = None
        
    async def start(self):
        """Start the Lean server process."""
        # For now, we'll use a simple subprocess approach
        # In production, this should use the Lean Language Server Protocol
        logger.info("Starting Lean server at %s", self.workspace_dir)
        
    async def check_proof(self, problem: LeanProblem, proof_attempt: str) -> Dict[str, Any]:
        """Check if a proof attempt is valid.
        
        Args:
            problem: The problem being solved
            proof_attempt: The proof text to validate
            
        Returns:
            Dictionary with validation results including:
            - valid: bool - whether the proof is correct
            - errors: List[str] - any error messages
            - warnings: List[str] - any warning messages
            - goals_remaining: int - number of unsolved goals
        """
        # Create a temporary Lean file with the proof attempt
        proof_file = Path(self.workspace_dir) / f"proof_{problem.id}.lean"
        
        lean_content = []
        if problem.context:
            lean_content.append(problem.context)
        lean_content.append(f"theorem {problem.id} : {problem.statement} := by")
        lean_content.append("  " + proof_attempt.replace("\n", "\n  "))
        
        proof_file.write_text("\n".join(lean_content))
        
        # Run Lean to check the proof
        try:
            result = subprocess.run(
                [self.lean_path, proof_file],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.workspace_dir
            )
            
            # Parse the output
            errors = []
            warnings = []
            goals_remaining = 0
            
            if result.returncode == 0:
                # Proof is valid
                return {
                    "valid": True,
                    "errors": [],
                    "warnings": warnings,
                    "goals_remaining": 0
                }
            else:
                # Parse error messages
                for line in result.stderr.split("\n"):
                    if "error:" in line.lower():
                        errors.append(line)
                    elif "warning:" in line.lower():
                        warnings.append(line)
                    elif "unsolved goals" in line.lower():
                        # Extract number of unsolved goals
                        try:
                            goals_remaining = int(''.join(filter(str.isdigit, line)) or 0)
                        except:
                            goals_remaining = 1
                            
                return {
                    "valid": False,
                    "errors": errors,
                    "warnings": warnings,
                    "goals_remaining": goals_remaining
                }
                
        except subprocess.TimeoutExpired:
            return {
                "valid": False,
                "errors": ["Proof checking timed out"],
                "warnings": [],
                "goals_remaining": -1
            }
        except Exception as e:
            logger.error("Error checking proof: %s", e)
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "goals_remaining": -1
            }
            
    async def stop(self):
        """Stop the Lean server process."""
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()


class LeanGameCurriculum:
    """Manages the curriculum of Lean problems from the Lean Game."""
    
    def __init__(self, curriculum_path: Optional[str] = None):
        """Initialize the curriculum.
        
        Args:
            curriculum_path: Path to JSON file containing problems
        """
        self.problems = self._load_default_problems()
        if curriculum_path:
            self._load_custom_problems(curriculum_path)
            
    def _load_default_problems(self) -> List[LeanProblem]:
        """Load default starter problems."""
        # These are simple problems to get started
        # In production, load from the actual Lean Game curriculum
        return [
            LeanProblem(
                id="nat_add_zero",
                level=1,
                category="natural_numbers",
                statement="∀ n : Nat, n + 0 = n",
                hint="Use induction on n",
                solution="induction n <;> simp [*, Nat.add_succ]",
                context="import Mathlib.Data.Nat.Basic"
            ),
            LeanProblem(
                id="nat_zero_add",
                level=1,
                category="natural_numbers", 
                statement="∀ n : Nat, 0 + n = n",
                hint="Use the definition of addition",
                solution="intro n; simp",
                context="import Mathlib.Data.Nat.Basic"
            ),
            LeanProblem(
                id="nat_add_comm",
                level=2,
                category="natural_numbers",
                statement="∀ m n : Nat, m + n = n + m",
                hint="Use induction on one of the variables",
                solution="intros m n; induction m <;> simp [*, Nat.add_succ, Nat.succ_add]",
                context="import Mathlib.Data.Nat.Basic"
            ),
            LeanProblem(
                id="imp_self",
                level=1,
                category="logic",
                statement="∀ P : Prop, P → P",
                hint="This is the identity function",
                solution="intro P hp; exact hp",
                context=""
            ),
            LeanProblem(
                id="and_comm",
                level=2,
                category="logic",
                statement="∀ P Q : Prop, P ∧ Q → Q ∧ P",
                hint="Destructure the conjunction and rebuild it",
                solution="intros P Q hpq; exact ⟨hpq.2, hpq.1⟩",
                context=""
            ),
        ]
        
    def _load_custom_problems(self, path: str):
        """Load problems from a JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.problems.append(LeanProblem(**item))
        except Exception as e:
            logger.warning("Could not load custom problems from %s: %s", path, e)
            
    def get_problem_by_level(self, max_level: int) -> LeanProblem:
        """Get a random problem up to the specified level."""
        eligible = [p for p in self.problems if p.level <= max_level]
        if not eligible:
            eligible = self.problems
        return random.choice(eligible)
        
    def get_problem_by_id(self, problem_id: str) -> Optional[LeanProblem]:
        """Get a specific problem by ID."""
        for p in self.problems:
            if p.id == problem_id:
                return p
        return None


class LeanEnv(AbstractMarinEnv):
    """Lean theorem proving environment for training RL agents.
    
    This environment samples Lean problems, sends them to an inference
    endpoint for proof attempts, validates the proofs using a Lean server,
    and returns rewards based on correctness and proof quality.
    """
    
    def __init__(
        self,
        inference: InferenceEndpoint,
        rollout_sink: RolloutSink,
        *,
        curriculum_path: Optional[str] = None,
        lean_path: str = "lean",
        workspace_dir: Optional[str] = None,
        max_level: int = 10,
        max_attempts: int = 3,
        seed: int = 0,
    ):
        """Initialize the Lean environment.
        
        Args:
            inference: Inference endpoint for generating proofs
            rollout_sink: Callback for completed rollouts
            curriculum_path: Path to problem curriculum JSON
            lean_path: Path to Lean executable
            workspace_dir: Directory for Lean workspace
            max_level: Maximum problem difficulty level
            max_attempts: Maximum proof attempts per problem
            seed: Random seed for reproducibility
        """
        super().__init__(inference, rollout_sink)
        
        self.curriculum = LeanGameCurriculum(curriculum_path)
        self.lean_server = LeanServer(lean_path, workspace_dir)
        self.max_level = max_level
        self.max_attempts = max_attempts
        self.rng = random.Random(seed)
        
        self.current_level = 1  # Start with easy problems
        self.solved_count = 0
        self.total_attempts = 0
        
        logger.info("LeanEnv initialized with %d problems", len(self.curriculum.problems))
        
    async def run(self) -> None:
        """Main environment loop."""
        await self.lean_server.start()
        
        try:
            while not await self._should_stop():
                # Sample a problem based on current level
                problem = self.curriculum.get_problem_by_level(self.current_level)
                
                # Generate and evaluate proof attempts
                rollout_group = await self._solve_problem(problem)
                
                # Send the rollout group to the sink
                if rollout_group:
                    self._rollout_sink([rollout_group])
                    
                # Adjust difficulty based on performance
                self._update_difficulty()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
        finally:
            await self.shutdown()
            
    async def _solve_problem(self, problem: LeanProblem) -> Optional[RolloutGroup]:
        """Attempt to solve a single problem.
        
        Args:
            problem: The Lean problem to solve
            
        Returns:
            RolloutGroup containing the interaction
        """
        turns = []
        
        # Create the initial prompt
        prompt = self._format_problem_prompt(problem)
        
        # User turn with the problem statement
        turns.append(Turn(
            message=prompt,
            role="user",
            logprobs=None,
            reward=0.0,
            inference_metadata={}
        ))
        
        # Try to solve the problem with multiple attempts
        solved = False
        best_reward = 0.0
        
        for attempt in range(self.max_attempts):
            # Get proof attempt from inference endpoint
            try:
                # Make inference call
                response = await self._call_inference(prompt)
                proof_text = self._extract_proof_from_response(response)
                
                # Check the proof
                result = await self.lean_server.check_proof(problem, proof_text)
                
                # Calculate reward
                reward = self._calculate_reward(result, attempt)
                best_reward = max(best_reward, reward)
                
                # Add assistant turn
                turns.append(Turn(
                    message=response,
                    role="assistant",
                    logprobs=None,
                    reward=reward,
                    inference_metadata={
                        "attempt": attempt + 1,
                        "valid": result["valid"],
                        "goals_remaining": result.get("goals_remaining", 0)
                    }
                ))
                
                if result["valid"]:
                    solved = True
                    self.solved_count += 1
                    logger.info("Solved problem %s in %d attempts", problem.id, attempt + 1)
                    break
                    
                # Add feedback for next attempt if not solved
                if attempt < self.max_attempts - 1:
                    feedback = self._format_feedback(result)
                    turns.append(Turn(
                        message=feedback,
                        role="user",
                        logprobs=None,
                        reward=0.0,
                        inference_metadata={}
                    ))
                    prompt = feedback  # Update prompt for next attempt
                    
            except Exception as e:
                logger.error("Error during inference for problem %s: %s", problem.id, e)
                break
                
        self.total_attempts += 1
        
        # Create rollout
        rollout = Rollout(
            turns=turns,
            metadata={
                "problem_id": problem.id,
                "level": problem.level,
                "category": problem.category,
                "solved": solved,
                "best_reward": best_reward
            }
        )
        
        # Create rollout group
        return RolloutGroup(
            id=f"lean-{problem.id}-{self.total_attempts}",
            source="lean_env",
            created=time.time(),
            rollouts=[rollout],
            metadata={
                "current_level": self.current_level,
                "solve_rate": self.solved_count / max(1, self.total_attempts)
            }
        )
        
    def _format_problem_prompt(self, problem: LeanProblem) -> str:
        """Format a problem into a prompt for the model."""
        parts = [
            "You are a Lean 4 theorem prover. Prove the following theorem:",
            f"Theorem: {problem.statement}",
        ]
        
        if problem.hint:
            parts.append(f"Hint: {problem.hint}")
            
        parts.extend([
            "",
            "Provide your proof using Lean 4 tactics. Your proof should be valid Lean code.",
            "Start your proof with the tactics directly (don't include the theorem declaration).",
            ""
        ])
        
        return "\n".join(parts)
        
    def _extract_proof_from_response(self, response: str) -> str:
        """Extract the proof text from model response."""
        # Look for code blocks
        if "```lean" in response:
            start = response.find("```lean") + 7
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
                
        # Otherwise assume the whole response is the proof
        return response.strip()
        
    def _calculate_reward(self, result: Dict[str, Any], attempt: int) -> float:
        """Calculate reward based on proof validation result."""
        if result["valid"]:
            # Full reward for correct proof, with bonus for fewer attempts
            return 1.0 - (0.1 * attempt)
        else:
            # Partial reward based on progress
            goals = result.get("goals_remaining", -1)
            if goals == 0:
                return 0.9  # Almost solved
            elif goals > 0:
                # Partial credit based on goals remaining
                return max(0.1, 0.5 - (0.1 * goals))
            else:
                # Syntax error or timeout
                return 0.0
                
    def _format_feedback(self, result: Dict[str, Any]) -> str:
        """Format validation feedback for the next attempt."""
        parts = ["Your proof attempt was not successful."]
        
        if result.get("errors"):
            parts.append("Errors:")
            for error in result["errors"][:3]:  # Limit to 3 errors
                parts.append(f"- {error}")
                
        goals = result.get("goals_remaining", 0)
        if goals > 0:
            parts.append(f"You have {goals} unsolved goal(s) remaining.")
            
        parts.append("Please try again with a different approach.")
        
        return "\n".join(parts)
        
    def _update_difficulty(self):
        """Adjust problem difficulty based on performance."""
        if self.total_attempts >= 10:
            solve_rate = self.solved_count / self.total_attempts
            
            if solve_rate > 0.8 and self.current_level < self.max_level:
                # Doing well, increase difficulty
                self.current_level = min(self.current_level + 1, self.max_level)
                logger.info("Increasing difficulty to level %d", self.current_level)
            elif solve_rate < 0.3 and self.current_level > 1:
                # Struggling, decrease difficulty
                self.current_level = max(self.current_level - 1, 1)
                logger.info("Decreasing difficulty to level %d", self.current_level)
                
    async def _call_inference(self, prompt: str) -> str:
        """Call the inference endpoint to get a proof attempt."""
        # This is a placeholder - implement actual inference call
        # based on your InferenceEndpoint interface
        
        # For now, return a dummy response
        await asyncio.sleep(0.1)  # Simulate inference time
        return "sorry"  # Lean's way of admitting defeat
        
    async def shutdown(self) -> None:
        """Clean up resources."""
        await self.lean_server.stop()
        logger.info("LeanEnv shutdown complete. Solved %d/%d problems", 
                   self.solved_count, self.total_attempts)


class LeanEnvConfig(AbstractEnvConfig):
    """Configuration for the Lean environment."""
    
    curriculum_path: Optional[str] = None
    lean_path: str = "lean"
    workspace_dir: Optional[str] = None
    max_level: int = 10
    max_attempts: int = 3
    num_replicas: int = 1
    
    def resources(self) -> RayResources:
        """Return Ray resource requirements."""
        return RayResources(cpu=1)  # Lean checking is CPU-bound
        
    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink, seed: int) -> ActorHandle:
        """Build the Lean environment actor."""
        ActorCls = ray.remote(num_cpus=1)(LeanEnv)
        actor = ActorCls.remote(
            inference,
            rollout_sink,
            curriculum_path=self.curriculum_path,
            lean_path=self.lean_path,
            workspace_dir=self.workspace_dir,
            max_level=self.max_level,
            max_attempts=self.max_attempts,
            seed=seed
        )
        actor.run.remote()  # Start the environment loop
        return actor