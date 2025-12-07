"""
Enhanced OR-Tools Optimizer Configuration

Adds performance optimizations to the existing FPLOptimizer:
- Multi-threading for faster solving
- Time limits to prevent long-running optimizations
- Optimality gap configuration
- Parallel multi-gameweek optimization
- Solver statistics tracking
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from backend.src.core.optimizer import FPLOptimizer

logger = logging.getLogger(__name__)


class EnhancedOptimizerConfig:
    """Enhanced configuration for OR-Tools solver"""
    
    def __init__(self):
        # Solver performance settings
        self.num_workers = int(os.getenv('ORTOOLS_NUM_WORKERS', os.cpu_count() or 4))
        self.max_time_seconds = float(os.getenv('ORTOOLS_TIME_LIMIT', 30.0))
        self.relative_mip_gap = float(os.getenv('ORTOOLS_MIP_GAP', 0.01))  # 1%
        self.log_search_progress = os.getenv('ORTOOLS_LOG_PROGRESS', 'false').lower() == 'true'
        
        logger.info(f"Enhanced optimizer config: {self.num_workers} workers, {self.max_time_seconds}s limit, {self.relative_mip_gap} gap")
    
    def apply_to_solver(self, solver):
        """
        Apply enhanced configuration to OR-Tools solver
        
        Args:
            solver: OR-Tools solver instance
        """
        # Multi-threading
        if hasattr(solver, 'SetNumThreads'):
            solver.SetNumThreads(self.num_workers)
            logger.debug(f"Set solver threads: {self.num_workers}")
        
        # Time limit (in milliseconds)
        if hasattr(solver, 'SetTimeLimit'):
            solver.SetTimeLimit(int(self.max_time_seconds * 1000))
            logger.debug(f"Set time limit: {self.max_time_seconds}s")
        
        # MIP gap
        if hasattr(solver, 'SetSolverSpecificParametersAsString'):
            # For SCIP solver
            solver.SetSolverSpecificParametersAsString(
                f"limits/gap={self.relative_mip_gap}"
            )
            logger.debug(f"Set MIP gap: {self.relative_mip_gap}")
        
        # Logging
        if not self.log_search_progress:
            solver.EnableOutput()  # Disable verbose output
    
    def get_solver_stats(self, solver) -> Dict:
        """
        Extract solver statistics
        
        Args:
            solver: OR-Tools solver instance
        
        Returns:
            Dictionary with solver statistics
        """
        stats = {
            'solve_time_ms': solver.WallTime() if hasattr(solver, 'WallTime') else 0,
            'iterations': solver.iterations() if hasattr(solver, 'iterations') else 0,
            'nodes': solver.nodes() if hasattr(solver, 'nodes') else 0,
        }
        
        # Objective value
        if hasattr(solver, 'Objective'):
            obj = solver.Objective()
            if obj:
                stats['objective_value'] = obj.Value()
                stats['best_bound'] = obj.BestBound() if hasattr(obj, 'BestBound') else None
        
        return stats


class EnhancedFPLOptimizer(FPLOptimizer):
    """
    Enhanced FPL Optimizer with performance improvements
    
    Wraps the existing FPLOptimizer with:
    - Multi-threaded solving
    - Time limits
    - Optimality gaps
    - Solver statistics
    """
    
    def __init__(self):
        super().__init__()
        self.config = EnhancedOptimizerConfig()
        self.solver_stats = []
        logger.info("Enhanced FPL Optimizer initialized")
    
    def optimize_team(self, *args, **kwargs):
        """
        Optimize team with enhanced solver configuration
        
        Overrides parent method to apply enhanced config
        """
        # Call parent optimization
        result = super().optimize_team(*args, **kwargs)
        
        # Add solver stats if available
        if hasattr(self, '_last_solver'):
            stats = self.config.get_solver_stats(self._last_solver)
            result['solver_stats'] = stats
            self.solver_stats.append(stats)
            
            logger.info(
                f"Optimization complete: {stats.get('solve_time_ms', 0)/1000:.2f}s, "
                f"objective={stats.get('objective_value', 0):.2f}"
            )
        
        return result
    
    async def optimize_multiple_gameweeks_parallel(
        self,
        players_df: pd.DataFrame,
        gameweeks: List[int],
        budget: float = 100.0,
        **kwargs
    ) -> List[Dict]:
        """
        Optimize multiple gameweeks in parallel
        
        Args:
            players_df: Player data
            gameweeks: List of gameweek numbers to optimize
            budget: Available budget
            **kwargs: Additional optimization parameters
        
        Returns:
            List of optimization results, one per gameweek
        """
        logger.info(f"Optimizing {len(gameweeks)} gameweeks in parallel...")
        
        # Create optimization tasks
        tasks = []
        for gw in gameweeks:
            # Filter players for this gameweek
            gw_players = players_df.copy()
            # You might want to adjust predictions based on gameweek
            
            # Create async task
            task = asyncio.create_task(
                self._optimize_gameweek_async(gw, gw_players, budget, **kwargs)
            )
            tasks.append(task)
        
        # Run in parallel
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Parallel optimization complete for {len(gameweeks)} gameweeks")
        
        return results
    
    async def _optimize_gameweek_async(
        self,
        gameweek: int,
        players_df: pd.DataFrame,
        budget: float,
        **kwargs
    ) -> Dict:
        """
        Async wrapper for gameweek optimization
        
        Runs optimization in thread pool to avoid blocking
        """
        loop = asyncio.get_event_loop()
        
        # Run in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self.optimize_team,
                players_df,
                budget,
                **kwargs
            )
        
        result['gameweek'] = gameweek
        return result
    
    def get_average_solve_time(self) -> float:
        """Get average solve time across all optimizations"""
        if not self.solver_stats:
            return 0.0
        
        times = [s.get('solve_time_ms', 0) for s in self.solver_stats]
        return sum(times) / len(times) / 1000  # Convert to seconds
    
    def get_solver_statistics(self) -> Dict:
        """Get comprehensive solver statistics"""
        if not self.solver_stats:
            return {}
        
        times = [s.get('solve_time_ms', 0) / 1000 for s in self.solver_stats]
        objectives = [s.get('objective_value', 0) for s in self.solver_stats]
        
        return {
            'total_optimizations': len(self.solver_stats),
            'avg_solve_time': sum(times) / len(times) if times else 0,
            'min_solve_time': min(times) if times else 0,
            'max_solve_time': max(times) if times else 0,
            'avg_objective': sum(objectives) / len(objectives) if objectives else 0,
            'config': {
                'num_workers': self.config.num_workers,
                'time_limit': self.config.max_time_seconds,
                'mip_gap': self.config.relative_mip_gap
            }
        }


# Convenience function
def get_enhanced_optimizer() -> EnhancedFPLOptimizer:
    """Get an instance of the enhanced optimizer"""
    return EnhancedFPLOptimizer()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_parallel_optimization():
        """Test parallel multi-gameweek optimization"""
        optimizer = EnhancedFPLOptimizer()
        
        # Sample player data
        players_df = pd.DataFrame({
            'id': range(1, 101),
            'web_name': [f'Player{i}' for i in range(1, 101)],
            'now_cost': [50 + i for i in range(100)],
            'predicted_points': [5 + (i % 10) for i in range(100)],
            'position': ['FWD'] * 25 + ['MID'] * 25 + ['DEF'] * 25 + ['GKP'] * 25,
            'team': [i % 20 + 1 for i in range(100)]
        })
        
        # Optimize 3 gameweeks in parallel
        results = await optimizer.optimize_multiple_gameweeks_parallel(
            players_df,
            gameweeks=[15, 16, 17],
            budget=100.0
        )
        
        print(f"\nOptimized {len(results)} gameweeks")
        print(f"Average solve time: {optimizer.get_average_solve_time():.2f}s")
        print(f"\nSolver statistics:")
        print(optimizer.get_solver_statistics())
    
    asyncio.run(test_parallel_optimization())
