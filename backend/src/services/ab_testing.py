"""
A/B Testing Framework

Enables testing different recommendation strategies and measuring their effectiveness.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

from backend.src.core.postgres_db import PostgresManagerDB


logger = logging.getLogger(__name__)


class ABTestingFramework:
    """Framework for A/B testing different recommendation strategies"""
    
    def __init__(self):
        self.postgres_db = PostgresManagerDB()
        self.logger = logging.getLogger(__name__)
        self._initialize_ab_tables()
    
    def _initialize_ab_tables(self):
        """Create A/B testing tables if they don't exist"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Experiments table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ab_experiments (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(200) UNIQUE NOT NULL,
                            description TEXT,
                            control_strategy VARCHAR(100),
                            treatment_strategy VARCHAR(100),
                            allocation_ratio NUMERIC(3,2) DEFAULT 0.5,
                            start_date TIMESTAMP DEFAULT NOW(),
                            end_date TIMESTAMP,
                            status VARCHAR(50) DEFAULT 'active',
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    
                    # User assignments table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ab_user_assignments (
                            user_id INTEGER,
                            experiment_id INTEGER REFERENCES ab_experiments(id),
                            variant VARCHAR(50),
                            assigned_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (user_id, experiment_id)
                        );
                    """)
                    
                    # Results table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ab_experiment_results (
                            id SERIAL PRIMARY KEY,
                            experiment_id INTEGER REFERENCES ab_experiments(id),
                            user_id INTEGER,
                            variant VARCHAR(50),
                            metric_name VARCHAR(100),
                            metric_value NUMERIC(10,2),
                            gameweek INTEGER,
                            recorded_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_ab_results_experiment ON ab_experiment_results(experiment_id, variant);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_ab_assignments_user ON ab_user_assignments(user_id);")
                    
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error initializing A/B tables: {str(e)}")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        control_strategy: str,
        treatment_strategy: str,
        allocation_ratio: float = 0.5
    ) -> Optional[int]:
        """
        Create a new A/B test experiment
        
        Args:
            name: Experiment name (unique)
            description: Description of what's being tested
            control_strategy: Name of control strategy
            treatment_strategy: Name of treatment strategy
            allocation_ratio: Ratio of users in treatment (0-1)
        
        Returns:
            Experiment ID or None if failed
        """
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ab_experiments (
                            name, description, control_strategy, treatment_strategy, allocation_ratio
                        ) VALUES (%s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (name, description, control_strategy, treatment_strategy, allocation_ratio))
                    
                    result = cur.fetchone()
                    experiment_id = result['id'] if result else None
                    
                    conn.commit()
                    
                    self.logger.info(f"Created experiment '{name}' with ID {experiment_id}")
                    return experiment_id
                    
        except Exception as e:
            self.logger.error(f"Error creating experiment: {str(e)}")
            return None
    
    def assign_user_to_variant(
        self,
        user_id: int,
        experiment_id: int
    ) -> str:
        """
        Assign user to control or treatment variant
        
        Uses deterministic hashing to ensure consistent assignment
        
        Args:
            user_id: User ID
            experiment_id: Experiment ID
        
        Returns:
            Variant name ('control' or 'treatment')
        """
        # Check if already assigned
        existing = self._get_user_assignment(user_id, experiment_id)
        if existing:
            return existing
        
        # Get experiment details
        experiment = self._get_experiment(experiment_id)
        if not experiment:
            return 'control'
        
        # Deterministic hash-based assignment
        hash_input = f"{user_id}:{experiment_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        ratio = (hash_value % 100) / 100.0
        
        variant = 'treatment' if ratio < experiment['allocation_ratio'] else 'control'
        
        # Store assignment
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ab_user_assignments (user_id, experiment_id, variant)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, experiment_id) DO NOTHING;
                    """, (user_id, experiment_id, variant))
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing assignment: {str(e)}")
        
        return variant
    
    def _get_user_assignment(self, user_id: int, experiment_id: int) -> Optional[str]:
        """Get existing user assignment"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT variant FROM ab_user_assignments
                        WHERE user_id = %s AND experiment_id = %s;
                    """, (user_id, experiment_id))
                    result = cur.fetchone()
                    return result['variant'] if result else None
        except Exception as e:
            self.logger.error(f"Error getting assignment: {str(e)}")
            return None
    
    def _get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment details"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT * FROM ab_experiments WHERE id = %s;
                    """, (experiment_id,))
                    return cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error getting experiment: {str(e)}")
            return None
    
    def record_metric(
        self,
        experiment_id: int,
        user_id: int,
        metric_name: str,
        metric_value: float,
        gameweek: int = None
    ):
        """
        Record a metric for A/B test analysis
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID
            metric_name: Name of metric (e.g., 'points_gained', 'user_satisfaction')
            metric_value: Metric value
            gameweek: Gameweek number (optional)
        """
        variant = self._get_user_assignment(user_id, experiment_id)
        if not variant:
            self.logger.warning(f"User {user_id} not assigned to experiment {experiment_id}")
            return
        
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ab_experiment_results (
                            experiment_id, user_id, variant, metric_name, metric_value, gameweek
                        ) VALUES (%s, %s, %s, %s, %s, %s);
                    """, (experiment_id, user_id, variant, metric_name, metric_value, gameweek))
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Error recording metric: {str(e)}")
    
    def get_experiment_results(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get results for an experiment
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Dictionary with results by variant
        """
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get summary statistics
                    cur.execute("""
                        SELECT 
                            variant,
                            metric_name,
                            COUNT(*) as sample_size,
                            AVG(metric_value) as mean,
                            STDDEV(metric_value) as std_dev,
                            MIN(metric_value) as min_value,
                            MAX(metric_value) as max_value
                        FROM ab_experiment_results
                        WHERE experiment_id = %s
                        GROUP BY variant, metric_name
                        ORDER BY variant, metric_name;
                    """, (experiment_id,))
                    
                    results = cur.fetchall()
                    
                    # Organize by variant
                    by_variant = {'control': {}, 'treatment': {}}
                    for row in results:
                        variant = row['variant']
                        metric = row['metric_name']
                        by_variant[variant][metric] = {
                            'sample_size': row['sample_size'],
                            'mean': float(row['mean']) if row['mean'] else 0,
                            'std_dev': float(row['std_dev']) if row['std_dev'] else 0,
                            'min': float(row['min_value']) if row['min_value'] else 0,
                            'max': float(row['max_value']) if row['max_value'] else 0
                        }
                    
                    return by_variant
                    
        except Exception as e:
            self.logger.error(f"Error getting experiment results: {str(e)}")
            return {}
    
    def calculate_statistical_significance(
        self,
        experiment_id: int,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance of experiment results
        
        Uses Welch's t-test for comparing means
        
        Args:
            experiment_id: Experiment ID
            metric_name: Metric to analyze
        
        Returns:
            Dictionary with statistical test results
        """
        results = self.get_experiment_results(experiment_id)
        
        if metric_name not in results.get('control', {}) or metric_name not in results.get('treatment', {}):
            return {'error': 'Insufficient data for statistical test'}
        
        control = results['control'][metric_name]
        treatment = results['treatment'][metric_name]
        
        # Calculate t-statistic (Welch's t-test)
        n1, n2 = control['sample_size'], treatment['sample_size']
        mean1, mean2 = control['mean'], treatment['mean']
        std1, std2 = control['std_dev'], treatment['std_dev']
        
        if n1 < 2 or n2 < 2:
            return {'error': 'Insufficient sample size'}
        
        # Welch's t-test
        se = ((std1**2 / n1) + (std2**2 / n2)) ** 0.5
        if se == 0:
            return {'error': 'Zero standard error'}
        
        t_stat = (mean2 - mean1) / se
        
        # Degrees of freedom (Welch-Satterthwaite)
        df = ((std1**2/n1 + std2**2/n2)**2) / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
        
        # Simple p-value approximation (for |t| > 2, p < 0.05)
        significant = abs(t_stat) > 2.0
        
        # Effect size (Cohen's d)
        pooled_std = ((std1**2 + std2**2) / 2) ** 0.5
        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        
        return {
            'control_mean': mean1,
            'treatment_mean': mean2,
            'difference': mean2 - mean1,
            'percent_change': ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0,
            't_statistic': t_stat,
            'degrees_of_freedom': df,
            'significant': significant,
            'cohens_d': cohens_d,
            'control_sample_size': n1,
            'treatment_sample_size': n2
        }
    
    def end_experiment(self, experiment_id: int):
        """Mark experiment as ended"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ab_experiments
                        SET status = 'ended', end_date = NOW()
                        WHERE id = %s;
                    """, (experiment_id,))
                    conn.commit()
                    self.logger.info(f"Ended experiment {experiment_id}")
        except Exception as e:
            self.logger.error(f"Error ending experiment: {str(e)}")
    
    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT * FROM ab_experiments
                        WHERE status = 'active'
                        ORDER BY created_at DESC;
                    """)
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error getting active experiments: {str(e)}")
            return []
