"""
Model versioning and performance tracking system
Tracks model versions, performance metrics, and enables rollback
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Manage ML model versions and performance"""
    
    def __init__(self, models_dir: str = "backend/data/models"):
        self.models_dir = Path(models_dir)
        self.versions_file = self.models_dir / "versions.json"
        self.metrics_file = self.models_dir / "metrics.json"
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize version tracking
        self._initialize_tracking()
    
    def _initialize_tracking(self):
        """Initialize version and metrics tracking files"""
        if not self.versions_file.exists():
            self._save_json(self.versions_file, {})
        
        if not self.metrics_file.exists():
            self._save_json(self.metrics_file, {})
    
    def _load_json(self, filepath: Path) -> Dict:
        """Load JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return {}
    
    def _save_json(self, filepath: Path, data: Dict):
        """Save JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving {filepath}: {e}")
    
    def save_model_version(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a new model version
        
        Args:
            model_name: Name of the model (e.g., 'ensemble', 'minutes')
            model_path: Path to saved model file
            metrics: Performance metrics
            metadata: Additional metadata
        
        Returns:
            Version ID
        """
        version_id = f"{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load existing versions
        versions = self._load_json(self.versions_file)
        
        # Add new version
        if model_name not in versions:
            versions[model_name] = []
        
        version_info = {
            'version_id': version_id,
            'model_path': model_path,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {},
            'is_active': True  # Mark as active
        }
        
        # Deactivate previous versions
        for v in versions[model_name]:
            v['is_active'] = False
        
        versions[model_name].append(version_info)
        
        # Save versions
        self._save_json(self.versions_file, versions)
        
        # Save metrics
        self._save_metrics(version_id, metrics)
        
        self.logger.info(f"✅ Saved model version: {version_id}")
        return version_id
    
    def _save_metrics(self, version_id: str, metrics: Dict):
        """Save metrics for a version"""
        all_metrics = self._load_json(self.metrics_file)
        all_metrics[version_id] = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self._save_json(self.metrics_file, all_metrics)
    
    def get_active_version(self, model_name: str) -> Optional[Dict]:
        """Get currently active version for a model"""
        versions = self._load_json(self.versions_file)
        
        if model_name not in versions:
            return None
        
        for version in reversed(versions[model_name]):
            if version.get('is_active', False):
                return version
        
        return None
    
    def get_version_history(self, model_name: str) -> List[Dict]:
        """Get version history for a model"""
        versions = self._load_json(self.versions_file)
        return versions.get(model_name, [])
    
    def compare_versions(
        self,
        model_name: str,
        version_id_1: str,
        version_id_2: str
    ) -> Dict:
        """
        Compare two model versions
        
        Args:
            model_name: Model name
            version_id_1: First version ID
            version_id_2: Second version ID
        
        Returns:
            Comparison results
        """
        versions = self.get_version_history(model_name)
        
        v1 = next((v for v in versions if v['version_id'] == version_id_1), None)
        v2 = next((v for v in versions if v['version_id'] == version_id_2), None)
        
        if not v1 or not v2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version_1': {
                'id': version_id_1,
                'created_at': v1['created_at'],
                'metrics': v1['metrics']
            },
            'version_2': {
                'id': version_id_2,
                'created_at': v2['created_at'],
                'metrics': v2['metrics']
            },
            'improvements': {}
        }
        
        # Calculate improvements
        for metric in v1['metrics']:
            if metric in v2['metrics']:
                v1_val = v1['metrics'][metric]
                v2_val = v2['metrics'][metric]
                
                if isinstance(v1_val, (int, float)) and isinstance(v2_val, (int, float)):
                    improvement = ((v2_val - v1_val) / v1_val * 100) if v1_val != 0 else 0
                    comparison['improvements'][metric] = {
                        'v1': v1_val,
                        'v2': v2_val,
                        'change_percent': improvement
                    }
        
        return comparison
    
    def rollback_to_version(self, model_name: str, version_id: str) -> bool:
        """
        Rollback to a previous version
        
        Args:
            model_name: Model name
            version_id: Version to rollback to
        
        Returns:
            Success status
        """
        versions = self._load_json(self.versions_file)
        
        if model_name not in versions:
            self.logger.error(f"Model {model_name} not found")
            return False
        
        # Find target version
        target_version = None
        for v in versions[model_name]:
            if v['version_id'] == version_id:
                target_version = v
                break
        
        if not target_version:
            self.logger.error(f"Version {version_id} not found")
            return False
        
        # Deactivate all versions
        for v in versions[model_name]:
            v['is_active'] = False
        
        # Activate target version
        target_version['is_active'] = True
        target_version['rollback_at'] = datetime.now().isoformat()
        
        # Save
        self._save_json(self.versions_file, versions)
        
        self.logger.info(f"✅ Rolled back {model_name} to {version_id}")
        return True
    
    def get_performance_summary(self, model_name: str) -> Dict:
        """Get performance summary for a model"""
        versions = self.get_version_history(model_name)
        
        if not versions:
            return {'error': 'No versions found'}
        
        # Get metrics over time
        metrics_over_time = []
        for v in versions:
            metrics_over_time.append({
                'version_id': v['version_id'],
                'created_at': v['created_at'],
                'is_active': v.get('is_active', False),
                **v['metrics']
            })
        
        # Current active version
        active = self.get_active_version(model_name)
        
        return {
            'model_name': model_name,
            'total_versions': len(versions),
            'active_version': active['version_id'] if active else None,
            'active_metrics': active['metrics'] if active else {},
            'history': metrics_over_time
        }
    
    def cleanup_old_versions(
        self,
        model_name: str,
        keep_last_n: int = 5
    ) -> int:
        """
        Clean up old model versions, keeping only the last N
        
        Args:
            model_name: Model name
            keep_last_n: Number of versions to keep
        
        Returns:
            Number of versions deleted
        """
        versions = self._load_json(self.versions_file)
        
        if model_name not in versions:
            return 0
        
        model_versions = versions[model_name]
        
        if len(model_versions) <= keep_last_n:
            return 0
        
        # Sort by creation date
        sorted_versions = sorted(
            model_versions,
            key=lambda x: x['created_at'],
            reverse=True
        )
        
        # Keep last N and active version
        to_keep = sorted_versions[:keep_last_n]
        active_version = self.get_active_version(model_name)
        
        if active_version and active_version not in to_keep:
            to_keep.append(active_version)
        
        # Delete old model files
        deleted_count = 0
        for v in model_versions:
            if v not in to_keep:
                model_path = Path(v['model_path'])
                if model_path.exists():
                    try:
                        model_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        self.logger.error(f"Error deleting {model_path}: {e}")
        
        # Update versions
        versions[model_name] = to_keep
        self._save_json(self.versions_file, versions)
        
        self.logger.info(f"🗑️  Cleaned up {deleted_count} old versions of {model_name}")
        return deleted_count
