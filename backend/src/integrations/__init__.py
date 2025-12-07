"""
Data integrations package

External data source integrations for enhanced FPL predictions.
"""

from backend.src.integrations.fbref_scraper import FBRefScraper
from backend.src.integrations.xg_estimator import XGEstimator
from backend.src.integrations.xg_integrator import XGIntegrator

__all__ = [
    'FBRefScraper',
    'XGEstimator',
    'XGIntegrator',
]
