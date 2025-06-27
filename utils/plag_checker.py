import requests
import os
import json
from typing import Dict, Optional
import hashlib
import time

def basic_similarity_check(text: str, reference_texts: list = None) -> Dict:
    """Basic similarity check as fallback"""
    try:
        # Simple heuristic-based plagiarism detection
        text_length = len(text.strip())
        
        if text_length == 0:
            return {
                'plagiarism_percentage': 0,
                'sources_found': [],
                'api_used': 'basic_fallback',
                'note': 'Empty text provided'
            }
        
        # Check for common patterns that might indicate copying
        suspicious_patterns = [
            'copy', 'paste', 'copied from', 'source:', 'wikipedia',
            'according to', 'retrieved from', 'cited from'
        ]
        
        text_lower = text.lower()
        pattern_matches = sum(1 for pattern in suspicious_patterns if pattern in text_lower)
        
        # Simple scoring
        plagiarism_percentage = min(pattern_matches * 15, 50)  # Max 50% for pattern matching
        
        # Check for very short answers (might indicate incomplete copying)
        if text_length < 30:
            plagiarism_percentage = max(plagiarism_percentage, 10)
        
        return {
            'plagiarism_percentage': plagiarism_percentage,
            'sources_found': [],
            'api_used': 'basic_fallback',
            'patterns_detected': pattern_matches,
            'note': 'Basic heuristic-based detection used'
        }
        
    except Exception as e:
        print(f"Error in basic similarity check: {str(e)}")
        return {
            'plagiarism_percentage': 0,
            'sources_found': [],
            'api_used': 'error_fallback',
            'error': str(e)
        }

def check_plagiarism(text: str) -> Dict:
    """Main plagiarism checking function (basic only)"""
    return basic_similarity_check(text)

def get_plagiarism_explanation(plagiarism_data: Dict) -> str:
    """Generate explanation for plagiarism results"""
    try:
        percentage = plagiarism_data.get('plagiarism_percentage', 0)
        api_used = plagiarism_data.get('api_used', 'unknown')
        
        if percentage == 0:
            return "No plagiarism detected."
        elif percentage < 15:
            return f"Low similarity detected ({percentage:.1f}%). This is within acceptable limits."
        elif percentage < 30:
            return f"Moderate similarity detected ({percentage:.1f}%). Some content may be borrowed."
        elif percentage < 50:
            return f"High similarity detected ({percentage:.1f}%). Significant portions may be copied."
        else:
            return f"Very high similarity detected ({percentage:.1f}%). Major plagiarism concerns."
    
    except Exception as e:
        return f"Error generating plagiarism explanation: {str(e)}"
