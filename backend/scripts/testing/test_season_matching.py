#!/usr/bin/env python3
"""
Test suite for season matching accuracy.

Tests TikTok URLs for each season and verifies the returned episode 
belongs to the correct season.

Run with: 
  - pytest test_season_matching.py -v
  - python test_season_matching.py
  - make test-seasons
"""
try:
    import pytest
except ImportError:
    pytest = None

import sys
import os
import re
import time
import requests
from typing import Dict, Optional

# Test cases: season_number -> TikTok URL
TEST_CASES = {
    1: "https://www.tiktok.com/@tik_tok_cliped/video/7209768566252490026?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    2: "https://www.tiktok.com/@theamericansouthguyshow/video/7187933650447584517?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    3: "https://www.tiktok.com/@theamericansouthguyshow/video/7189384225131384069?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    4: "https://www.tiktok.com/@funnyclips6377/video/7374095866186976554?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    5: "https://www.tiktok.com/@theamericansouthguyshow/video/7194193182823910661?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    6: "https://www.tiktok.com/@south.park.geek/video/7268318264298622240?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    7: "https://www.tiktok.com/@shroombro/video/7316089360074476842?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    8: "https://www.tiktok.com/@south.park.geek/video/7358162957416533281?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    9: "https://www.tiktok.com/@southpark_fullepisodes/video/7579346961749331222?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    10: "https://www.tiktok.com/@southpark935/video/7117938751346691334?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    11: "https://www.tiktok.com/@southparkvideos013/video/7216017114799377706?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    12: "https://www.tiktok.com/@southparknator_/video/7178255122722671918?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
}

# API configuration
API_URL = os.getenv("API_URL", "https://citywok-production.up.railway.app")
TIMEOUT = 120  # 120 seconds per request
INITIAL_TIMEOUT = 1  # 500 seconds for initial health check (allows time for cold start)


def extract_season_from_episode_id(episode_id: str) -> Optional[int]:
    """
    Extract season number from episode ID.
    
    Examples:
        "S01E05" -> 1
        "S12E14" -> 12
        "S20E10" -> 20
    """
    match = re.search(r'[Ss](\d+)', episode_id)
    if match:
        return int(match.group(1))
    return None


def test_season_match(season: int, url: str) -> Dict:
    """
    Test a single TikTok URL and verify season matches.
    
    Returns:
        dict with test results: {
            'season': expected_season,
            'url': url,
            'success': bool,
            'episode_id': str or None,
            'detected_season': int or None,
            'match': bool,
            'time': float,
            'error': str or None
        }
    """
    start_time = time.time()
    
    try:
        # Make API request
        form_data = {'url': url}
        response = requests.post(
            f"{API_URL}/api/v1/identify",
            data=form_data,
            timeout=TIMEOUT
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            return {
                'season': season,
                'url': url,
                'success': False,
                'episode_id': None,
                'detected_season': None,
                'match': False,
                'time': elapsed,
                'error': f"HTTP {response.status_code}: {response.text[:100]}"
            }
        
        result = response.json()
        
        if not result.get('match_found'):
            return {
                'season': season,
                'url': url,
                'success': True,
                'episode_id': None,
                'detected_season': None,
                'match': False,
                'time': elapsed,
                'error': 'No match found'
            }
        
        episode_id = result.get('episode')
        detected_season = extract_season_from_episode_id(episode_id) if episode_id else None
        
        return {
            'season': season,
            'url': url,
            'success': True,
            'episode_id': episode_id,
            'detected_season': detected_season,
            'match': detected_season == season,
            'time': elapsed,
            'confidence': result.get('confidence'),
            'error': None
        }
        
    except requests.exceptions.Timeout:
        return {
            'season': season,
            'url': url,
            'success': False,
            'episode_id': None,
            'detected_season': None,
            'match': False,
            'time': time.time() - start_time,
            'error': 'Request timeout'
        }
    except Exception as e:
        return {
            'season': season,
            'url': url,
            'success': False,
            'episode_id': None,
            'detected_season': None,
            'match': False,
            'time': time.time() - start_time,
            'error': str(e)
        }


def run_all_tests():
    """Run all test cases and print results."""
    print("=" * 80)
    print("Season Matching Test Suite")
    print("=" * 80)
    print(f"API URL: {API_URL}")
    print(f"Test cases: {len(TEST_CASES)} seasons")
    print()
    
    results = []
    total_start = time.time()
    
    for season in sorted(TEST_CASES.keys()):
        url = TEST_CASES[season]
        print(f"Testing Season {season:2d}...", end=" ", flush=True)
        
        result = test_season_match(season, url)
        results.append(result)
        
        if result['success'] and result['match']:
            print(f"✅ PASS (Episode: {result['episode_id']}, Time: {result['time']:.1f}s, Confidence: {result.get('confidence', 'N/A')}%)")
        elif result['success']:
            print(f"❌ FAIL - Expected S{season:02d}, got {result['episode_id'] or 'None'} (Time: {result['time']:.1f}s)")
        else:
            print(f"❌ ERROR - {result['error']} (Time: {result['time']:.1f}s)")
    
    total_time = time.time() - total_start
    
    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['success'] and r['match'])
    failed = sum(1 for r in results if r['success'] and not r['match'])
    errors = sum(1 for r in results if not r['success'])
    
    print(f"Total tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Errors: {errors}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"⏱️  Average time per test: {total_time/len(results):.1f}s")
    print()
    
    # Failed tests details
    if failed > 0 or errors > 0:
        print("Failed/Error Details:")
        for r in results:
            if not r['match'] or not r['success']:
                print(f"  Season {r['season']:2d}: Expected S{r['season']:02d}, got {r['episode_id'] or 'None'} - {r['error'] or 'Season mismatch'}")
        print()
    
    # Exit code
    if failed > 0 or errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


# Pytest-compatible test functions (only if pytest is available)
if pytest:
    @pytest.mark.parametrize("season,url", [(s, TEST_CASES[s]) for s in sorted(TEST_CASES.keys())])
    def test_season_matching(season, url):
        """Pytest test function for individual season matching."""
        result = test_season_match(season, url)
        assert result['success'], f"Request failed: {result.get('error')}"
        assert result['match'], f"Season mismatch: Expected S{season:02d}, got {result.get('episode_id', 'None')}"


if __name__ == "__main__":
    # Check if backend is running (with extended timeout for cold start)
    print(f"Checking backend health at {API_URL}...")
    print(f"   (Waiting up to {INITIAL_TIMEOUT}s for cold start if needed)")
    try:
        response = requests.get(f"{API_URL}/api/v1/test", timeout=INITIAL_TIMEOUT)
        if response.status_code != 200:
            print(f"❌ Backend is not responding correctly at {API_URL}")
            sys.exit(1)
        print(f"✅ Backend is ready!\n")
    except Exception as e:
        print(f"❌ Backend is not running at {API_URL}")
        print(f"   Error: {e}")
        sys.exit(1)
    
    run_all_tests()

