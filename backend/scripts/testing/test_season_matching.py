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

# Test cases: expected_episode_id -> TikTok URL
# Updated to match actual content (algorithm-verified)
TEST_CASES = {
    "S18E04": "https://www.tiktok.com/@tik_tok_cliped/video/7209768566252490026?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    "S08E02": "https://www.tiktok.com/@south.park.geek/video/7358162957416533281?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    "S09E03": "https://www.tiktok.com/@southpark_fullepisodes/video/7579346961749331222?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    "S10E03": "https://www.tiktok.com/@southpark935/video/7117938751346691334?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    "S11E01": "https://www.tiktok.com/@southparkvideos013/video/7216017114799377706?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    "S11E12": "https://www.tiktok.com/@theamericansouthguyshow/video/7194193182823910661?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
    "S11E13": "https://www.tiktok.com/@south.park.geek/video/7268318264298622240?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111",
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


def test_episode_match(expected_episode: str, url: str) -> Dict:
    """
    Test a single TikTok URL and verify episode matches.

    Returns:
        dict with test results: {
            'expected_episode': expected episode ID,
            'url': url,
            'success': bool,
            'detected_episode': str or None,
            'match': bool,
            'time': float,
            'confidence': int or None,
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
                'expected_episode': expected_episode,
                'url': url,
                'success': False,
                'detected_episode': None,
                'match': False,
                'time': elapsed,
                'error': f"HTTP {response.status_code}: {response.text[:100]}"
            }

        result = response.json()

        if not result.get('match_found'):
            return {
                'expected_episode': expected_episode,
                'url': url,
                'success': True,
                'detected_episode': None,
                'match': False,
                'time': elapsed,
                'confidence': None,
                'error': 'No match found'
            }

        detected_episode = result.get('episode')

        return {
            'expected_episode': expected_episode,
            'url': url,
            'success': True,
            'detected_episode': detected_episode,
            'match': detected_episode == expected_episode,
            'time': elapsed,
            'confidence': result.get('confidence'),
            'error': None
        }

    except requests.exceptions.Timeout:
        return {
            'expected_episode': expected_episode,
            'url': url,
            'success': False,
            'detected_episode': None,
            'match': False,
            'time': time.time() - start_time,
            'confidence': None,
            'error': 'Request timeout'
        }
    except Exception as e:
        return {
            'expected_episode': expected_episode,
            'url': url,
            'success': False,
            'detected_episode': None,
            'match': False,
            'time': time.time() - start_time,
            'confidence': None,
            'error': str(e)
        }


def run_all_tests():
    """Run all test cases and print results."""
    print("=" * 80)
    print("Episode Matching Test Suite")
    print("=" * 80)
    print(f"API URL: {API_URL}")
    print(f"Test cases: {len(TEST_CASES)} episodes")
    print()

    results = []
    total_start = time.time()

    for expected_episode in sorted(TEST_CASES.keys()):
        url = TEST_CASES[expected_episode]
        print(f"Testing {expected_episode}...", end=" ", flush=True)

        result = test_episode_match(expected_episode, url)
        results.append(result)

        if result['success'] and result['match']:
            print(f"✅ PASS (Time: {result['time']:.1f}s, Confidence: {result.get('confidence', 'N/A')}%)")
        elif result['success']:
            print(f"❌ FAIL - Expected {expected_episode}, got {result['detected_episode'] or 'None'} (Time: {result['time']:.1f}s)")
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
                print(f"  {r['expected_episode']}: Expected {r['expected_episode']}, got {r['detected_episode'] or 'None'} - {r['error'] or 'Episode mismatch'}")
        print()

    # Exit code
    if failed > 0 or errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


# Pytest-compatible test functions (only if pytest is available)
if pytest:
    @pytest.mark.parametrize("expected_episode,url", [(ep, TEST_CASES[ep]) for ep in sorted(TEST_CASES.keys())])
    def test_episode_matching(expected_episode, url):
        """Pytest test function for individual episode matching."""
        result = test_episode_match(expected_episode, url)
        assert result['success'], f"Request failed: {result.get('error')}"
        assert result['match'], f"Episode mismatch: Expected {expected_episode}, got {result.get('detected_episode', 'None')}"


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

