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

# Test cases: list of (season_number, TikTok URL) tuples
# Multiple URLs per season are supported
TEST_CASES = [
    # Season 1
    (1, "https://www.tiktok.com/@tik_tok_cliped/video/7209768566252490026?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 2
    (2, "https://www.tiktok.com/@theamericansouthguyshow/video/7187933650447584517?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 3
    (3, "https://www.tiktok.com/@theamericansouthguyshow/video/7189384225131384069?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 4
    (4, "https://www.tiktok.com/@funnyclips6377/video/7374095866186976554?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 5
    (5, "https://www.tiktok.com/@theamericansouthguyshow/video/7194193182823910661?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 6
    (6, "https://www.tiktok.com/@south.park.geek/video/7268318264298622240?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 7
    (7, "https://www.tiktok.com/@shroombro/video/7316089360074476842?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 8 - Multiple test cases
    (8, "https://www.tiktok.com/@randomtvclips63/video/7186824962966129926?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (8, "https://www.tiktok.com/@randomtvclips63/video/7265267618444053792?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (8, "https://www.tiktok.com/@randomtvclips63/video/7209791745511492870?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (8, "https://www.tiktok.com/@juniperberii/video/7209319779239267590?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (8, "https://www.tiktok.com/@ssouthparkclipz/video/7206219254130248966?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 9
    (9, "https://www.tiktok.com/@southpark_fullepisodes/video/7579346961749331222?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 10
    (10, "https://www.tiktok.com/@southpark935/video/7117938751346691334?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 11
    (11, "https://www.tiktok.com/@southparkvideos013/video/7216017114799377706?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 12
    (12, "https://www.tiktok.com/@southparknator_/video/7178255122722671918?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 14 - Multiple test cases
    (14, "https://www.tiktok.com/@cartmansbedroom/video/7191866921539685678?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (14, "https://www.tiktok.com/@spclips._/video/7531687511618260238?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 15 - Multiple test cases
    (15, "https://www.tiktok.com/@southparkclips7183/video/7349630527990811906?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (15, "https://www.tiktok.com/@flawedrealityyt/video/7243968033973177642?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (15, "https://www.tiktok.com/@ashethecrash/video/7233604932404809002?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (15, "https://www.tiktok.com/@flawedrealityyt/video/7223937358008159530?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (15, "https://www.tiktok.com/@clipsofsouth/video/7220758175908203781?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 16 - Multiple test cases
    (16, "https://www.tiktok.com/@thecartmanlover/video/7202289342306454826?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (16, "https://www.tiktok.com/@brrrberrysh0rty2/video/7201186627471904046?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (16, "https://www.tiktok.com/@followformore666/video/7201712113344318766?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (16, "https://www.tiktok.com/@southparkfunnyclips3/video/7192311517889580334?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (16, "https://www.tiktok.com/@southparkfunnyclips3/video/7192375296304172330?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 17 - Multiple test cases
    (17, "https://www.tiktok.com/@tv_shows_for_tiktok/video/7229125680053898542?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (17, "https://www.tiktok.com/@spamjoecool/video/7529331467130883359?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (17, "https://www.tiktok.com/@southparkepisodes_/video/7199431355393051950?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (17, "https://www.tiktok.com/@fgsp_movies/video/7257032582179753262?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (17, "https://www.tiktok.com/@southparkclips85/video/7178924663916055854?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 18 - Multiple test cases
    (18, "https://www.tiktok.com/@spclips._/video/7532211509049478413?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (18, "https://www.tiktok.com/@randomtvclips63/video/7216847351804185861?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (18, "https://www.tiktok.com/@southparkclips450/video/7535450188144528662?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (18, "https://www.tiktok.com/@cartman_x_heidi_official/video/7186039862758313258?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (18, "https://www.tiktok.com/@jussumclipz4u/video/7185004032623217963?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 19 - Multiple test cases
    (19, "https://www.tiktok.com/@its.just.m3.bruhh/video/7553774238046375223?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (19, "https://www.tiktok.com/@spamjoecool/video/7528415868414381342?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (19, "https://www.tiktok.com/@flawedrealityyt/video/7227135393789922606?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    # Season 20 - Multiple test cases
    (20, "https://www.tiktok.com/@southparkclipz01/video/7186738651391479045?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (20, "https://www.tiktok.com/@animationcontentenjoyer/video/7576418097700097302?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (20, "https://www.tiktok.com/@southparkclipz01/video/7186738651391462661?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
    (20, "https://www.tiktok.com/@tiktokspot6942/video/7219404724196478251?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"),
]

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
            'match_rank': int or None,  # 1, 2, or 3 if matched in top 3
            'time': float,
            'confidence': int or None,
            'unsure': bool,
            'candidates': list,  # Top 3 candidates
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
                'match_rank': None,
                'time': elapsed,
                'confidence': None,
                'unsure': True,
                'candidates': [],
                'error': f"HTTP {response.status_code}: {response.text[:100]}"
            }

        result = response.json()

        # Check top 3 candidates
        candidates = result.get('candidates', [])
        match_rank = None
        episode_id = None
        detected_season = None
        
        # Check if any candidate matches the expected season
        for rank, candidate in enumerate(candidates[:3], 1):
            candidate_episode = candidate.get('episode_id')
            if candidate_episode:
                candidate_season = extract_season_from_episode_id(candidate_episode)
                if candidate_season == season:
                    match_rank = rank
                    episode_id = candidate_episode
                    detected_season = candidate_season
                    break
        
        # If no candidate matches, use the first one (or episode from match_found)
        if not episode_id:
            if result.get('match_found') and result.get('episode'):
                episode_id = result.get('episode')
                detected_season = extract_season_from_episode_id(episode_id) if episode_id else None
            elif candidates:
                episode_id = candidates[0].get('episode_id')
                detected_season = extract_season_from_episode_id(episode_id) if episode_id else None

        return {
            'season': season,
            'url': url,
            'success': True,
            'episode_id': episode_id,
            'detected_season': detected_season,
            'match': match_rank is not None,
            'match_rank': match_rank,  # 1, 2, or 3 if matched, None if not
            'time': elapsed,
            'confidence': result.get('confidence'),
            'unsure': result.get('unsure', True),
            'candidates': candidates[:3],  # Top 3 candidates
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
            'match_rank': None,
            'time': time.time() - start_time,
            'confidence': None,
            'unsure': True,
            'candidates': [],
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
            'match_rank': None,
            'time': time.time() - start_time,
            'confidence': None,
            'unsure': True,
            'candidates': [],
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
    
    # Sort test cases by season number, then by URL for consistency
    sorted_tests = sorted(TEST_CASES, key=lambda x: (x[0], x[1]))
    
    for season, url in sorted_tests:
        # Extract short identifier from URL for display
        url_id = url.split('/video/')[1].split('?')[0] if '/video/' in url else url[-20:]
        print(f"Testing Season {season:2d} ({url_id[:12]}...)...", end=" ", flush=True)
        
        result = test_season_match(season, url)
        results.append(result)
        
        if result['success'] and result['match']:
            rank_str = f" (Rank {result['match_rank']})" if result.get('match_rank') else ""
            unsure_str = " [UNSURE]" if result.get('unsure') else ""
            print(f"✅ PASS{rank_str}{unsure_str} (Episode: {result['episode_id']}, Time: {result['time']:.1f}s, Confidence: {result.get('confidence', 'N/A')}%)")
        elif result['success']:
            # Show which candidates were returned
            candidates = result.get('candidates', [])
            candidate_str = ""
            if candidates:
                candidate_episodes = [c.get('episode_id', 'N/A') for c in candidates[:3]]
                candidate_str = f" [Candidates: {', '.join(candidate_episodes)}]"
            print(f"❌ FAIL - Expected S{season:02d}, got {result['episode_id'] or 'None'}{candidate_str} (Time: {result['time']:.1f}s)")
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
    
    # Count matches by rank
    rank_1_matches = sum(1 for r in results if r.get('match_rank') == 1)
    rank_2_matches = sum(1 for r in results if r.get('match_rank') == 2)
    rank_3_matches = sum(1 for r in results if r.get('match_rank') == 3)

    print(f"Total tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"   - Rank 1 (1st guess): {rank_1_matches}")
    print(f"   - Rank 2 (2nd guess): {rank_2_matches}")
    print(f"   - Rank 3 (3rd guess): {rank_3_matches}")
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
                candidates = r.get('candidates', [])
                candidate_list = [c.get('episode_id', 'N/A') for c in candidates[:3]] if candidates else []
                candidate_str = f" [Top 3: {', '.join(candidate_list)}]" if candidate_list else ""
                print(f"  Season {r['season']:2d}: Expected S{r['season']:02d}, got {r['episode_id'] or 'None'}{candidate_str} - {r['error'] or 'Season mismatch'}")
        print()

    # Exit code
    if failed > 0 or errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


# Pytest-compatible test functions (only if pytest is available)
if pytest:
    @pytest.mark.parametrize("season,url", TEST_CASES)
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

