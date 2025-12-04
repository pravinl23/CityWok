#!/usr/bin/env python3
"""Quick database status check - writes to file for reliability"""

import sys
import os
import json
import pickle
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

output = []

# Check video database
output.append("=" * 60)
output.append("VIDEO DATABASE STATUS")
output.append("=" * 60)

metadata_path = settings.METADATA_DB_PATH
video_seasons = set()
audio_seasons = set()

if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    episodes = Counter([v['episode_id'] for v in data.values()])
    seasons = Counter()
    for ep in episodes.keys():
        if ep.startswith('S') and len(ep) >= 6:
            try:
                season = int(ep[1:3])
                seasons[season] += 1
                video_seasons.add(season)
            except:
                pass
    
    output.append(f"\nTotal episodes: {len(episodes)}")
    output.append(f"Total frames: {len(data)}")
    output.append("\nBy Season:")
    for s in sorted(seasons.keys()):
        count = sum(1 for ep in episodes.keys() if ep.startswith(f'S{s:02d}'))
        output.append(f"  Season {s:02d}: {count} episodes")
else:
    output.append("\n❌ Video database not found")

# Check audio database
output.append("\n" + "=" * 60)
output.append("AUDIO DATABASE STATUS")
output.append("=" * 60)

audio_db_path = os.path.join(settings.DATA_DIR, "audio_fingerprints.pkl")

if os.path.exists(audio_db_path):
    try:
        with open(audio_db_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old and new format
        if isinstance(data, dict) and 'fingerprints' in data:
            fingerprints = data['fingerprints']
        else:
            fingerprints = data
        
        episodes = set()
        for ep_list in fingerprints.values():
            if isinstance(ep_list, list):
                for entry in ep_list:
                    if isinstance(entry, tuple) and len(entry) >= 1:
                        ep_id = entry[0]  # First element is always episode_id
                        episodes.add(ep_id)
                    elif isinstance(entry, str):
                        episodes.add(entry)
        
        seasons = Counter()
        for ep in episodes:
            if ep.startswith('S') and len(ep) >= 6:
                try:
                    season = int(ep[1:3])
                    seasons[season] += 1
                    audio_seasons.add(season)
                except:
                    pass
        
        output.append(f"\nTotal episodes: {len(episodes)}")
        output.append("\nBy Season:")
        for s in sorted(seasons.keys()):
            count = sum(1 for ep in episodes if ep.startswith(f'S{s:02d}'))
            output.append(f"  Season {s:02d}: {count} episodes")
    except Exception as e:
        output.append(f"\n⚠️  Error reading audio database: {e}")
else:
    output.append("\n❌ Audio database not found")

output.append("\n" + "=" * 60)
output.append("NEXT SEASON TO START")
output.append("=" * 60)

# Determine next season based on video database (primary)
all_seasons = video_seasons.union(audio_seasons)
if all_seasons:
    last_season = max(all_seasons)
    next_season = last_season + 1
    output.append(f"\n🎯 Next season to start: Season {next_season:02d}")
    output.append(f"\nLast completed season: Season {last_season:02d}")
else:
    output.append(f"\n🎯 Next season to start: Season 01")
    output.append(f"\nNo seasons have been ingested yet.")

output.append("\n" + "=" * 60)

# Write to file and print
result = "\n".join(output)
with open("/tmp/db_status.txt", "w") as f:
    f.write(result)
print(result)


