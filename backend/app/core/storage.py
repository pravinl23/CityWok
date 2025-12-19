"""
Object Storage Integration for Fingerprint Databases

Supports S3, Cloudflare R2, and Google Cloud Storage (S3-compatible APIs).
Handles database download, upload, and verification.
"""

import os
import tarfile
import hashlib
import json
from typing import Optional, Dict, Any
from pathlib import Path


class FingerprintStorage:
    """
    Manage fingerprint database downloads/uploads from object storage.

    Supports:
    - AWS S3
    - Cloudflare R2 (S3-compatible)
    - Google Cloud Storage (S3-compatible)
    """

    def __init__(self):
        """Initialize storage client with credentials from environment."""
        self.s3_bucket = os.getenv("S3_BUCKET", "citywok-audio-db")
        self.s3_prefix = os.getenv("S3_PREFIX", "fingerprints")
        self.db_version = os.getenv("DB_VERSION", "v1")

        # Configure S3 client (works with S3, R2, GCS)
        # For Cloudflare R2, set AWS_ENDPOINT_URL to your R2 endpoint
        endpoint_url = os.getenv("AWS_ENDPOINT_URL")

        try:
            import boto3
            self.s3 = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "auto")  # 'auto' for R2
            )
            self.available = True
            print(f"📦 S3 client initialized (bucket: {self.s3_bucket})")
            if endpoint_url:
                print(f"   Using custom endpoint: {endpoint_url}")
        except ImportError:
            print("⚠️  boto3 not installed - S3 features disabled")
            self.available = False
            self.s3 = None
        except Exception as e:
            print(f"⚠️  S3 client initialization failed: {e}")
            self.available = False
            self.s3 = None

    def download_season_db(
        self,
        season: int,
        output_dir: str,
        verify_checksum: bool = True
    ) -> bool:
        """
        Download a single season's database from S3.

        Args:
            season: Season number (1-20)
            output_dir: Local directory to extract database
            verify_checksum: Verify SHA256 checksum after download

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            print(f"❌ S3 client not available")
            return False

        s3_key = f"{self.s3_prefix}/{self.db_version}/season_{season:02d}.lmdb.tar.gz"
        local_tarball = Path(output_dir) / f"season_{season:02d}.lmdb.tar.gz"
        local_db_dir = Path(output_dir) / f"season_{season:02d}.lmdb"

        # Check if already exists
        if local_db_dir.exists() and any(local_db_dir.iterdir()):
            print(f"  ℹ️  Season {season:02d} already exists, skipping")
            return True

        try:
            print(f"  Downloading season_{season:02d}.lmdb.tar.gz...")

            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Download from S3
            self.s3.download_file(
                self.s3_bucket,
                s3_key,
                str(local_tarball)
            )

            file_size = local_tarball.stat().st_size
            print(f"    Downloaded {file_size / 1024 / 1024:.1f} MB")

            # Verify checksum if requested
            if verify_checksum:
                checksum_key = f"checksums/{self.db_version}/season_{season:02d}.sha256"
                try:
                    response = self.s3.get_object(Bucket=self.s3_bucket, Key=checksum_key)
                    expected_checksum = response['Body'].read().decode().strip()

                    print(f"    Verifying checksum...")
                    with open(local_tarball, 'rb') as f:
                        actual_checksum = hashlib.sha256(f.read()).hexdigest()

                    if actual_checksum != expected_checksum:
                        print(f"    ❌ Checksum mismatch!")
                        print(f"       Expected: {expected_checksum}")
                        print(f"       Actual: {actual_checksum}")
                        local_tarball.unlink()
                        return False

                    print(f"    ✓ Checksum verified")
                except Exception as e:
                    print(f"    ⚠️  Checksum verification failed: {e}")
                    # Continue anyway if checksum file doesn't exist

            # Extract tarball
            print(f"    Extracting...")
            with tarfile.open(local_tarball, 'r:gz') as tar:
                tar.extractall(output_dir)

            # Clean up tarball
            local_tarball.unlink()

            print(f"    ✓ Season {season:02d} ready")
            return True

        except Exception as e:
            print(f"    ❌ Error downloading season {season:02d}: {e}")
            # Clean up partial files
            if local_tarball.exists():
                local_tarball.unlink()
            return False

    def upload_season_db(
        self,
        season: int,
        db_path: str,
        create_checksum: bool = True
    ) -> bool:
        """
        Upload a season's database to S3 (for ingestion).

        Args:
            season: Season number
            db_path: Path to LMDB database directory
            create_checksum: Create and upload SHA256 checksum

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            print(f"❌ S3 client not available")
            return False

        tarball_path = f"/tmp/season_{season:02d}.lmdb.tar.gz"
        s3_key = f"{self.s3_prefix}/{self.db_version}/season_{season:02d}.lmdb.tar.gz"

        try:
            print(f"📦 Compressing season {season:02d}...")

            # Create tarball
            with tarfile.open(tarball_path, 'w:gz') as tar:
                tar.add(db_path, arcname=f"season_{season:02d}.lmdb")

            file_size = Path(tarball_path).stat().st_size
            print(f"   Compressed to {file_size / 1024 / 1024:.1f} MB")

            # Create checksum
            checksum = None
            if create_checksum:
                print(f"   Creating checksum...")
                with open(tarball_path, 'rb') as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                print(f"   SHA256: {checksum}")

            # Upload to S3
            print(f"   Uploading to s3://{self.s3_bucket}/{s3_key}...")
            self.s3.upload_file(tarball_path, self.s3_bucket, s3_key)

            # Upload checksum
            if checksum:
                checksum_key = f"checksums/{self.db_version}/season_{season:02d}.sha256"
                self.s3.put_object(
                    Bucket=self.s3_bucket,
                    Key=checksum_key,
                    Body=checksum.encode()
                )
                print(f"   ✓ Checksum uploaded")

            # Clean up
            os.unlink(tarball_path)

            print(f"   ✓ Season {season:02d} uploaded successfully")
            return True

        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            # Clean up
            if os.path.exists(tarball_path):
                os.unlink(tarball_path)
            return False

    def download_all_seasons(
        self,
        output_dir: str,
        seasons: range = range(1, 21),
        verify_checksums: bool = True
    ) -> int:
        """
        Download all season databases.

        Args:
            output_dir: Local directory to store databases
            seasons: Range of seasons to download
            verify_checksums: Verify checksums after download

        Returns:
            Number of successfully downloaded seasons
        """
        print(f"📦 Downloading fingerprint databases from S3")
        print(f"   Bucket: s3://{self.s3_bucket}/{self.s3_prefix}/{self.db_version}/")
        print(f"   Output: {output_dir}")
        print()

        successful = 0
        for season in seasons:
            if self.download_season_db(season, output_dir, verify_checksums):
                successful += 1

        print(f"\n✓ Downloaded {successful}/{len(list(seasons))} seasons")
        return successful

    def get_version_manifest(self) -> Optional[Dict[str, Any]]:
        """
        Download and parse version manifest from S3.

        Returns:
            Version manifest dictionary, or None if not found
        """
        if not self.available:
            return None

        try:
            manifest_key = f"{self.s3_prefix}/versions.json"
            response = self.s3.get_object(Bucket=self.s3_bucket, Key=manifest_key)
            manifest = json.loads(response['Body'].read().decode())
            return manifest
        except Exception as e:
            print(f"⚠️  Could not load version manifest: {e}")
            return None

    def list_available_seasons(self) -> list:
        """
        List available season databases in S3.

        Returns:
            List of season numbers
        """
        if not self.available:
            return []

        try:
            prefix = f"{self.s3_prefix}/{self.db_version}/"
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )

            seasons = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                # Extract season number from key like "season_01.lmdb.tar.gz"
                if 'season_' in key and '.tar.gz' in key:
                    parts = key.split('season_')[1].split('.')[0]
                    season_num = int(parts)
                    seasons.append(season_num)

            return sorted(seasons)

        except Exception as e:
            print(f"⚠️  Could not list seasons: {e}")
            return []


# Global storage instance
storage = FingerprintStorage()
