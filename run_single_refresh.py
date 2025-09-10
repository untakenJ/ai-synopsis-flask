#!/usr/bin/env python3
"""
Single cache refresh script for AI Synopsis Flask application.
This script runs a single analysis cycle and saves the results to cache.
It properly handles refresh status, locks, and can inherit running refresh operations.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function to run single cache refresh"""
    try:
        # Import required functions
        from app import (
            analyze_all, save_cache_to_s3, load_cache_from_s3, 
            get_refresh_status, acquire_refresh_lock, release_refresh_lock,
            save_refresh_status_to_s3, is_cache_valid_safe
        )
        
        def check_existing_refresh():
            """Check if there's already a refresh in progress"""
            refresh_status = get_refresh_status()
            if refresh_status["refresh_in_progress"]:
                print(f"Refresh already in progress (started at {refresh_status.get('start_time', 'unknown')})")
                print("Waiting for existing refresh to complete...")
                
                # Wait for existing refresh to complete
                timeout = 18000  # 5 hours timeout
                start_wait = time.time()
                
                while refresh_status["refresh_in_progress"]:
                    if time.time() - start_wait > timeout:
                        print("Timeout waiting for existing refresh to complete")
                        return None, "timeout"
                    
                    time.sleep(60)  # Wait 1 minute before checking again
                    refresh_status = get_refresh_status()
                    print(f"Still waiting... elapsed: {refresh_status.get('elapsed_seconds', 0):.0f}s")
                
                print("Existing refresh completed. Loading results...")
                
                # Load the results from the completed refresh
                cache_data = load_cache_from_s3()
                if cache_data and is_cache_valid_safe(cache_data):
                    return cache_data["data"], "inherited"
                else:
                    print("No valid cache found after existing refresh completed")
                    return None, "no_cache"
            
            return None, "none"
        
        async def run_single_refresh():
            """Run a single analysis cycle with proper status management"""
            try:
                # Check if there's already a refresh in progress
                existing_results, status = check_existing_refresh()
                
                if status == "inherited":
                    print("Successfully inherited results from existing refresh operation.")
                    return existing_results, True
                elif status == "timeout":
                    print("Timeout waiting for existing refresh. Starting new refresh...")
                elif status == "no_cache":
                    print("No valid cache after existing refresh. Starting new refresh...")
                else:
                    print("No existing refresh found. Starting new refresh...")
                
                # Try to acquire refresh lock
                print("Attempting to acquire refresh lock...")
                if not acquire_refresh_lock():
                    print("Failed to acquire refresh lock. Another process may have started refresh.")
                    print("Waiting for refresh to complete...")
                    
                    # Wait for the other process to complete
                    existing_results, status = check_existing_refresh()
                    if status == "inherited":
                        return existing_results, True
                    else:
                        print("Failed to get results from other process")
                        return None, False
                
                print("Successfully acquired refresh lock. Starting analysis...")
                
                try:
                    # Set refresh status to in progress
                    save_refresh_status_to_s3(True, datetime.now())
                    
                    print("Starting analyze_all...")
                    results = await analyze_all()
                    
                    if results:
                        print(f"Analysis completed successfully. Found {len(results)} news items.")
                        
                        # Save results to cache
                        execution_time = datetime.now().isoformat()
                        save_cache_to_s3(results, execution_time)
                        print("Cache saved successfully to S3.")
                        
                        # Set refresh status to completed
                        save_refresh_status_to_s3(False, None)
                        
                        return results, True
                    else:
                        print("No results found.")
                        save_refresh_status_to_s3(False, None)
                        return None, False
                        
                except Exception as e:
                    print(f"Error during analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    save_refresh_status_to_s3(False, None)
                    return None, False
                finally:
                    # Always release the lock
                    release_refresh_lock()
                    print("Released refresh lock.")
                    
            except Exception as e:
                print(f"Unexpected error during refresh: {e}")
                import traceback
                traceback.print_exc()
                return None, False
        
        # Run the analysis
        results, success = asyncio.run(run_single_refresh())
        
        if success and results:
            print("Single refresh operation completed successfully.")
            
            # Print summary
            today_news = [item for item in results if item.get("happened_today") == "yes"]
            print(f"Today news items: {len(today_news)}")
            
            # Print first 5 items
            print("\nTop 5 news items:")
            for i, item in enumerate(results[:5], 1):
                title = item.get("title", "No title")
                summary = item.get("summary", "No summary")
                happened_today = item.get("happened_today", "unknown")
                print(f"{i}. {title}")
                print(f"   Summary: {summary[:100]}...")
                print(f"   Happened today: {happened_today}")
                print()
            
            sys.exit(0)
        else:
            print("Single refresh operation failed.")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed and environment variables are set.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
