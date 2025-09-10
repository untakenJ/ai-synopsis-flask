#!/usr/bin/env python3
"""
Single cache refresh script for AI Synopsis Flask application.
This script runs a single analysis cycle and saves the results to cache.
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function to run single cache refresh"""
    try:
        # Import the analyze_all function and cache utilities
        from app import analyze_all, save_cache_to_s3
        
        async def run_single_refresh():
            """Run a single analysis cycle"""
            try:
                print("Starting analyze_all...")
                results = await analyze_all()
                
                if results:
                    print(f"Analysis completed successfully. Found {len(results)} news items.")
                    
                    # Save results to cache
                    execution_time = datetime.now().isoformat()
                    save_cache_to_s3(results, execution_time)
                    print("Cache saved successfully to S3.")
                    
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
                    
                    return True
                else:
                    print("No results found.")
                    return False
                    
            except Exception as e:
                print(f"Error during analysis: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Run the analysis
        success = asyncio.run(run_single_refresh())
        
        if success:
            print("Single refresh operation completed successfully.")
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
