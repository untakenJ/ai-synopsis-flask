"""
Test script for image generation fallback functionality.
This script can be run standalone without starting the Flask server.

Usage:
    python test/test_image_fallback.py
    or from project root: python -m test.test_image_fallback

Requirements:
    - OPENAI_API_KEY must be set in .env file or environment variables
    - Required packages: python-dotenv, openai, asyncio (built-in)

Test Modes:
    1. Normal test: Tests both normal image generation and fallback mechanism
    2. Force fallback test: Tests only the fallback mechanism (bypasses normal generation)

Input:
    - Title: Enter custom title or press Enter to use default
    - Summary: Only asked if custom title is provided (default summary used with default title)
    - Save path: Enter custom path or press Enter to use default (default: test/output/<sanitized_title>_<timestamp>.png)

Output:
    - Generated images are saved to the specified location (default: test/output/ directory)
    - Images are saved as PNG files

The fallback mechanism will trigger when:
    - Normal image generation fails 3 times
    - Or when max_retries=0 is set (for direct fallback testing)

Fallback generates a simple text-only image with:
    - White background
    - Blue cartoon-style text
    - News headline text only (no graphics)
"""
from dotenv import load_dotenv
import asyncio
import os
import sys
import base64
from datetime import datetime

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from app.py
from app import generate_image_with_dalle, generate_image_prompt

# Load environment variables
load_dotenv()

# Get OpenAI API key for validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default values for testing
DEFAULT_TITLE = "Bianca Censori: Home Is Where The Heat Is!!!🔥"
DEFAULT_SUMMARY = """Bianca Censori made a headline-grabbing appearance in Melbourne, Australia, on November 3, 2025, wearing a revealing silver bodysuit while visiting family, as reported by multiple news outlets. This event underscores her signature bold fashion and continued presence in the celebrity spotlight, drawing widespread attention and discussion.

On Monday, November 3, 2025, Censori was photographed in the suburbs of Melbourne accompanied by her mother, Alexandra, and sister, Angelina. The outing marked a rare public visit to her home country, attracting immediate interest from local onlookers and international media. Her presence in Australia without her husband, Kanye West, added to the ongoing public curiosity about her personal life and movements.

Censori's attire featured a skintight, strapless silver bodysuit that highlighted her curves, paired with matching stockings, knee-high socks, and high heels. This ensemble is consistent with her recent fashion choices, which often push boundaries and spark conversations about style and self-expression. Her daring outfits have become a trademark, frequently compared to aesthetics in the celebrity world, though she maintains a distinct identity.

The visit to Australia comes amid Censori's split time between Los Angeles, where she resides in a multi-million dollar home with West, and her native country. While West was not present during this trip, the couple's relationship has been a subject of speculation, though no recent confirmations have been made. Censori's focus on family during this outing highlights a personal side often overshadowed by her public image.

News outlets like TMZ and Yahoo covered the event extensively, with photos and descriptions quickly circulating online. The coverage emphasized the viral nature of such celebrity sightings, reflecting the public's fascination with high-profile figures and their everyday activities. Social media reactions were mixed, with some users praising her confidence and others critiquing the outfit's appropriateness.

This appearance reinforces Censori's ability to command attention independently, separate from West's controversies. Her fashion statements have positioned her as a influential figure in entertainment culture, often driving trends and discussions. The event also illustrates how celebrity news blends personal narrative with broader cultural themes, engaging audiences worldwide.

Looking ahead, Censori's continued embrace of bold fashion may open doors to further opportunities in modeling, design, or media. As she navigates her career, such public moments will likely shape her legacy and influence, highlighting the evolving dynamics of fame in the digital age. The enduring interest in her actions suggests a sustained relevance in pop culture narratives."""

# Default save path (relative to test/ directory)
DEFAULT_SAVE_PATH = "output"


def save_image_from_base64(image_base64, save_path):
    """Save base64 encoded image to file"""
    try:
        # Get directory path
        dir_path = os.path.dirname(save_path)
        
        # Create directory if it doesn't exist (only if there's a directory component)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Write to file
        with open(save_path, 'wb') as f:
            f.write(image_data)
        
        print(f"✓ Image saved successfully to: {save_path}")
        print(f"  File size: {len(image_data)} bytes")
        return True
    except Exception as e:
        print(f"✗ Error saving image: {e}")
        return False


def generate_default_filename(title):
    """Generate a default filename based on title and timestamp"""
    # Get the directory where this test file is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Sanitize title for filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{timestamp}.png"
    
    # Return path relative to test directory
    return os.path.join(test_dir, DEFAULT_SAVE_PATH, filename)


async def test_image_generation():
    """Test image generation with fallback functionality"""
    print("="*60)
    print("Image Generation Fallback Test")
    print("="*60)
    print()
    
    # Get user input for title
    print(f"Enter news title (or press Enter to use default):")
    print(f"  Default: {DEFAULT_TITLE}")
    title = input().strip()
    use_default_title = not title
    
    if use_default_title:
        title = DEFAULT_TITLE
        summary = DEFAULT_SUMMARY
        print(f"\nUsing default title: {title}")
        print(f"Using default summary: {summary}")
    else:
        # Only ask for summary if user provided custom title
        print(f"\nEnter news summary (or press Enter to skip):")
        summary = input().strip()
    
    print("\n" + "="*60)
    print("Test Configuration:")
    print(f"  Title: {title}")
    print(f"  Summary: {summary if summary else '(empty)'}")
    print("="*60 + "\n")
    
    # Generate prompt
    prompt = generate_image_prompt(title, summary)
    print(f"Generated prompt:\n{prompt}\n")
    
    # Test mode selection
    print("Select test mode:")
    print("  1. Normal test (will attempt regular generation first, then fallback if needed)")
    print("  2. Force fallback test (skip normal generation, test fallback directly)")
    mode = input("Enter choice (1 or 2, default: 1): ").strip() or "1"
    
    # Get save path
    default_filename = generate_default_filename(title)
    print(f"\nEnter save path for image (or press Enter to use default):")
    print(f"  Default: {default_filename}")
    save_path = input().strip()
    if not save_path:
        save_path = default_filename
    
    if mode == "2":
        # Force fallback by using max_retries=0 for normal generation
        print("\n" + "="*60)
        print("FORCE FALLBACK MODE: Skipping normal generation")
        print("="*60 + "\n")
        result = await generate_image_with_dalle(prompt, max_retries=0, title=title)
    else:
        # Normal test
        print("\n" + "="*60)
        print("NORMAL TEST MODE: Will attempt regular generation first")
        print("="*60 + "\n")
        result = await generate_image_with_dalle(prompt, max_retries=3, title=title)
    
    # Display results
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    if result:
        print("✓ Image generation successful!")
        print(f"  Image data received (length: {len(result)} characters)")
        
        # Save image to file
        print(f"\nSaving image to: {save_path}")
        if save_image_from_base64(result, save_path):
            print(f"  Full path: {os.path.abspath(save_path)}")
        else:
            print("  Warning: Failed to save image, but image data is available in memory")
    else:
        print("✗ Image generation failed")
    print("="*60)


def main():
    """Main function to run tests"""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please set OPENAI_API_KEY in your .env file or environment")
        return
    
    asyncio.run(test_image_generation())


if __name__ == "__main__":
    main()

