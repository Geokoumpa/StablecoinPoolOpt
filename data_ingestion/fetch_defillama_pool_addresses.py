import asyncio
import logging
import re
from datetime import datetime, timezone

from playwright.async_api import async_playwright
from sqlalchemy import text
from database.db_utils import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def extract_pool_id(href):
    """
    Extract pool ID from DeFiLlama URL pattern.
    """
    if not href:
        return None
    
    # Match pattern /yields/pool/[pool-id]
    match = re.search(r'/yields/pool/([a-f0-9\-]+)', href)
    return match.group(1) if match else None

def extract_ethereum_address(url):
    """
    Extract Ethereum address from URL - matches original JavaScript logic.
    """
    if not url:
        return None
    
    # Match 0x followed by 40 hexadecimal characters (total 42 chars)
    match = re.search(r'0x[a-fA-F0-9]{40}', url)
    return match.group(0) if match else None

async def crawl_defillama_ethereum_pools():
    """
    Crawl all DeFiLlama Ethereum pools to extract pool IDs and Ethereum addresses.
    Simple approach: just get all Ethereum pools without filtering.
    """
    collected_pools = {}
    no_change_count = 0
    previous_size = 0
    
    logger.info("=== STARTING DEFILLAMA ETHEREUM POOL CRAWLING ===")
    
    # Simple URL for all Ethereum pools
    url = "https://defillama.com/yields/stablecoins?chain=Ethereum"
    logger.info(f"Target URL: {url}")
    
    async with async_playwright() as p:
        # Launch browser in headless mode with longer timeout
        logger.info("Launching browser...")
        browser = await p.chromium.launch(
            headless=True,
            timeout=60000  # 60 seconds timeout
        )
        try:
            # Create context with realistic user agent
            logger.info("Creating browser context...")
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            # Set longer timeout for navigation
            page.set_default_timeout(60000)
            
            # Navigate to URL
            logger.info("Navigating to DeFiLlama...")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                logger.info("Successfully navigated to DeFiLlama")
            except Exception as e:
                logger.error(f"Navigation timeout: {e}")
                return {}
            
            # Wait for page to fully load
            logger.info("Waiting for page to load...")
            await page.wait_for_timeout(5000)
            
            # Check if page loaded properly
            title = await page.title()
            logger.info(f"Page title: {title}")
            if "Stablecoins" not in title:
                logger.error(f"Page title unexpected: {title}")
                return {}
            
            # Main crawling loop - exactly like original JavaScript
            logger.info("Starting pool collection loop...")
            while True:
                # Find all pool links first
                pool_links = await page.query_selector_all('a[href*="/yields/pool/"]')
                
                logger.info(f"Found {len(pool_links)} pool links on current page")
                
                for i, link in enumerate(pool_links):
                    try:
                        href = await link.get_attribute('href')
                        pool_id = extract_pool_id(href)
                        
                        if not pool_id:
                            continue
                        
                        # Skip if we already have this pool
                        if pool_id in collected_pools:
                            continue
                        
                        # Extract address using exact JavaScript logic from original
                        ethereum_address = await link.evaluate('''
                            (poolLink) => {
                                // Extract pool ID (already done in Python, but keeping for consistency)
                                const match = poolLink.href.match(/\/yields\/pool\/([a-f0-9\-]+)/);
                                if (!match) return null;
                                const poolId = match[1];
                                
                                // Row container (go up a bit to row element) - exact from original
                                let row = poolLink.closest('tr') || poolLink.parentElement;
                                if (!row) {
                                    return null;
                                }
                                
                                // Find external link in same row - exact from original
                                const allLinksInRow = row.querySelectorAll('a[href]');
                                let ethereumAddress = null;
                                for (const a of allLinksInRow) {
                                    const href = a.getAttribute('href') || '';
                                    if (!href.includes('/yields/pool/')) {
                                        // Extract Ethereum address from href - exact from original
                                        const addressMatch = href.match(/0x[a-fA-F0-9]{40}/);
                                        if (addressMatch) {
                                            ethereumAddress = addressMatch[0];
                                            break;
                                        }
                                    }
                                }
                                
                                return ethereumAddress;
                            }
                        ''')
                        
                        if ethereum_address:
                            collected_pools[pool_id] = ethereum_address
                            if i < 5:  # Log first few for debugging
                                logger.info(f"✓ Pool {pool_id}: {ethereum_address}")
                        else:
                            collected_pools[pool_id] = None
                            if i < 5:  # Log first few for debugging
                                logger.info(f"✗ Pool {pool_id}: No address found")
                            
                    except Exception as e:
                        logger.warning(f"Error processing pool link {i}: {e}")
                        continue
                
                # Check if we found new pools - exact from original
                current_size = len(collected_pools)
                logger.info(f"Collected {current_size} unique pools so far...")
                
                if current_size == previous_size:
                    no_change_count += 1
                    logger.info(f"No new pools found (count: {no_change_count})")
                    if no_change_count >= 5:
                        logger.info("No new pools found for 5 checks - done!")
                        break
                else:
                    no_change_count = 0
                
                previous_size = current_size
                
                # Scroll down - exact from original
                await page.evaluate('window.scrollBy(0, window.innerHeight)')
                await page.wait_for_timeout(500)
                
                # Safety limit - increased to get all pools
                if current_size > 1400:  # Slightly above 1361 to be safe
                    logger.warning("Reached safety limit of 1400 pools")
                    break
            
            logger.info(f"Crawling completed. Found {len(collected_pools)} total pools.")
            
        finally:
            await browser.close()
            logger.info("Browser closed")
    
    return collected_pools

def update_pool_addresses(engine, pool_addresses):
    """
    Update pool addresses in the database.
    """
    if not pool_addresses:
        logger.warning("No pool addresses to update")
        return 0
    
    updated_count = 0
    try:
        logger.info(f"Updating {len(pool_addresses)} pool addresses in database...")
        with engine.connect() as conn:
            with conn.begin():
                for pool_id, address in pool_addresses.items():
                    if address:  # Only update if we have a valid address
                        update_query = text("""
                            UPDATE pools 
                            SET pool_address = :address 
                            WHERE pool_id = :pool_id
                        """)
                        result = conn.execute(update_query, {
                            "address": address,
                            "pool_id": pool_id
                        })
                        
                        if result.rowcount > 0:
                            updated_count += 1
                            logger.debug(f"Updated address for pool {pool_id}: {address}")
                        else:
                            logger.warning(f"Pool {pool_id} not found in database")
                
                logger.info(f"Successfully updated {updated_count} pool addresses")
                return updated_count
                
    except Exception as e:
        logger.error(f"Error updating pool addresses: {e}")
        return 0

async def fetch_defillama_pool_addresses():
    """
    Main function to fetch DeFiLlama pool addresses using web crawling.
    """
    engine = None
    try:
        # Get database connection
        logger.info("Establishing database connection...")
        engine = get_db_connection()
        if not engine:
            logger.error("Could not establish database connection. Exiting.")
            return
        
        logger.info("Starting DeFiLlama pool address crawler...")
        start_time = datetime.now(timezone.utc)
        
        # Crawl all DeFiLlama Ethereum pools
        pool_addresses = await crawl_defillama_ethereum_pools()
        
        # Filter out pools with no addresses
        valid_addresses = {k: v for k, v in pool_addresses.items() if v is not None}
        
        logger.info(f"Found {len(valid_addresses)} pools with valid addresses out of {len(pool_addresses)} total pools")
        
        # Update database
        updated_count = update_pool_addresses(engine, valid_addresses)
        
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📥 DEFILLAMA POOL ADDRESS CRAWLER SUMMARY")
        logger.info("="*60)
        logger.info(f"🌐 URL: https://defillama.com/yields/stablecoins?chain=Ethereum")
        logger.info(f"📊 Total pools crawled: {len(pool_addresses):,}")
        logger.info(f"✅ Pools with valid addresses: {len(valid_addresses):,}")
        logger.info(f"❌ Pools without addresses: {len(pool_addresses) - len(valid_addresses):,}")
        logger.info(f"💾 Database updates: {updated_count:,}")
        logger.info(f"⏱️  Duration: {duration}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in fetch_defillama_pool_addresses: {e}")
        raise
    finally:
        if engine:
            engine.dispose()
            logger.info("Database connection closed")

def main():
    """
    Entry point for the script.
    """
    logger.info("=== DEFI LLAMA POOL ADDRESS CRAWLER STARTING ===")
    asyncio.run(fetch_defillama_pool_addresses())

if __name__ == "__main__":
    main()