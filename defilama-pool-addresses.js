// Collect pool IDs and Ethereum addresses while scrolling (handles virtual scrolling)
(async () => {
    console.log('🔄 Starting to scroll and collect all pools...');

    const collectedPools = new Map(); // Map: poolId -> ethereumAddress
    let noChangeCount = 0;
    let previousSize = 0;

    // Function to extract Ethereum address from URL
    function extractEthereumAddress(url) {
        if (!url) return 'NOT_FOUND';
        // Match 0x followed by 40 hexadecimal characters (total 42 chars)
        const match = url.match(/0x[a-fA-F0-9]{40}/);
        return match ? match[0] : 'NOT_FOUND';
    }

    while (true) {
        // Find all pool links first
        const poolLinks = document.querySelectorAll('a[href*="/yields/pool/"]');

        poolLinks.forEach(poolLink => {
            const match = poolLink.href.match(/\/yields\/pool\/([a-f0-9\-]+)/);
            if (!match) return;

            const poolId = match[1];

            // Row container (go up a bit to the row element)
            let row = poolLink.closest('tr') || poolLink.parentElement;
            if (!row) {
                if (!collectedPools.has(poolId)) {
                    collectedPools.set(poolId, 'NOT_FOUND');
                }
                return;
            }

            // Find external link in the same row:
            // usually an <a> with href not containing "/yields/pool/"
            const allLinksInRow = row.querySelectorAll('a[href]');
            let ethereumAddress = null;
            for (const a of allLinksInRow) {
                const href = a.getAttribute('href') || '';
                if (!href.includes('/yields/pool/')) {
                    ethereumAddress = extractEthereumAddress(a.href);
                    break;
                }
            }

            if (ethereumAddress && ethereumAddress !== 'NOT_FOUND') {
                collectedPools.set(poolId, ethereumAddress);
            } else if (!collectedPools.has(poolId)) {
                collectedPools.set(poolId, '');
            }
        });

        const currentSize = collectedPools.size;
        console.log(`Collected ${currentSize} unique pools so far...`);

        if (currentSize === previousSize) {
            noChangeCount++;
            if (noChangeCount >= 5) {
                console.log('✅ No new pools found for 5 checks - done!');
                break;
            }
        } else {
            noChangeCount = 0;
        }

        previousSize = currentSize;

        // Scroll down
        window.scrollBy(0, window.innerHeight);
        await new Promise(resolve => setTimeout(resolve, 500));

        // Safety limit
        if (currentSize > 600) {
            console.log('⚠️ Reached safety limit of 600 pools');
            break;
        }
    }

    // Convert to CSV format
    const csvRows = ['Pool ID,Ethereum Address']; // Header
    collectedPools.forEach((ethereumAddress, poolId) => {
        csvRows.push(`${poolId},${ethereumAddress}`);
    });

    const csvText = csvRows.join('\n');

    console.log(`\n📊 RESULTS:`);
    console.log(`Total unique pools found: ${collectedPools.size}`);
    console.log('\n📋 CSV Data (first 5 rows):');
    console.log(csvRows.slice(0, 6).join('\n'));

    // Copy to clipboard
    try {
        await navigator.clipboard.writeText(csvText);
        console.log('\n✅ CSV data copied to clipboard!');
    } catch (err) {
        console.log('\n⚠️ Could not auto-copy. Here is your CSV data:');
        console.log(csvText);
    }

    // Make available globally
    window.poolData = Array.from(collectedPools.entries()).map(([id, address]) => ({
        poolId: id,
        ethereumAddress: address
    }));
    console.log('\n💾 Available as: window.poolData (array of objects)');
})();