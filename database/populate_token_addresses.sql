-- Populate approved_tokens table with token addresses
-- This query updates existing records or inserts new ones with the provided token addresses

-- Update USDT
UPDATE approved_tokens 
SET token_address = '0xdAC17F958D2ee523a2206206994597C13D831ec7' 
WHERE token_symbol = 'USDT';

-- Insert USDT if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDT', '0xdAC17F958D2ee523a2206206994597C13D831ec7'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDT');

-- Update USDF
UPDATE approved_tokens 
SET token_address = '0xFa2B947eEc368f42195f24F36d2aF29f7c24CeC2' 
WHERE token_symbol = 'USDF';

-- Insert USDF if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDF', '0xFa2B947eEc368f42195f24F36d2aF29f7c24CeC2'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDF');

-- Update crvUSD
UPDATE approved_tokens 
SET token_address = '0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E' 
WHERE token_symbol = 'crvUSD';

-- Insert crvUSD if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'crvUSD', '0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'crvUSD');

-- Update USTBL
UPDATE approved_tokens 
SET token_address = '0xe4880249745eAc5F1eD9d8F7DF844792D560e750' 
WHERE token_symbol = 'USTBL';

-- Insert USTBL if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USTBL', '0xe4880249745eAc5F1eD9d8F7DF844792D560e750'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USTBL');

-- Update TBILL
UPDATE approved_tokens 
SET token_address = '0xdd50C053C096CB04A3e3362E2b622529EC5f2e8a' 
WHERE token_symbol = 'TBILL';

-- Insert TBILL if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'TBILL', '0xdd50C053C096CB04A3e3362E2b622529EC5f2e8a'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'TBILL');

-- Update OUSG
UPDATE approved_tokens 
SET token_address = '0x1B19C19393e2d034D8Ff31ff34c81252FcBbee92' 
WHERE token_symbol = 'OUSG';

-- Insert OUSG if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'OUSG', '0x1B19C19393e2d034D8Ff31ff34c81252FcBbee92'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'OUSG');

-- Update USCC
UPDATE approved_tokens 
SET token_address = '0x14d60E7FDC0D71d8611742720E4C50E7a974020c' 
WHERE token_symbol = 'USCC';

-- Insert USCC if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USCC', '0x14d60E7FDC0D71d8611742720E4C50E7a974020c'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USCC');

-- Update USTB
UPDATE approved_tokens 
SET token_address = '0x43415eB6ff9DB7E26A15b704e7A3eDCe97d31C4e' 
WHERE token_symbol = 'USTB';

-- Insert USTB if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USTB', '0x43415eB6ff9DB7E26A15b704e7A3eDCe97d31C4e'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USTB');

-- Update USDY
UPDATE approved_tokens 
SET token_address = '0x96F6eF951840721AdBF46Ac996b59E0235CB985C' 
WHERE token_symbol = 'USDY';

-- Insert USDY if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDY', '0x96F6eF951840721AdBF46Ac996b59E0235CB985C'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDY');

-- Update BUIDL
UPDATE approved_tokens 
SET token_address = '0x7712c34205737192402172409a8F7ccef8aA2AEc' 
WHERE token_symbol = 'BUIDL';

-- Insert BUIDL if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'BUIDL', '0x7712c34205737192402172409a8F7ccef8aA2AEc'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'BUIDL');

-- Update USYC
UPDATE approved_tokens 
SET token_address = '0x136471a34f6ef19fE571EFFC1CA711fdb8E49f2b' 
WHERE token_symbol = 'USYC';

-- Insert USYC if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USYC', '0x136471a34f6ef19fE571EFFC1CA711fdb8E49f2b'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USYC');

-- Update USDe
UPDATE approved_tokens 
SET token_address = '0x4c9EDD5852cd905f086C759E8383e09bff1E68B3' 
WHERE token_symbol = 'USDe';

-- Insert USDe if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDe', '0x4c9EDD5852cd905f086C759E8383e09bff1E68B3'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDe');

-- Update USDM
UPDATE approved_tokens 
SET token_address = '0x59D9356E565Ab3A36dD77763Fc0d87fEaf85508C' 
WHERE token_symbol = 'USDM';

-- Insert USDM if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDM', '0x59D9356E565Ab3A36dD77763Fc0d87fEaf85508C'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDM');

-- Update USDL
UPDATE approved_tokens 
SET token_address = '0xbdC7c08592Ee4aa51D06C27Ee23D5087D65aDbcD' 
WHERE token_symbol = 'USDL';

-- Insert USDL if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDL', '0xbdC7c08592Ee4aa51D06C27Ee23D5087D65aDbcD'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDL');

-- Update USDG
UPDATE approved_tokens 
SET token_address = '0xe343167631d89B6Ffc58B88d6b7fB0228795491D' 
WHERE token_symbol = 'USDG';

-- Insert USDG if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDG', '0xe343167631d89B6Ffc58B88d6b7fB0228795491D'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDG');

-- Update USD0
UPDATE approved_tokens 
SET token_address = '0x73A15FeD60Bf67631dC6cd7Bc5B6e8da8190aCF5' 
WHERE token_symbol = 'USD0';

-- Insert USD0 if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USD0', '0x73A15FeD60Bf67631dC6cd7Bc5B6e8da8190aCF5'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USD0');

-- Update USDtb
UPDATE approved_tokens 
SET token_address = '0xC139190F447e929f090Edeb554D95AbB8b18aC1C' 
WHERE token_symbol = 'USDtb';

-- Insert USDtb if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDtb', '0xC139190F447e929f090Edeb554D95AbB8b18aC1C'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDtb');

-- Update RLUSD
UPDATE approved_tokens 
SET token_address = '0x8292Bb45bf1Ee4d140127049757C2E0fF06317eD' 
WHERE token_symbol = 'RLUSD';

-- Insert RLUSD if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'RLUSD', '0x8292Bb45bf1Ee4d140127049757C2E0fF06317eD'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'RLUSD');

-- Update GHO
UPDATE approved_tokens 
SET token_address = '0x40D16FC0246aD3160Ccc09B8D0D3A2cD28aE6C2f' 
WHERE token_symbol = 'GHO';

-- Insert GHO if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'GHO', '0x40D16FC0246aD3160Ccc09B8D0D3A2cD28aE6C2f'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'GHO');

-- Update GUSD
UPDATE approved_tokens 
SET token_address = '0x056Fd409E1d7A124BD7017459dFEa2F387b6d5Cd' 
WHERE token_symbol = 'GUSD';

-- Insert GUSD if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'GUSD', '0x056Fd409E1d7A124BD7017459dFEa2F387b6d5Cd'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'GUSD');

-- Update USDP
UPDATE approved_tokens 
SET token_address = '0x8E870D67F660D95d5be530380D0eC0bd388289E1' 
WHERE token_symbol = 'USDP';

-- Insert USDP if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDP', '0x8E870D67F660D95d5be530380D0eC0bd388289E1'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDP');

-- Update PYUSD
UPDATE approved_tokens 
SET token_address = '0x6c3ea9036406852006290770BEdFcAbA0e23A0e8' 
WHERE token_symbol = 'PYUSD';

-- Insert PYUSD if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'PYUSD', '0x6c3ea9036406852006290770BEdFcAbA0e23A0e8'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'PYUSD');

-- Update BOLD
UPDATE approved_tokens 
SET token_address = '0x6440f144b7e50D6a8439336510312d2F54beB01D' 
WHERE token_symbol = 'BOLD';

-- Insert BOLD if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'BOLD', '0x6440f144b7e50D6a8439336510312d2F54beB01D'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'BOLD');

-- Update LUSD
UPDATE approved_tokens 
SET token_address = '0x5f98805A4E8be255a32880FDeC7F6728C6568bA0' 
WHERE token_symbol = 'LUSD';

-- Insert LUSD if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'LUSD', '0x5f98805A4E8be255a32880FDeC7F6728C6568bA0'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'LUSD');

-- Update DAI
UPDATE approved_tokens 
SET token_address = '0x6B175474E89094C44Da98b954EedeAC495271d0F' 
WHERE token_symbol = 'DAI';

-- Insert DAI if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'DAI', '0x6B175474E89094C44Da98b954EedeAC495271d0F'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'DAI');

-- Update USDS
UPDATE approved_tokens 
SET token_address = '0xdC035D45d973E3EC169d2276DDab16f1e407384F' 
WHERE token_symbol = 'USDS';

-- Insert USDS if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDS', '0xdC035D45d973E3EC169d2276DDab16f1e407384F'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDS');

-- Update USDC
UPDATE approved_tokens 
SET token_address = '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48' 
WHERE token_symbol = 'USDC';

-- Insert USDC if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'USDC', '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'USDC');

-- Update sBOLD
UPDATE approved_tokens 
SET token_address = '0x50bd66d59911f5e086ec87ae43c811e0d059dd11' 
WHERE token_symbol = 'sBOLD';

-- Insert sBOLD if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'sBOLD', '0x50bd66d59911f5e086ec87ae43c811e0d059dd11'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'sBOLD');

-- Update SUSDE
UPDATE approved_tokens 
SET token_address = '0x9d39a5de30e57443bff2a8307a4256c8797a3497' 
WHERE token_symbol = 'SUSDE';

-- Insert SUSDE if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'SUSDE', '0x9d39a5de30e57443bff2a8307a4256c8797a3497'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'SUSDE');

-- Update sDAI
UPDATE approved_tokens 
SET token_address = '0x83F20F44975D03b1b09e64809B757c47f942BEeA' 
WHERE token_symbol = 'sDAI';

-- Insert sDAI if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'sDAI', '0x83F20F44975D03b1b09e64809B757c47f942BEeA'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'sDAI');

-- Update SUSDS
UPDATE approved_tokens 
SET token_address = '0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD' 
WHERE token_symbol = 'SUSDS';

-- Insert SUSDS if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'SUSDS', '0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'SUSDS');

-- Update STKGHO
UPDATE approved_tokens 
SET token_address = '0x1a88Df1cFe15Af22B3c4c783D4e6F7F9e0C1885d' 
WHERE token_symbol = 'STKGHO';

-- Insert STKGHO if it doesn't exist
INSERT INTO approved_tokens (token_symbol, token_address)
SELECT 'STKGHO', '0x1a88Df1cFe15Af22B3c4c783D4e6F7F9e0C1885d'
WHERE NOT EXISTS (SELECT 1 FROM approved_tokens WHERE token_symbol = 'STKGHO');


-- Verify the results
SELECT token_symbol, token_address, added_timestamp 
FROM approved_tokens 
WHERE token_address IS NOT NULL 
ORDER BY token_symbol;