# Address-Based Pool Filtering Implementation

## Overview

This document describes the implementation of address-based pool filtering for the StablecoinPoolOpt system. The new approach uses on-chain token addresses to make pool filtering more deterministic and predictable, replacing the previous symbol-based matching approach.

## Architecture

### Database Schema Changes

#### 1. Approved Tokens Table Enhancement
- **New Column**: `token_address` (TEXT)
- **Purpose**: Stores the on-chain address for each approved token
- **Migration**: V21__add_token_address_to_approved_tokens.sql

#### 2. Pools Table Enhancement
- **New Columns**: 
  - `underlying_tokens` (JSONB)
  - `underlying_token_addresses` (JSONB)
- **Purpose**: 
  - `underlying_tokens`: Array of token symbols for approved pools
  - `underlying_token_addresses`: Array of on-chain addresses from DefiLlama
- **Migration**: V22__add_underlying_tokens_to_pools.sql

## Data Flow

### 1. Data Ingestion (fetch_defillama_pools.py)
- Fetches pool data from DefiLlama API
- Extracts `underlyingTokens` array from API response
- Stores both token addresses and symbols in the pools table
- Populates `underlying_token_addresses` with raw address data from API

### 2. Pool Filtering (filter_pools_pre.py)
- **Address-Based Validation**: 
  - Compares pool's `underlying_token_addresses` against approved token addresses
  - Only pools where ALL underlying tokens have approved addresses are allowed
- **Symbol Resolution**: 
  - Maps addresses back to approved token symbols
  - Populates `underlying_tokens` with resolved symbols for approved pools
- **Deterministic Results**: Eliminates ambiguity in token matching

### 3. Asset Allocation (optimize_allocations.py)
- **Token Universe Building**:
  - Uses `underlying_tokens` from pools table (populated by filter_pools_pre)
  - No fallback needed - underlying_tokens should always be available
- **Pool Data Query**: 
  - Joins with pools table to include `underlying_tokens` column
  - Ensures allocation uses verified token symbols

### 4. Data Quality Reporting (data_quality_report.py)
- **Enhanced Analysis**: 
  - Validates presence of `underlying_tokens` in pool data
  - Reports on address-based filtering effectiveness
  - Monitors data quality across the new pipeline

## Key Benefits

### 1. Deterministic Filtering
- **Before**: Symbol matching could be ambiguous (e.g., "USDC" vs "USDC.e")
- **After**: Address matching is exact and unambiguous
- **Result**: More predictable pool inclusion/exclusion

### 2. Improved Data Quality
- **Source of Truth**: On-chain addresses from DefiLlama
- **Verification**: All tokens in approved pools must have known addresses
- **Traceability**: Clear mapping from address → symbol for approved tokens

### 3. Better Token Universe Management
- **Accurate Composition**: Only verified tokens included in optimization
- **Consistent Naming**: Standardized symbols from approved tokens table
- **Reduced Errors**: No more partial symbol matches or false positives

## Implementation Details

### Address Resolution Logic
```python
# For each pool's underlying token addresses
for address in pool_underlying_addresses:
    # Find matching approved token
    approved_token = query_approved_token_by_address(address)
    if not approved_token:
        # Pool filtered out - unapproved token
        return False
    # Add to resolved symbols list
    resolved_symbols.append(approved_token.symbol)
```



### Error Handling
- **Missing Addresses**: Pools without `underlying_token_addresses` are filtered out
- **Invalid JSON**: Graceful handling of malformed token arrays
- **Unknown Tokens**: Clear logging of unapproved token addresses

## Testing

### Test Coverage
1. **Database Schema**: Verify new columns exist and are populated
2. **Filtering Logic**: Test address-based validation with known pools
3. **Allocation Pipeline**: Ensure token universe uses resolved symbols
4. **Data Quality**: Validate reporting accuracy

### Test Results
- ✅ Database schema migrations applied successfully
- ✅ Address-based filtering working correctly
- ✅ Token universe building using resolved symbols
- ✅ Data quality reporting enhanced

## Migration Strategy

### Phase 1: Schema Updates
1. Apply V21 migration (token_address column)
2. Apply V22 migration (underlying_tokens columns)
3. Populate token addresses for approved tokens

### Phase 2: Data Pipeline Updates
1. Update data ingestion to capture token addresses
2. Modify filtering logic to use address validation
3. Update allocation to use resolved symbols

### Phase 3: Validation
1. Run comprehensive tests
2. Verify filtering results match expectations
3. Monitor data quality metrics

## Future Enhancements

### 1. Automated Address Discovery
- Query blockchain for token metadata
- Auto-populate token addresses for new tokens
- Reduce manual address management

### 2. Multi-Chain Support
- Extend address validation to other chains
- Chain-specific address formats
- Cross-chain token mapping

### 3. Real-time Validation
- Monitor on-chain for token contract changes
- Automatic updates for token metadata
- Enhanced security monitoring

## Troubleshooting

### Common Issues

#### 1. Pools Filtered Out Unexpectedly
- **Cause**: Missing token addresses in approved_tokens table
- **Solution**: Verify all pool tokens have corresponding addresses
- **Check**: `SELECT * FROM approved_tokens WHERE token_address IS NULL`

#### 2. Empty underlying_tokens Arrays
- **Cause**: Data ingestion issues or API changes
- **Solution**: Re-run data ingestion with fresh data
- **Check**: `SELECT pool_id, underlying_token_addresses FROM pools WHERE underlying_tokens = '[]'`

#### 3. Symbol Mismatches
- **Cause**: Address-to-symbol mapping errors
- **Solution**: Verify approved token addresses are correct
- **Check**: `SELECT token_symbol, token_address FROM approved_tokens`

### Performance Considerations

#### 1. Index Optimization
- Add index on `approved_tokens.token_address`
- Add index on `pools.underlying_token_addresses`
- Improves JOIN performance for address lookups

#### 2. Query Optimization
- Use JSONB operators for address array containment
- Batch address resolution queries
- Cache frequently accessed token mappings

## Conclusion

The address-based filtering implementation provides a more robust, deterministic approach to pool filtering and token management. By leveraging on-chain addresses as the source of truth, the system eliminates ambiguity in token identification and improves the overall reliability of the asset allocation pipeline.

The implementation maintains backward compatibility through fallback mechanisms while providing clear migration paths for future enhancements. Comprehensive testing ensures the new approach works correctly across all components of the system.