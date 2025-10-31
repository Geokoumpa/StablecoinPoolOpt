# Stablecoin Pool Optimization Admin UI

## Overview

This document outlines requirements and specifications for admin web interface of the Stablecoin Pool Optimization system. The UI provides administrators with tools to monitor, configure, and manage the automated stablecoin yield optimization pipeline.

## Technical Architecture

### Frontend Framework
- **Remix**: Full-stack React framework with server-side rendering
  - Version: Latest stable (2.x)
  - TypeScript for type safety
  - File-based routing structure

### Database Layer
- **Prisma ORM**: Type-safe database access
  - Introspect existing PostgreSQL database to generate Prisma models
  - Database connection pooling for performance
  - Migrations handled separately through existing pipeline

### Styling & UI Components
- **Tailwind CSS**: Utility-first CSS framework
  - Responsive design with mobile-first approach
  - Custom design system tokens
- **Shadcn/ui**: Pre-built React components
  - Consistent design language
  - Accessible components out of the box
  - Theme customization for brand alignment

### Authentication & Authorization
- **Clerk**: User authentication and session management
  - Existing Clerk project integration
  - Simple authentication (all users are admins)
  - Session persistence and automatic token refresh

### Deployment & Infrastructure
- **GCP Cloud Run**: Serverless container deployment
  - Auto-scaling based on traffic
  - Environment-specific configurations
  - CI/CD pipeline integration
- **GCP Cloud SQL**: Existing PostgreSQL database
  - Connection through VPC for security
  - Read replicas for query performance

## Application Structure

```
app/
├── routes/
│   ├── _index.tsx              # Dashboard
│   ├── login.tsx               # Authentication
│   ├── optimization/
│   │   ├── _index.tsx          # Optimization runs list
│   │   └── $runId.tsx         # Single optimization run details
│   ├── pools/
│   │   ├── _index.tsx          # Pools list
│   │   ├── metrics.tsx         # Pool metrics with filtering
│   │   └── $poolId.tsx        # Single pool details
│   ├── protocols/
│   │   └── _index.tsx          # Protocols & tokens management
│   └── config/
│       └── _index.tsx          # Configuration management
├── components/
│   ├── ui/                     # Shadcn components
│   ├── charts/                 # Reusable chart components
│   ├── forms/                  # Form components
│   └── layout/                 # Layout components
├── lib/
│   ├── db.ts                   # Prisma client
│   ├── auth.ts                 # Clerk integration
│   └── utils.ts                # Utility functions
└── styles/
    └── globals.css             # Global styles
```

## Feature Specifications

### 1. Authentication System

#### Login Page (`/login`)
- Clerk authentication widgets
- Redirect to dashboard on successful login
- Error handling for authentication failures
- Password reset flow

#### Authorization
- Simple authentication (all users are admins)
- Route protection middleware
- Session timeout handling

### 2. Dashboard (`/`)

#### Overview Section
**Metrics Display:**
- Total Assets Under Management (AUM)
- Projected Daily Yield (USD and percentage)
- Current Allocation Count
- Last Optimization Run Status

**Latest Allocations:**
- Top 5 allocations by percentage
- Pool name, protocol, APY, allocated amount
- Visual representation (pie chart or bar chart)
- Link to full optimization run details

**Navigation Cards:**
- Quick access to main sections
- Status indicators for system health
- Recent activity feed

### 3. Optimization History (`/optimization`)

#### Optimization Runs List (`/optimization`)
**Table Display (Paginated, 50 per page):**
- Run ID
- Timestamp
- Projected APY
- Transaction Costs (USD)
- Status (Success/Failed/In Progress)
- Actions (View Details)

**Features:**
- Date range filtering
- Status filtering
- Search functionality
- Export to CSV

#### Single Optimization Run (`/optimization/$runId`)
**Summary Card:**
- Run parameters snapshot
- Total AUM
- Expected yield
- Gas costs
- Execution time

**Allocation Details:**
- List of all pools in allocation
- Amount and percentage allocated
- Expected APY for each pool
- Transaction sequence (if available)

**Transaction Sequence:**
- Step-by-step transaction list
- From/To assets
- Gas costs per transaction
- Transaction status

### 4. Pool Management (`/pools`)

#### Pools List (`/pools`)
**Table Display:**
- Pool ID
- Name
- Protocol
- Chain
- Current TVL
- Current APY
- Status (Active/Inactive)
- Actions (Edit, View Details)

**Features:**
- Protocol filtering
- Search by name or ID
- Status filtering
- Pagination (50 per page)

**Edit Modal:**
- Update pool address
- Deactivate/reactivate pool
- Save with validation

#### Pool Metrics (`/pools/metrics`)
**Advanced Filtering:**
- Date range selector
- APY range slider
- TVL range slider
- Filter status toggle
- Protocol multi-select

**Table Display:**
- Pool basic information
- Actual and forecasted APY
- Actual and forecasted TVL
- Filter status and reason
- Date of metrics

**Features:**
- Sortable columns
- Export filtered data
- Save filter presets

#### Single Pool Details (`/pools/$poolId`)
**Metrics Summary Card:**
- Current pool metrics
- 24h changes
- Pool grouping information

**Historical Charts:**
- TVL over time (actual vs forecasted)
- APY over time (actual vs forecasted)
- Volume indicators
- Date range selector

**Pool Information:**
- Underlying tokens
- Contract addresses
- Protocol information
- Associated optimization runs

### 5. Protocols & Tokens Management (`/protocols`)

#### Approved Protocols Section
**List Display:**
- Protocol name
- Added date
- Status (Active/Inactive)
- Pool count
- Actions (Edit, Delete)

**Features:**
- Client-side search
- Status filtering
- Sort by name/date

**Add/Edit Modal:**
- Protocol name input
- Status toggle
- Save with validation

#### Approved Tokens Section
**List Display:**
- Token symbol
- Token address
- Added date
- Status (Active/Inactive)
- Actions (Edit, Delete)

**Features:**
- Client-side search
- Status filtering
- Copy address to clipboard

**Add/Edit Modal:**
- Token symbol input
- Token address input with validation
- Status toggle

#### Blacklisted Tokens Section
**List Display:**
- Token symbol
- Added date
- Reason for blacklisting
- Status (Active/Inactive)
- Actions (Edit, Delete)

**Features:**
- Client-side search
- Status filtering
- Reason display

**Add/Edit Modal:**
- Token symbol input
- Reason dropdown/text
- Status toggle

### 6. Configuration Management (`/config`)

#### Pool Filtering Parameters
**Form Fields:**
- TVL limits (min/max)
- APY thresholds
- Volume requirements
- Risk parameters
- Save/Reset buttons

#### Allocation Parameters
**Form Fields:**
- TVL limit percentage
- Maximum allocation percentage
- Conversion rate
- Minimum pools requirement
- Profit optimization toggle
- Group allocation limits
- Position limits
- Save/Reset buttons

#### Wallets & Addresses
**Form Fields:**
- Main asset holding address
- Warm wallet address
- Backup addresses
- Address validation
- Save/Reset buttons

## Database Schema Requirements

### Existing Tables to be Used:
- `pools` - Pool information and metadata
- `pool_daily_metrics` - Daily pool metrics and forecasts
- `approved_protocols` - Approved protocol list
- `approved_tokens` - Approved token list
- `blacklisted_tokens` - Blacklisted token list
- `allocation_parameters` - Optimization run parameters
- `asset_allocations` - Allocation results
- `daily_balances` - Daily balance information

### Schema Modifications Required:

#### New Default Configuration Table
```sql
CREATE TABLE default_allocation_parameters (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(255) UNIQUE NOT NULL,
    parameter_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert default values
INSERT INTO default_allocation_parameters (parameter_name, parameter_value, description) VALUES
('tvl_limit_percentage', '0.05', 'Maximum percentage of pool TVL that can be allocated'),
('max_alloc_percentage', '0.25', 'Maximum allocation to any single pool'),
('conversion_rate', '0.0004', 'Token conversion fee rate'),
('min_pools', '5', 'Minimum number of pools in allocation'),
('profit_optimization', 'true', 'Enable profit optimization mode'),
('token_marketcap_limit', '1000000000', 'Minimum token market cap requirement'),
('pool_tvl_limit', '100000', 'Minimum pool TVL requirement'),
('pool_apy_limit', '0.01', 'Minimum pool APY requirement');
```

#### Allocation Parameters Table Enhancement
```sql
ALTER TABLE allocation_parameters ADD COLUMN projected_apy DECIMAL(20, 4);
ALTER TABLE allocation_parameters ADD COLUMN transaction_costs DECIMAL(20, 2);
ALTER TABLE allocation_parameters ADD COLUMN transaction_sequence JSONB;
```

#### New Views for UI Optimization:
```sql
CREATE VIEW pool_metrics_extended AS
SELECT 
    p.*,
    pdm.actual_apy,
    pdm.forecasted_apy,
    pdm.actual_tvl,
    pdm.forecasted_tvl,
    pdm.date,
    pdm.is_filtered_out,
    pdm.filter_reason
FROM pools p
JOIN pool_daily_metrics pdm ON p.pool_id = pdm.pool_id;
```

## API Design

### Authentication Endpoints
- Handled by Clerk SDK

### Data Fetching Patterns
- Server-side data loading with Remix loaders
- Optimized queries with proper indexing
- Pagination for large datasets
- Real-time updates for critical data

### Mutation Patterns
- Form submissions through Remix actions
- Optimistic updates for better UX
- Error handling and validation
- Success notifications

## Performance Considerations

### Database Optimization
- Proper indexing on frequently queried columns
- Query result caching where appropriate
- Connection pooling configuration
- Read replicas for reporting queries

### Frontend Optimization
- Code splitting by route
- Image optimization
- Lazy loading for large datasets
- Service worker for offline support

### Caching Strategy
- API response caching for static data
- Browser caching for assets
- CDN integration for deployment

## Security Considerations

### Data Protection
- Input validation and sanitization
- SQL injection prevention through Prisma
- XSS protection through React
- CSRF protection through Remix

### Access Control
- Simple authentication (all users are admins)
- API rate limiting
- Audit logging for sensitive actions
- Secure session management

## Testing Strategy

### Unit Testing
- Component testing with React Testing Library
- Utility function testing
- API endpoint testing

### Integration Testing
- Database interaction testing
- Authentication flow testing
- Form submission testing

### E2E Testing
- Critical user journey testing
- Cross-browser compatibility
- Mobile responsiveness

## Deployment Strategy

### Environment Configuration
- Development: Local with Docker
- Staging: GCP Cloud Run with test data
- Production: GCP Cloud Run with production data

### CI/CD Pipeline
- Automated testing on PR
- Build and deployment on merge
- Environment-specific configurations
- Rollback capabilities
