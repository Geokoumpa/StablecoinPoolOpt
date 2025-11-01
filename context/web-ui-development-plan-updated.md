# Stablecoin Pool Optimization Admin UI - Development Plan (Updated for Next.js)

## Overview

This document provides a comprehensive development plan for implementing the admin web interface for the Stablecoin Pool Optimization system using **Next.js**. The plan is organized into phases with clear deliverables, dependencies, and implementation details. 

**🚨 IMPORTANT UPDATE**: This plan has been updated to reflect the migration from React Router v7 to Next.js while maintaining all existing functionality and UI/UX design.

## Migration Summary

### Framework Change
- **Previous**: React Router v7 (formerly Remix)
- **New**: Next.js 14+ with App Router
- **Reason**: Team familiarity with Next.js ecosystem and better community support
- **Approach**: Maintain exact same UI/UX design and component structure

### Migration Timeline
- **Duration**: 5 weeks for migration
- **Overlap**: Migration will run parallel to existing development phases
- **Impact**: Minimal disruption to ongoing development

## Development Phases

### Phase 0: Pipeline Integration (Week 0.5) ✅ **COMPLETED**

#### 0.1 Pipeline Modification for Default Parameters ✅ **COMPLETED**
**Tasks:**
- Modify asset allocation pipeline to read from default_allocation_parameters
- Update create_allocation_snapshots.py to use defaults when no run exists
- Modify fetch_allocation_parameters() in optimize_allocations.py to read defaults first
- Implement parameter override logic for custom runs
- Add logging for parameter source tracking

**Deliverables:**
- Modified create_allocation_snapshots.py to read from default_allocation_parameters
- Updated fetch_allocation_parameters() in optimize_allocations.py
- Pipeline changes to use defaults when no run exists
- Parameter override functionality for custom runs
- Logging for parameter source tracking

**Dependencies:**
- Database schema implementation
- Access to asset allocation codebase

#### 0.2 Database Migration for Defaults ✅ **COMPLETED**
**Tasks:**
- Create migration script for default_allocation_parameters table
- Populate table with initial default values
- Test parameter retrieval in pipeline
- Update database documentation

**Deliverables:**
- Migration script
- Default parameter data
- Updated documentation

**Dependencies:**
- Database access permissions

### Phase 1: Next.js Migration Foundation (Week 1-2)

#### 1.1 Project Migration Setup ✅ **COMPLETED**
**Tasks:**
- Initialize new Next.js project with App Router
- Configure development environment for Next.js
- Set up version control with migration branch
- Create migration project structure
- Backup existing React Router codebase

**Deliverables:**
- Next.js application structure
- Development environment configuration
- Git repository with migration strategy
- Backup of existing codebase

**Dependencies:**
- Node.js 18+
- npm/yarn package manager
- Git

#### 1.2 Core Dependencies Migration ✅ **COMPLETED**
**Tasks:**
- Remove React Router dependencies
- Install Next.js specific packages
- Update Prisma for Next.js compatibility
- Migrate Clerk to @clerk/nextjs
- Set up Tailwind CSS for Next.js
- Configure TypeScript for Next.js

**Deliverables:**
- Updated package.json with Next.js dependencies
- Next.js configuration files
- Tailwind CSS configuration
- TypeScript configuration

**Dependencies:**
- Phase 1.1 completion
- Database credentials
- Clerk API keys

#### 1.3 Authentication System Migration ✅ **COMPLETED**
**Tasks:**
- Migrate from @clerk/clerk-react to @clerk/nextjs
- Implement Next.js middleware for authentication
- Convert ProtectedRoute component to Next.js patterns
- Update authentication flow for App Router
- Implement server-side auth checks

**Deliverables:**
- Working Next.js authentication system
- Middleware configuration
- Protected route implementation
- Session management

**Dependencies:**
- Phase 1.2 completion
- Clerk configuration

### Phase 2: Routing & Layout Migration (Week 2-3)

#### 2.1 Routing Structure Conversion ✅ **COMPLETED**
**Tasks:**
- Convert React Router routes to Next.js App Router
- Migrate dynamic routes ($param → [param])
- Implement nested layouts
- Set up route groups for organization
- Update navigation components

**Deliverables:**
- Next.js App Router structure
- Migrated routes
- Layout system
- Navigation components

**Dependencies:**
- Phase 1.3 completion

#### 2.2 Layout Components Migration ✅ **COMPLETED**
**Tasks:**
- Convert DashboardLayout to Next.js layout pattern
- Implement root layout with providers
- Create nested layouts for sections
- Migrate responsive design patterns
- Implement loading states with Next.js patterns

**Deliverables:**
- Next.js layout components
- Navigation system
- Responsive design foundation
- Loading states

**Dependencies:**
- Phase 2.1 completion

#### 2.3 Common UI Components Migration 🔄 **PARTIALLY COMPLETED**
**Tasks:**
- ✅ Migrate Shadcn/ui components to Next.js
- ✅ Update custom data table component
- ⚠️ Implement modal system for Next.js
- ❌ Create form validation utilities
- ❌ Set up notification system

**Deliverables:**
- ✅ Component library
- ❌ Form validation system
- ❌ Notification framework
- ✅ Updated UI components

**Dependencies:**
- Phase 2.2 completion

### Phase 3: Data Layer Migration (Week 3-4)

#### 3.1 Database Integration Migration ✅ **COMPLETED**
**Tasks:**
- Migrate Prisma client setup for Next.js
- Update database connection for Next.js environment
- Implement proper connection pooling
- Set up database middleware if needed
- Test database operations

**Deliverables:**
- Next.js database integration
- Prisma configuration
- Connection optimization
- Database testing

**Dependencies:**
- Phase 2.3 completion

#### 3.2 API Routes Migration ❌ **NOT STARTED**
**Tasks:**
- Convert Remix loaders to Next.js Server Components
- Implement Next.js API routes
- Migrate form submissions to route handlers
- Set up proper error handling
- Implement caching strategies

**Deliverables:**
- Next.js API endpoints
- Server Components
- Route handlers
- Error handling

**Dependencies:**
- Phase 3.1 completion

#### 3.3 Data Fetching Patterns ❌ **NOT STARTED**
**Tasks:**
- Convert client-side data fetching
- Implement Next.js data fetching patterns
- Set up proper caching strategies
- Update data transformation utilities
- Implement real-time updates

**Deliverables:**
- Data loading strategies
- Caching implementation
- Data utilities
- Real-time features

**Dependencies:**
- Phase 3.2 completion

### Phase 4: Feature Implementation (Week 4-8)

#### 4.1 Dashboard Implementation (Week 4) 🔄 **PARTIALLY COMPLETED**
**Tasks:**
- ✅ Migrate dashboard data layer to Next.js
- ✅ Implement dashboard UI components
- ✅ Create metric cards component
- ❌ Build chart components for visualizations
- ✅ Assemble dashboard page

**Deliverables:**
- ✅ Complete dashboard page
- ✅ Dashboard UI components
- ❌ Chart visualizations
- ✅ Interactive elements

**Dependencies:**
- Phase 3.3 completion

#### 4.2 Optimization History (Week 5) 🔄 **PARTIALLY COMPLETED**
**Tasks:**
- ✅ Migrate optimization runs list
- ✅ Implement single optimization run page
- ❌ Add export & search features
- ❌ Create pagination logic
- ❌ Implement filtering capabilities

**Deliverables:**
- ❌ Optimization runs API
- ✅ Detailed optimization run page
- ❌ Export functionality
- ❌ Search and filter features

**Dependencies:**
- Phase 4.1 completion

#### 4.3 Pool Management (Week 6) 🔄 **PARTIALLY COMPLETED**
**Tasks:**
- ✅ Migrate pools list page
- ✅ Implement pool metrics page
- ✅ Create single pool details
- ❌ Add protocol filtering
- ❌ Implement advanced filtering system

**Deliverables:**
- ✅ Pools list page
- ✅ Pool metrics page
- ✅ Single pool page
- ❌ Filtering system

**Dependencies:**
- Phase 4.2 completion

#### 4.4 Protocols & Tokens Management (Week 7) 🔄 **PARTIALLY COMPLETED**
**Tasks:**
- Migrate approved protocols management
- Implement token management
- Create blacklisted tokens management
- Add CRUD operations
- Implement address validation

**Deliverables:**
- Protocol management page
- Token management pages
- Blacklist management
- CRUD operations

**Dependencies:**
- Phase 4.3 completion

#### 4.5 Configuration Management (Week 8) 🔄 **PARTIALLY COMPLETED**
**Tasks:**
- ❌ Migrate configuration API
- ✅ Implement configuration UI
- ⚠️ Create wallet management
- ❌ Add parameter validation
- ❌ Implement save/reset functionality

**Deliverables:**
- ❌ Configuration API
- ✅ Configuration management page
- ⚠️ Wallet management interface
- ❌ Form validation

**Dependencies:**
- Phase 4.4 completion

### Phase 5: Testing & Deployment (Week 9-10)

#### 5.1 Migration Testing (Week 9) ❌ **NOT STARTED**
**Tasks:**
- Test all migrated routes and functionality
- Validate authentication flow
- Test database connections
- Performance testing
- Cross-browser compatibility testing

**Deliverables:**
- Test suite
- Migration validation
- Performance benchmarks
- Compatibility report

**Dependencies:**
- Phase 4.5 completion

#### 5.2 Performance Optimization (Week 10) ❌ **NOT STARTED**
**Tasks:**
- Implement Next.js image optimization
- Set up proper caching headers
- Optimize bundle size
- Configure build optimizations
- Implement code splitting

**Deliverables:**
- Optimized application
- Improved performance metrics
- Caching implementation
- Build optimization

**Dependencies:**
- Phase 5.1 completion

#### 5.3 Deployment Setup (Week 10) ❌ **NOT STARTED**
**Tasks:**
- Configure GCP Cloud Run for Next.js
- Set up CI/CD pipeline for Next.js
- Implement environment configurations
- Create deployment documentation
- Migrate production environment

**Deliverables:**
- Deployed Next.js application
- CI/CD pipeline
- Deployment documentation
- Production migration

**Dependencies:**
- Phase 5.2 completion

## Current Completion Status Summary

### Overall Progress: ~60% Complete

**Phase Completion Breakdown:**
- ✅ **Phase 0**: Pipeline Integration - **100% Complete**
- ✅ **Phase 1**: Next.js Migration Foundation - **100% Complete**  
- ✅ **Phase 2**: Routing & Layout Migration - **100% Complete**
- 🔄 **Phase 3**: Data Layer Migration - **33% Complete** (Database done, API/Data Fetching missing)
- 🔄 **Phase 4**: Feature Implementation - **40% Complete** (UI structure done, data integration partially complete)
- ❌ **Phase 5**: Testing & Deployment - **0% Complete** (Not started)

### Key Achievements ✅
- Successfully migrated from React Router to Next.js 14 with App Router
- Complete authentication system with Clerk integration
- Full UI component library with Shadcn/ui
- Responsive layout system with navigation
- Database schema and migrations completed
- Basic page structures for all features implemented

### Critical Next Steps 🚨
1. **Implement API Routes** - Create Next.js API endpoints in `web-ui/src/app/api/`
2. **Add Data Integration** - Connect UI components to actual data sources
3. **Complete CRUD Operations** - Implement create, read, update, delete operations
4. **Add Chart Components** - Implement data visualizations for dashboard
5. **Implement Form Validation** - Add validation utilities and error handling
6. **Set Up Testing** - Create test suite and validation

### Immediate Priorities (Next 1-2 weeks)
1. Phase 3.2: API Routes Migration
2. Phase 3.3: Data Fetching Patterns  
3. Phase 4.1-4.5: Complete data integration for all features

### Medium-term Priorities (Next 2-4 weeks)
1. Complete remaining Phase 4 features (CRUD, validation, charts)
2. Begin Phase 5: Testing & Deployment

---

## Technical Implementation Details

### Next.js Project Structure

```
web-ui-nextjs/
├── app/                          # Next.js App Router
│   ├── (auth)/                   # Route group for auth pages
│   │   ├── login/
│   │   │   └── page.tsx
│   │   └── signup/
│   │       └── page.tsx
│   ├── (dashboard)/               # Route group for dashboard
│   │   ├── layout.tsx           # Dashboard layout
│   │   ├── page.tsx             # Dashboard home
│   │   ├── optimization/
│   │   │   ├── page.tsx         # Optimization runs list
│   │   │   └── [runId]/
│   │   │       └── page.tsx     # Single run details
│   │   ├── pools/
│   │   │   ├── page.tsx         # Pools list
│   │   │   ├── [poolId]/
│   │   │   │   └── page.tsx     # Single pool details
│   │   │   └── metrics/
│   │   │       └── page.tsx     # Pool metrics
│   │   ├── protocols/
│   │   │   └── page.tsx         # Protocols management
│   │   └── config/
│   │       └── page.tsx         # Configuration management
│   ├── api/                     # Next.js API routes
│   │   ├── dashboard/
│   │   │   └── route.ts         # Dashboard metrics
│   │   ├── optimization/
│   │   │   ├── runs/
│   │   │   │   └── route.ts     # Optimization runs
│   │   │   └── [runId]/
│   │   │       └── route.ts     # Single run details
│   │   ├── pools/
│   │   │   ├── list/
│   │   │   │   └── route.ts     # Pools list
│   │   │   ├── metrics/
│   │   │   │   └── route.ts     # Pool metrics
│   │   │   └── [poolId]/
│   │   │       └── route.ts     # Single pool details
│   │   ├── protocols/
│   │   │   ├── approved/
│   │   │   │   └── route.ts     # Approved protocols
│   │   │   ├── tokens/
│   │   │   │   └── route.ts     # Approved tokens
│   │   │   └── blacklisted/
│   │   │       └── route.ts     # Blacklisted tokens
│   │   └── config/
│   │       ├── parameters/
│   │       │   └── route.ts     # Configuration parameters
│   │       └── wallets/
│   │           └── route.ts       # Wallet addresses
│   ├── globals.css              # Global styles
│   ├── layout.tsx              # Root layout
│   ├── loading.tsx             # Root loading
│   ├── error.tsx              # Root error boundary
│   └── not-found.tsx          # 404 page
├── components/                 # Reusable components
│   ├── ui/                   # Shadcn/ui components
│   ├── charts/               # Chart components
│   ├── forms/                # Form components
│   ├── tables/               # Table components
│   └── layout/               # Layout components
├── lib/                      # Utility libraries
│   ├── db.ts                 # Database connection
│   ├── auth.ts               # Authentication utilities
│   ├── utils.ts              # General utilities
│   └── validations.ts        # Form validations
├── prisma/                   # Database
│   ├── schema.prisma         # Database schema
│   └── migrations/           # Database migrations
├── public/                   # Static assets
├── middleware.ts             # Next.js middleware
├── next.config.js           # Next.js configuration
├── tailwind.config.js       # Tailwind configuration
├── tsconfig.json           # TypeScript configuration
└── package.json            # Dependencies and scripts
```

### Database Schema Implementation

#### New Tables Required:

1. **default_allocation_parameters**
```sql
CREATE TABLE default_allocation_parameters (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(255) UNIQUE NOT NULL,
    parameter_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

2. **allocation_parameters** (Enhancement)
```sql
ALTER TABLE allocation_parameters ADD COLUMN projected_apy DECIMAL(20, 4);
ALTER TABLE allocation_parameters ADD COLUMN transaction_costs DECIMAL(20, 2);
ALTER TABLE allocation_parameters ADD COLUMN transaction_sequence JSONB;
```

#### Database Views:
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

### API Routes Structure (Next.js)

```
app/api/
├── dashboard/route.ts          # Dashboard metrics
├── optimization/
│   ├── runs/route.ts          # Optimization runs list
│   └── [runId]/route.ts      # Single run details
├── pools/
│   ├── list/route.ts          # Pools list
│   ├── metrics/route.ts        # Pool metrics
│   └── [poolId]/route.ts     # Single pool details
├── protocols/
│   ├── approved/route.ts       # Approved protocols
│   ├── tokens/route.ts         # Approved tokens
│   └── blacklisted/route.ts    # Blacklisted tokens
└── config/
    ├── parameters/route.ts     # Configuration parameters
    └── wallets/route.ts       # Wallet addresses
```

### Component Architecture

```
components/
├── ui/                 # Shadcn/ui components
├── charts/             # Chart components
│   ├── LineChart.tsx
│   ├── PieChart.tsx
│   └── BarChart.tsx
├── forms/              # Form components
│   ├── ProtocolForm.tsx
│   ├── TokenForm.tsx
│   └── ConfigForm.tsx
├── tables/             # Table components
│   ├── DataTable.tsx
│   ├── PoolTable.tsx
│   └── OptimizationTable.tsx
└── layout/             # Layout components
    ├── Header.tsx
    ├── Sidebar.tsx
    └── Layout.tsx
```

## Migration Mapping

### Route Mapping
| React Router v7 | Next.js App Router |
|----------------|-------------------|
| `app/routes/_index.tsx` | `app/(dashboard)/page.tsx` |
| `app/routes/login.tsx` | `app/(auth)/login/page.tsx` |
| `app/routes/optimization/_index.tsx` | `app/(dashboard)/optimization/page.tsx` |
| `app/routes/optimization/$runId.tsx` | `app/(dashboard)/optimization/[runId]/page.tsx` |
| `app/routes/pools/_index.tsx` | `app/(dashboard)/pools/page.tsx` |
| `app/routes/pools/$poolId.tsx` | `app/(dashboard)/pools/[poolId]/page.tsx` |
| `app/routes/pools/metrics.tsx` | `app/(dashboard)/pools/metrics/page.tsx` |
| `app/routes/protocols/_index.tsx` | `app/(dashboard)/protocols/page.tsx` |
| `app/routes/config/_index.tsx` | `app/(dashboard)/config/page.tsx` |

### Component Migration
| React Router Pattern | Next.js Pattern |
|-------------------|-----------------|
| Remix loaders | Server Components / getServerSideProps |
| Remix actions | Route handlers (POST/PUT/DELETE) |
| useLoaderData | Direct data fetching in Server Components |
| useActionData | Form handling with Next.js patterns |
| Outlet | Children prop in layouts |

### Authentication Migration
| React Router v7 | Next.js |
|----------------|---------|
| `<ClerkProvider>` | `<ClerkProvider>` (same) |
| `ProtectedRoute` component | Middleware + Layout patterns |
| `useUser()` hook | `useUser()` hook (same) |
| Route-based protection | Middleware-based protection |

## Development Guidelines

### Code Standards
- TypeScript for type safety
- ESLint + Prettier for code formatting
- Conventional commits for version control
- Component-driven development
- Server/Client component separation

### Testing Strategy
- Unit tests with Jest + React Testing Library
- Integration tests for API routes
- E2E tests with Playwright
- Minimum 80% code coverage
- Migration-specific testing

### Performance Considerations
- Implement proper indexing on database queries
- Use React.memo for component optimization
- Implement virtual scrolling for large tables
- Optimize bundle size with Next.js code splitting
- Leverage Next.js caching mechanisms

### Security Measures
- Input validation on all forms
- SQL injection prevention through Prisma
- XSS protection through React
- CSRF protection through Next.js
- Rate limiting on API endpoints
- Middleware-based authentication

## Migration Risk Assessment

### High Risks
1. **Authentication Flow Changes**: Clerk integration differences
   - Mitigation: Thorough testing of auth flows
   
2. **Server Component Boundaries**: Proper separation of client/server code
   - Mitigation: Careful component architecture planning

### Medium Risks
1. **Build Process Changes**: Different build optimization
   - Mitigation: Performance testing and optimization
   
2. **Environment Variables**: Different handling in Next.js
   - Mitigation: Comprehensive environment setup

### Low Risks
1. **Styling Issues**: Minor CSS differences
   - Mitigation: Visual testing and adjustments
   
2. **Import Path Changes**: File structure differences
   - Mitigation: Systematic path updates

## Post-Migration Benefits

1. **Better Community Support**: Larger Next.js ecosystem
2. **Improved Performance**: Next.js optimizations
3. **Enhanced Developer Experience**: Better tooling
4. **Future-Proof**: Active development and updates
5. **Deployment Flexibility**: Multiple deployment options
6. **Built-in Features**: Image optimization, internationalization, edge functions

## Success Metrics

### Performance Metrics
- Page load time < 2 seconds
- API response time < 500ms
- 99.9% uptime
- Mobile responsiveness score > 95

### Quality Metrics
- Test coverage > 80%
- Zero critical security vulnerabilities
- Accessibility score > 90
- User satisfaction > 4.5/5

### Migration Success Metrics
- All existing functionality preserved
- Identical UI/UX design maintained
- Performance improved or maintained
- Zero data loss during migration
- Team comfortable with new stack

## Deployment Strategy

### Environment Setup
1. **Development**: Local development with Next.js dev server
2. **Staging**: GCP Cloud Run with test database
3. **Production**: GCP Cloud Run with production database

### CI/CD Pipeline
1. **Code Quality**: Automated linting and testing
2. **Build**: Next.js build and optimization
3. **Deploy**: Automated deployment to staging
4. **Release**: Manual promotion to production

### Migration Deployment
1. **Parallel Deployment**: Run both versions during migration
2. **Feature Flags**: Gradual rollout of Next.js features
3. **Database Migration**: Zero-downtime database migration
4. **DNS Switch**: Seamless transition to Next.js

## Resource Requirements

### Development Team
- 1 Full-stack developer (Next.js/React/TypeScript)
- 0.5 Database specialist (for optimization)
- 0.5 UI/UX designer (for design refinement)
- 0.5 DevOps engineer (for migration deployment)

### Infrastructure
- GCP Cloud Run for hosting (Next.js optimized)
- GCP Cloud SQL (existing)
- GCP Cloud Storage for assets
- Monitoring and logging tools

### Timeline
- **Migration Duration**: 5 weeks
- **Parallel Development**: Ongoing feature development continues
- **Total Duration**: 10 weeks (migration + remaining features)
- **MVP Delivery**: Week 5 (migration complete)
- **Full Release**: Week 10 (all features complete)
- **Post-launch support**: Ongoing

## Conclusion

This updated development plan provides a structured approach to migrating the admin UI from React Router v7 to Next.js while ensuring quality, performance, and security throughout the migration process. The phased approach minimizes risk and allows for thorough testing at each stage while maintaining all existing functionality and design.

The migration to Next.js will provide better community support, improved performance, and enhanced developer experience while preserving the exact same UI/UX design that users are familiar with.