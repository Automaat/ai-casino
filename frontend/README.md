# AI Casino Frontend

Modern trading dashboard built with SvelteKit + TypeScript + Tailwind CSS.

## Stack

- **Framework**: SvelteKit 2.x (Svelte 5)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + Typography plugin
- **Charts**: Apache ECharts + Lightweight Charts
- **Linter**: oxlint (blazing fast Rust-based linter)
- **API Client**: Auto-generated from FastAPI OpenAPI spec

## Setup

```bash
# Install dependencies
npm install

# Start dev server (http://localhost:5173)
npm run dev

# Generate API client (requires daemon running on :8484)
npm run generate-api

# Type check
npm run check

# Lint
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## Architecture

```
src/
├── lib/
│   ├── api/              # Auto-generated TypeScript client
│   ├── components/       # Reusable Svelte components
│   │   ├── charts/       # Chart components (ECharts, Lightweight Charts)
│   │   └── ui/           # UI components
│   ├── stores/           # Svelte stores for state management
│   └── types/            # TypeScript types
├── routes/               # File-based routing
│   ├── +layout.svelte    # Root layout (imports global CSS)
│   ├── +page.svelte      # Home page (overview dashboard)
│   ├── portfolio/        # Portfolio tab
│   ├── signals/          # Signals tab
│   └── risk/             # Risk tab
└── app.css               # Global styles (Tailwind directives)
```

## Development Notes

### API Client Generation

The `generate-api` script reads the OpenAPI spec from the FastAPI daemon and generates a type-safe TypeScript client:

```bash
# Daemon must be running on http://localhost:8484
npm run generate-api
```

This creates `src/lib/api/` with all API models, services, and types matching your Pydantic models.

### CORS Configuration

The FastAPI daemon needs CORS configured for development. Add to `~/.ai-casino/daemon.yaml`:

```yaml
api:
  cors_origins:
    - "http://localhost:5173"  # SvelteKit dev server
    - "http://localhost:4173"  # SvelteKit preview server
```

### Visualization Libraries

**Apache ECharts** (`svelte-echarts`):
- Kitchen-sink approach (100+ chart types)
- Use for: heatmaps, treemaps, Sankey diagrams, general charts

**Lightweight Charts**:
- Purpose-built for financial data
- Use for: candlestick charts, price charts with indicators
- Optimized for real-time tick data

### Linting with oxlint

oxlint is a fast Rust-based linter (10x faster than ESLint):

```bash
npm run lint              # Lint src/ directory
npx oxlint --fix src      # Auto-fix issues
```

Configuration via `oxlint.json` (optional - uses sensible defaults).

## Contributing

Follow the project's CLAUDE.md conventions:
- Type all function signatures
- Use `$:` for reactive statements
- Prefer composition over inheritance
- Keep components focused (single responsibility)
- No placeholder/TODO code
