# CLAUDE.md Best Practices - Research Findings

**Date:** Nov 29, 2025 | **Scope:** Claude Code, Copilot, Cursor, Windsurf, Aider

## TL;DR

CLAUDE.md = persistent project context for Claude Code. Anthropic: **"single most impactful optimization"**

**Core Principles:**

1. Version control project CLAUDE.md (automatic team consistency)
2. <500 lines (token efficiency)
3. Hierarchical: Global → Project → Subdirectory
4. Explicit anti-patterns (what NOT to do)
5. Incremental development rules (prevent over-engineering)
6. Data-driven loop: PR reviews → Update CLAUDE.md

## 1. Fundamentals

**Structure:**

```text
~/.claude/CLAUDE.md          # Global (personal)
./CLAUDE.md                  # Project (VERSION CONTROLLED)
./backend/CLAUDE.md          # Module-specific
```

**Resolution:** Home → Project → Subdirectory (most specific wins)

**Size:** <500 lines optimal | 10k+ words degrades | Large codebases: split by module
(47k → 9k words)

---

## 2. Content Structure

**Template:**

```markdown
# Project Name

## Overview
- 2-3 sentence description, tech stack, purpose

## Project Structure
- Directory layout, key modules, important files

## Development Workflow
- Requirements gathering (Plan Mode), incremental dev, testing

## [Language] Conventions
- Code style, error handling, testing patterns

## Simplicity Principles
- Anti-patterns to AVOID, enforcement rules, pattern consistency

## Code Generation Rules
- ALWAYS: incremental, complete code
- NEVER: long files, placeholders, over-engineering

## Common Commands
- Build, test, lint, git, deployment

## Project-Specific Context
- Domain knowledge, known issues, integration points
```

**DO Include:** Commands typed repeatedly | Architectural context (10+ min to explain) | Actual workflows |
Specific anti-patterns | Pattern examples with file refs | Domain terminology

**DON'T Include:** Generic advice | Obvious info | Outdated patterns | Sensitive data | Duplicated info |
Theoretical practices not followed

---

## 3. Language-Specific Patterns

### Go

```markdown
## Go Structure
- `/cmd/` - Main apps (one per executable, small main.go)
- `/internal/` - Private packages (90% of code, compiler enforced)
- `/pkg/` - Public libs (only if truly public)
- `/api/` - API definitions (OpenAPI, protobuf)

## Go Conventions
- Errors: return last value, check `if err != nil`, wrap with `fmt.Errorf("op: %w", err)`
- Use errors.Is/errors.As, early returns, NO panic (except exceptional)
- Table-driven tests preferred
```

**Other:** TypeScript/JS (structure, ESLint, testing) | Python (packages, type hints, pytest) | Rust (Cargo, errors, ownership)

---

## 4. Infrastructure & DevOps

```markdown
## Kubernetes
- Namespace conventions, deployment strategy (blue-green/canary/rolling)
- ALWAYS: health checks, rollback, monitoring
- Resource limits, RBAC: least-privilege

## Terraform/IaC
- Module organization, least-privilege default
- Document every permission with justification
```

**Impact:** Hours saved on IAM docs | Comprehensive infra generation | Consistent security | Faster onboarding

---

## 5. Large Codebase Strategies

**Problem:** 300k+ LoC overwhelms single CLAUDE.md

**Solutions:**

1. **Hierarchical files** - Root: overview/principles | Modules: specific patterns | Features: context-specific
2. **MCP semantic search** - claude-context server, vector DB, load only relevant code
3. **Scoped sessions** - One goal per session, explicit naming, reset when goal changes

**Session Management:**

```markdown
## Starting: State goal, set boundaries, explore, plan before code
## During: Focus on goal, no scope creep, test frequently, commit incrementally
## End: When goal achieved OR changes, clear context
```

---

## 6. Preventing Over-Engineering

**Simplicity Principles:**

```markdown
## Anti-Patterns AVOID
- Over-engineering, unnecessary abstractions, one-time helpers
- Hypothetical future design, multi-layer for simple tasks
- Long code blocks, placeholders like `// ... rest ...`

## Enforce
- Simplest solution ALWAYS, 3 similar lines > premature abstraction
- Minimal surgical changes, examine codebase FIRST
- Reuse existing, consistency > perfection

## Complexity Check
1. Can this be simpler?
2. Abstractions needed NOW (not future)?
3. Similar code exists to reuse?
4. Minimal change to achieve goal?

If unsure: STOP, ask approval
```

**Pattern Drift Prevention:** Document "drift threat vectors" - areas Claude over-complicates in YOUR codebase

```markdown
## Pattern Drift Threats
### API Handlers
- Claude adds middleware unnecessarily
- Keep simple: validate → service → response
- No caching/retry unless explicitly needed
```

---

## 7. Development Workflow

### Problem

Claude jumps to coding without: clarifying questions, understanding patterns, planning, approval

### Solution: Plan Mode (Shift+Tab twice)

**Workflow:** Explore → Plan → Approve → Code → Commit

**CLAUDE.md Instructions:**

```markdown
## Before Coding
1. ASK clarifying questions until 95% confident
2. Research existing patterns
3. Create plan, get approval
4. Work incrementally

## Plan Mode (Shift+Tab twice) for:
- Architecture decisions, multi-file changes
- New patterns/abstractions, security-critical code
```

**Breakthrough Prompt:** "Before [task], ask clarifying questions until 95% confident, then create plan and wait for approval"

---

## 8. Incremental Development

**Problem:** Claude generates 100+ line files with placeholders, untested logic, mixed concerns

**Solution:**

```markdown
## Code Generation Rules

### NEVER
- Generate entire long files, placeholders `// ... rest ...`
- Big tasks in single step, >100 lines per response

### ALWAYS
- Incremental changes (20-50 lines), surgical minimal changes
- Complete code (no placeholders), test after each step

### Incremental Steps
1. Define interfaces/types
2. Implement core (minimal)
3. Add error handling
4. Add tests
5. Iterate

### TDD
- Write tests FIRST (input/output pairs)
- Run after each change, green before next step
```

**Session Hygiene:** Reset when goal changes | <500 lines CLAUDE.md | Subdirectory files for specific areas |
One objective per session

---

## 9. PR Review Automation

**Built-in:** `/security-review` (OWASP top 10, remediation suggestions)

**GitHub Actions:**

```yaml
# .github/workflows/claude-security-review.yml
name: Claude Security Review
on: [pull_request]
jobs:
  security-review:
    runs-on: ubuntu-latest
    steps:
      - uses: anthropics/claude-code-security-review@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Custom Slash Command** (`.claude/commands/pr-review.md`):

```markdown
Review PR for: Code quality (conventions, error handling, no over-engineering) | Testing (exist, pass,
coverage) | Security (validation, auth, no sensitive data) | Completeness (no placeholders)

Commands: `gh pr view`, `git diff main...HEAD`

Output: ✅ Good | ⚠️ Concerns | 🔧 Improvements
```

**Team Pattern:** Dev uses `/pr-review` → Addresses → Creates PR → GHA security review → Team review →
Capture patterns in CLAUDE.md

**Improvement Loop:** PR reviews → identify patterns → update CLAUDE.md → better code

---

## 10. Team Consistency

**Version Control:**

```markdown
✅ Commit: ./CLAUDE.md, ./.claude/commands/*.md, ./.mcp.json
❌ Ignore: ~/.claude/CLAUDE.md, .env, CLAUDE.local.md
```

**Onboarding Impact:**

| Before | After |
| ------ | ----- |
| 2-4 weeks ramp-up | Immediate context on clone |
| Repeated explanations | Consistent AI code |
| Inconsistent AI style | Auto-enforced patterns |
| Manual enforcement | Self-documenting |

**Daily Workflow:** Start with goal → Claude reads CLAUDE.md → Plan Mode for complex → Questions before code →
Review/approve → `/pr-review` → Reset when goal changes

**Pattern Discovery:** Note manual additions → Monthly review → Create slash command if used 3+ times

**Cross-Tool:** Maintain CLAUDE.md + AGENTS.md for compatibility (Cursor, Windsurf, etc.)

---

## 11. Advanced Features

### Custom Slash Commands (`.claude/commands/`)

**Example `/commit`:**

```markdown
1. Review: `git status`, `git diff`
2. Check style: `git log -10 --oneline`
3. Format: <type>: <summary> (50 chars) | Types: feat/fix/docs/refactor/test/chore
4. Sign: `git commit -s -S -m "message"`
```

### MCP Integration (`.mcp.json`)

```json
{
  "mcpServers": {
    "kubernetes": {"command": "mcp-server-kubernetes", "args": ["--context", "prod"]},
    "terraform": {"command": "mcp-server-terraform", "args": ["--workspace", "prod"]},
    "brave-search": {"command": "mcp-server-brave-search", "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}}
  }
}
```

**Scopes:** Local (personal) | Project (`.mcp.json` committed) | User (`~/.config/claude/mcp.json`)
**Debug:** `claude-code --mcp-debug`

### Hooks

```bash
# Post-edit hook: auto-format
gofmt -w $FILE              # Go
prettier --write $FILE      # JS/TS
black $FILE                 # Python
```

### Ultrathink Mode

**What:** Max thinking budget (31,999 tokens)
**How:** Add "ultrathink" keyword to prompt
**When:** Complex architecture, multi-service, security-critical, performance optimization, large refactoring
**Best:** Opus + ultrathink + Plan Mode

---

## 12. Other AI Tools Comparison

| Tool | Config | Key Features |
| ------ | -------- | ------------ |
| **GitHub Copilot** | `.github/copilot-instructions.md` | Short statements, one per line, markdown links, enable via setting |
| **Cursor** | `.cursor/*.mdc` (new), `.cursorrules` (legacy) | <500 lines, composable rules, cursor.directory/awesome-cursorrules |
| **Windsurf** | `global_rules.md`, workspace rules | "Constitutional framework", activation modes, rulebook slash commands |
| **Aider** | `.env`, YAML | Env vars for keys, git-centric, `/add`+`/drop` commands |

**Patterns from Cursor (100+ top rules):** Functional/declarative | Early returns | Modular/reusable | No placeholders

**Windsurf Principles:** Simplicity First (SF) | Readability Priority (RP) | Dependency Minimalism (DM)

**Convergence:** All tools agree: <500 lines | Version control | Explicit anti-patterns | Incremental dev |
Project-specific context | Simplicity > complexity

---

## 13. Real-World Examples

### Java/Gradle (IntelliJ Plugin)

```markdown
# AI IntelliJ Plugin
Tech: Java 17, Gradle, IntelliJ Platform SDK

## Structure
- `/src/main/java` - Implementation | `/src/main/resources` - UI/i18n | `/src/test/java` - Tests

## Commands
./gradlew runIde/test/buildPlugin/publishPlugin

## Patterns
- UI: Swing thread | Actions: extend AnAction | Services: @Service | Extensions: plugin.xml
- i18n: `AIBundle.message("key")`, no hardcoded English
```

### Python/DevOps (AWS MCP)

```markdown
# AWS MCP Server
Tech: Python 3.10+, boto3, AWS SDK

## Style
Type hints required | Async/await for AWS | Specific exceptions (not bare except) | Google docstrings

## Security
Least-privilege IAM | Never hardcode credentials | Env vars/instance profiles | Log all API calls

## Error Handling
try: await ec2.describe_instances()
except ClientError as e:
    if e.response['Error']['Code'] == 'UnauthorizedOperation':
        logger.error(f"Insufficient perms: {e}")
        raise PermissionError("Need ec2:DescribeInstances")
    raise
```

### TypeScript/Next.js

```markdown
Tech: Next.js 14, TypeScript, Tailwind, shadcn/ui, TanStack Query

## Structure
/app (pages/layouts) | /components | /lib | /hooks | /types

## Commands
npm run dev/build/lint/test

## Conventions
Server Components default | 'use client' only when needed | TypeScript strict | Prettier

## Patterns
export function Component({ prop }: Props) { return <div>{prop}</div> }
import { Button } from "@/components/ui/button"
const { data } = useQuery({ queryKey: ['key'], queryFn: fetchData })
```

**Community Resources:** [claude-md-examples](https://github.com/ArthurClune/claude-md-examples) |
[awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) |
[awesome-cursorrules](https://github.com/PatrickJS/awesome-cursorrules) |
[ai-prompts](https://github.com/instructa/ai-prompts)

---

## 14. Metrics & Impact

**Token Efficiency:** Loaded once vs. repeated | Hierarchical: only relevant context | <500 lines optimal

**Team Impact:**

| Metric | Improvement |
| ------ | ----------- |
| Onboarding | Weeks → Days |
| Code consistency | Manual → Automatic |
| PR review cycles | -30-50% |
| Context re-explanation | -70% |
| PR review speed | +40% |
| Code style alignment | 60% → 85% |
| Time saved | 3-5 hrs/week per engineer |

---

## 15. Getting Started (30 Min)

1. **`/init`** - Auto-generates baseline, analyzes codebase
2. **Refine** - Remove generic, add project-specific, document top 10 commands
3. **Anti-patterns** - What Claude did wrong, patterns to follow, what to avoid
4. **Commit** - `git add CLAUDE.md && git commit -m "Add CLAUDE.md" && git push`
5. **Test** - New session, verify references CLAUDE.md, ask to summarize structure

**Incremental:** Week 1: structure+commands | Week 2: anti-patterns+simplicity | Week 3: 2-3 slash commands |
Week 4: MCP+refine | Ongoing: update from PR reviews

---

## 16. Common Pitfalls

| Pitfall | Problem | Solution |
| ------- | ------- | -------- |
| **Too Generic** | Copy-paste generic advice | Only project-specific patterns |
| **Too Verbose** | 5000-line CLAUDE.md | <500 lines, split by module |
| **Outdated** | Doesn't match current arch | Update during PRs, monthly audits |
| **No Anti-Patterns** | Only what TO do | Explicitly document what NOT to do |
| **Not Version Controlled** | Each engineer different context | Commit project CLAUDE.md |
| **Ignoring Feedback** | Never improves | Data-driven loop from PRs |

---

## 17. Future Directions

**Emerging:** Skills integration (reusable agents) | MCP ecosystem growth | Multi-agent orchestration | Context engineering

**Research Gaps:** Optimal size by codebase | Quantitative productivity metrics | Multi-repo best practices |
Cross-tool standardization

---

## Complete Source List

### Official

- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices) |
  [Using CLAUDE.MD](https://www.claude.com/blog/using-claude-md-files) |
  [Slash Commands](https://docs.claude.com/en/docs/claude-code/slash-commands) |
  [MCP](https://docs.claude.com/en/docs/claude-code/mcp)

### Best Practices

- [Arize Best Practices](https://arize.com/blog/claude-md-best-practices-learned-from-optimizing-claude-code-with-prompt-learning/)
- [What is CLAUDE.md](https://claudelog.com/faqs/what-is-claude-md/)
- [Plan Mode](https://claudelog.com/mechanics/plan-mode/)
- [Teaching Consistency](https://www.brandoncasci.com/2025/07/30/from-chaos-to-control-teaching-claude-code-consistency.html)
- [Stop Overengineering](https://www.nathanonn.com/how-to-stop-claude-code-from-overengineering-everything/)
- [Context Engineering](https://alabeduarte.com/context-engineering-with-claude-code-my-evolving-workflow/)

### Language-Specific

- [Go Development](https://dshills.medium.com/effective-go-development-with-claude-best-practices-for-ai-pair-programming-83fba0247a4f)
- [golang-standards/project-layout](https://github.com/golang-standards/project-layout)
- [Go Structure Guidelines](https://dev.to/jinxankit/go-project-structure-and-guidelines-4ccm)

### Infrastructure

- [DevOps](https://github.com/ruvnet/claude-flow/wiki/CLAUDE-MD-DevOps)
- [Claude for DevOps](https://www.cloudnativedeepdive.com/using-claude-and-llms-as-your-devops-platform-engineering-assistant/)
- [Terraform Workflow](https://medium.com/@balwant.matharu/how-claude-code-supercharged-my-terraform-workflow-0e0a53349251)

### Large Codebases

- [Monorepo Organization](https://dev.to/anvodev/how-i-organized-my-claudemd-in-a-monorepo-with-too-many-contexts-37k7)
- [Claude Context MCP](https://github.com/zilliztech/claude-context)
- [Large Codebase Practices](https://skywork.ai/blog/claude-code-plugin-best-practices-large-codebases-2025/)
- [Working with Large Codebases](https://medium.com/@tl_99311/claude-codes-memory-working-with-ai-in-large-codebases-a948f66c2d7e)

### Plan Mode

- [Mastering Plan Mode](https://agiinprogress.substack.com/p/mastering-claude-code-plan-mode-the)
- [Plan Mode Workflow](https://stevekinney.com/courses/ai-development/claude-code-plan-mode)
- [Incremental Development](https://www.vibesparking.com/en/blog/ai/claude-code/2025-08-14-claude-code-save-money-right-model-lean-context-permissions/)

### PR Review

- [PR Review Slash Command](https://nakamasato.medium.com/resolve-github-pr-reviews-consistently-and-rapidly-with-custom-claude-code-slash-command-3cdb25e1c2cf)
- [Automate Reviews](https://alirezarezvani.medium.com/5-tipps-to-automate-your-code-reviews-with-claude-code-5becd60bce5c)
- [Security Review Action](https://github.com/anthropics/claude-code-security-review)
- [Automate Security](https://www.claude.com/blog/automate-security-reviews-with-claude-code)

### Team

- [How Anthropic Teams Use Claude Code](https://www.anthropic.com/news/how-anthropic-teams-use-claude-code)

### Advanced

- [Configuring MCP](https://scottspence.com/posts/configuring-mcp-tools-in-claude-code)
- [Best MCP Servers](https://mcpcat.io/guides/best-mcp-servers-for-claude-code/)
- [Hooks Guide](https://liquidmetal.ai/casesAndBlogs/claude-code-hooks-guide/)
- [Ultrathink](https://www.claudecode101.com/en/tutorial/optimization/ultrathink-mode)

### Other Tools

- [Cursor Rules](https://www.prompthub.us/blog/top-cursor-rules-for-coding-agents)
- [awesome-cursorrules](https://github.com/PatrickJS/awesome-cursorrules)
- [Cursor Best Practices](https://kirill-markin.com/articles/cursor-ide-rules-for-ai/)
- [Windsurf Cascade](https://docs.windsurf.com/windsurf/cascade/cascade)
- [Windsurf Rules](https://medium.com/@wahengchang2024/mastering-windsurf-restricting-ai-output-with-windsurf-rules-d7e429654db2)
- [Copilot Instructions](https://docs.github.com/en/copilot/how-tos/custom-instructions/adding-repository-custom-instructions-for-github-copilot)
- [Tool Comparison](https://www.toolbit.ai/blog/best-ai-coding-tools-copilot-cursor-claude-comparison)
- [Aider](https://aider.chat/)

### Examples

- [claude-md-examples](https://github.com/ArthurClune/claude-md-examples)
- [awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code)
- [Claude Command Suite](https://github.com/qdhenry/Claude-Command-Suite)
- [Subagents](https://github.com/VoltAgent/awesome-claude-code-subagents)
- [Gist Examples](https://gist.github.com/cbh123/75dcd353b354b1eb3398c6d2781a502f)

---

**End of Research** | *Nov 29, 2025 - Verify against current docs as tools evolve rapidly*
