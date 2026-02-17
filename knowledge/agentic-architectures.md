# Agentic System Architectures - Research Summary

**Date:** 2026-02-12  
**Purpose:** Research compilation for AI Casino supervisor pattern migration

---

## Core Design Patterns (2026)

### Sequential Pipeline

**Characteristics:**

- Assembly line arrangement, each agent passes output to next
- Linear, deterministic, easy to debug
- Predefined fixed sequence

**Use Cases:**

- Highly structured, repeatable processes
- Clear dependencies (A → B → C → D)
- Speed/predictability critical
- Easy testing/CI integration

**Advantages:**

- Lower latency (no routing overhead)
- Predictable, easy to test
- Lower operational cost
- Reduces complexity

**Limitations:**

- Can't skip unnecessary steps
- No dynamic adaptation
- Inefficient if some analyses not needed
- Rigid, inflexible structure

**Example:**

```
Data extraction → Data cleaning → Data loading → Storage
```

### Parallel Fan-Out/Gather

**Characteristics:**

- Multiple agents operate simultaneously on same input
- Synthesizer agent aggregates results
- Independent parallel execution

**Use Cases:**

- Multi-dimensional analysis (style, security, performance)
- Independent analyses that don't depend on each other
- Time-sensitive operations requiring parallelization

**Benefits:**

- Significant speedup (3x-10x depending on agent count)
- Better resource utilization
- Diverse perspectives on same data

**Trade-offs:**

- Managing complexity and coordination overhead
- Need proper error handling for parallel tasks

### Generator + Critic (Reflection Pattern)

**Characteristics:**

- One agent creates content, another validates/refines
- Iterative improvement loop
- Quality assurance pattern

**Use Cases:**

- Quality assurance workflows
- Content creation requiring validation
- Decision verification before execution

**Benefits:**

- Catches errors, overconfident predictions
- Additional safety layer
- Iterative refinement improves quality

### Human-in-the-Loop

**Characteristics:**

- Pauses execution for human approval
- Critical for irreversible/high-consequence decisions
- Learning and trust-building mode

**Use Cases:**

- Regulatory compliance, ethical decisions
- High-risk operations (>$10k trades, <0.5 confidence)
- Learning mode (first 50 trades)
- Unusual market conditions

**Benefits:**

- Learn from expert feedback
- Build trust incrementally
- Safety for production systems

### Supervisor/Coordinator Pattern

**Characteristics:**

- Central orchestrator coordinates specialized worker agents
- Dynamic task decomposition
- Intelligent routing based on context
- Hierarchical control

**Architecture:**

```
Supervisor (Coordinator)
  ├── Analyzes request → determines needed work
  ├── Routes to workers (parallel if independent)
  ├── Worker 1 (specialized)
  ├── Worker 2 (specialized)
  ├── Worker 3 (specialized)
  └── Synthesizes final result
```

**Use Cases:**

- Multiple distinct domains (calendar, email, CRM, database)
- Complex task requiring different expertise
- Dynamic workflow (subtasks vary based on input)
- Centralized workflow control

**Benefits:**

- Adaptive workflow (skip irrelevant analyses)
- Intelligent routing based on context
- Multi-domain coordination
- Scalability - add workers without changing pipeline
- Separation of concerns

**Costs:**

- Higher latency (LLM routing decisions)
- More complex to test
- Higher operational cost (+1-3 LLM calls per analysis)

**Key Responsibilities:**

1. Receive user request
2. Decompose into subtasks
3. Route to appropriate workers
4. Monitor progress, validate outputs
5. Synthesize final unified response

**AgentOrchestra Performance:** 95.3% accuracy vs flat architectures

---

## Orchestration Patterns

### Hierarchical (Vertical) Architecture

**Structure:**

- Leader agent oversees subtasks, centralized control
- Subordinate agents report back
- Structured workflow, clear authority

**Best For:**

- Tasks too complex for single supervisor
- Layers of coordination needed
- Top-level → mid-level → lower-level delegation

### Peer Collaboration Model

**Structure:**

- Decentralized, agents work as equals
- Free collaboration, parallel processing
- Shared resources/ideas

**Best For:**

- No clear hierarchy needed
- Collaborative problem-solving
- Equal expertise domains

### Blackboard Pattern

**Structure:**

- Shared memory space for agent communication
- Agents read/write to common knowledge base
- Asynchronous coordination

**Best For:**

- Complex problem-solving requiring shared state
- Multiple agents contributing to solution
- Opportunistic reasoning

### Market-Based Pattern

**Structure:**

- Agents bid for tasks based on capability
- Economic incentive model
- Dynamic load balancing

**Best For:**

- Resource allocation optimization
- Competitive task assignment
- Dynamic pricing/priority

---

## LangGraph Architecture

### Core Concepts

**Graph-Based Orchestration:**

- Nodes = units of logic/action (LLM calls, data queries, tasks)
- Edges = control flow, conditional routing
- State management via typed schemas

**Key Capabilities:**

- Conditional routing (dynamic decision trees)
- Parallel execution
- Stateful interactions
- Branching, looping, multi-agent coordination
- Complex workflow patterns beyond linear pipelines

**Sophisticated Checkpointing:**

- Saves agent state at every step
- Pause, resume, rewind workflows
- Fault tolerance and recovery

**Architecture Approach:**

- Treats agent as graph of states and transitions
- Explicit orchestration - developer controls every transition
- More verbose, developer-driven
- Fine-grained control over complex workflows

**Benefits:**

- Precise control over state management
- Sophisticated multi-agent systems
- Cycles, branching, human oversight
- Excellent for complex workflow coordination

**Real Implementations:**

- **Vodafone:** Internal AI assistants, performance monitoring, documentation retrieval
- **AWS:** City info system (events DB, search, weather agents)
- **Creative assistants:** Screenplay generation workflows

---

## Pydantic AI Framework

### Overview

**Philosophy:**

- "Bring FastAPI feeling to GenAI development"
- Type-safe, maintainable agents
- Python-native patterns
- Software engineering rigor

**Core Concept:**

- Treats agent as high-level construct defined by data schemas and Python functions
- Focus on type safety, validation, structured outputs
- Encourages Python's native control flow

**Package Size:** ~70MB (vs LangGraph ~300MB)

### Multi-Agent Patterns

**Five Levels of Complexity:**

1. Single agent workflows
2. Agent delegation
3. Programmatic agent hand-off
4. Graph-based control flow
5. Deep agents

**Delegation Pattern:**

- Agents delegate specialized tasks through tools
- Parent agent maintains control
- Decides when to consult specialists
- Workers return control to supervisor when finished

**Example:**

```
Triage Agent (receives query)
  ├── Determines: support or loan query?
  ├── Calls Support Agent (via tool)
  └── Calls Loan Agent (via tool)
```

### Key Features

**Type Safety:**

- Every agent parameterized by types
- Pydantic BaseModel for output schema
- Automatic validation, retry on failure
- Structured outputs enforced

**Tool Integration:**

- `@agent.tool` decorator
- Combine multiple operations into tools
- One agent calls another as tool

**Orchestration Patterns:**

- **Parallelization:** Tasks divided into independent subtasks (known beforehand)
- **Orchestrator-Worker:** Orchestrator determines subtasks dynamically
- **Planning & Progress Tracking:** Break down complex tasks, track progress
- **Durable Execution:** Preserve state across failures, restarts

**Advanced Features:**

- Task delegation with isolated context
- Sandboxed code execution (Docker containers)
- Automatic conversation summarization
- Approval workflows for dangerous operations
- Graph support via type hints (alternative to LangGraph)

**Integration:**

- Prefect integration for workflow orchestration
- Automatic failure recovery
- Observability, scheduling, reliability

### Advantages Over LangGraph

**Simplicity:**

- Less verbose, more Pythonic
- Simpler state management (Pydantic models vs TypedDict)
- Combine nodes into tools
- Easier for basic conversational agents

**Performance:**

- Smaller package size (~70MB vs ~300MB)
- Lower complexity overhead
- Python-native control flow

**Type Safety:**

- Stronger validation guarantees
- Better IDE autocomplete
- Catches errors at definition time

**Maintainability:**

- Standard Python best practices
- Natural integration with Python apps
- Software engineering rigor

### When to Use Pydantic AI

**✅ Best For:**

- Type-safe, maintainable agents
- Python-centric environments
- Simpler workflows
- Rapid prototyping
- Standard software engineering practices

**❌ Not Ideal For:**

- Complex multi-agent orchestration (LangGraph's strength)
- Fine-grained control over every transition
- Sophisticated checkpointing requirements
- Leveraging LangChain ecosystem

---

## State & Memory Management

### Critical Requirements

**Mandatory Components:**

- No long-running agent without explicit plan object
- Task state tracking
- Historical decision logs
- Cross-session memory graphs

**State Types:**

- **Ephemeral:** Workflow execution state
- **Persistent:** Learning, context preservation
- **Shared:** Coordination between agents

**Best Practices:**

- Typed state schemas (TypedDict, Pydantic)
- Explicit state transitions
- State validation at boundaries
- Checkpointing for fault tolerance

---

## Coordination Mechanisms

### Task Decomposition

**Distributed Decomposition:**

- Break complex objectives into specialized subtasks
- Allocate based on agent capabilities and availability
- Dynamic vs static decomposition

**Routing Decisions:**

- Supervisor understands overall goal
- Routes tasks to appropriate workers
- Reviews output, decides next steps

### Communication Protocols

**Synchronous vs Asynchronous:**

- Synchronous: Wait for response before proceeding
- Asynchronous: Fire-and-forget, callback-based
- Event-driven messaging

**Patterns:**

- Shared memory (blackboard)
- Message passing (direct communication)
- Publish-subscribe (broadcast)

### Conflict Resolution

**Strategies:**

- Consensus mechanisms
- Priority-based arbitration
- Fallback strategies
- Voting systems

---

## Best Practices (2026)

### Architectural Principles

**Pattern over Prompt:**

- Architecture matters more than instruction wording
- Can't prompt your way out of system-level failure
- Choose pattern matching problem requirements

**Specialization Wins:**

- Skill-based specialists > generalist agents
- Hierarchical + specialized beats flat + general
- Clear responsibility boundaries
- Role-driven team composition

**State is Critical:**

- Explicit plan objects mandatory
- Memory graphs not optional
- State management foundation

**Match Pattern to Problem:**

- Sequential for pipelines
- Parallel for independence
- Supervisor for decomposition
- Hierarchical for complexity

### Production Requirements

**Monitoring & Governance:**

- Dashboards, observability tools mandatory
- Authentication, encryption for security
- Trust mechanisms between agents
- Audit trails for decisions

**Scalability:**

- Robust load balancing
- Handle nondeterministic outputs
- Graceful degradation
- Resource management

**Error Handling:**

- Validation at boundaries
- Retry mechanisms with backoff
- Fallback strategies
- Failure isolation

---

## Popular Frameworks Comparison

### LangGraph

**Strengths:**

- Complex workflow orchestration
- Graph-based architecture
- Sophisticated checkpointing
- Fine-grained control
- Mature ecosystem

**Best For:**

- Complex multi-agent systems
- Explicit workflow control
- Leveraging LangChain ecosystem
- Advanced coordination needs

**Package:** ~300MB with full support

### Pydantic AI

**Strengths:**

- Type safety, validation
- Pythonic, simple
- Smaller package size
- Rapid development
- Software engineering rigor

**Best For:**

- Type-safe maintainable agents
- Python-centric environments
- Simpler workflows
- Standard best practices

**Package:** ~70MB full

### CrewAI

**Strengths:**

- Rapid prototyping
- Role-driven orchestration
- Memory, tools, custom workflows
- Indie/open-source favorite

**Best For:**

- Quick multi-agent teams
- Defined responsibilities
- Open-source projects

### AutoGen (Microsoft)

**Strengths:**

- Conversation-driven development
- Multi-agent collaboration
- Customizable behaviors

**Best For:**

- Conversational agents
- Research/experimentation

---

## Decision Framework

### Sequential vs Supervisor

**Use Sequential When:**

- Straightforward, unchanging sequence
- Clear dependencies
- Speed critical
- Predictability required
- Easy testing needed

**Use Supervisor When:**

- Multiple distinct domains
- Dynamic routing needed
- Intelligent delegation valuable
- Complexity requires coordination
- Adaptive workflows important

### Pydantic AI vs LangGraph

**Use Pydantic AI When:**

- Type safety paramount
- Python-native patterns preferred
- Simpler workflows
- Maintainability critical
- Rapid development needed

**Use LangGraph When:**

- Complex multi-agent orchestration
- Fine-grained control required
- Sophisticated state management
- LangChain ecosystem valuable
- Advanced workflow patterns needed

---

## Migration Considerations

### LangGraph → Pydantic AI

**Simplifications:**

1. Replace TypedDict with Pydantic BaseModel
2. Combine multiple nodes into tools
3. Leverage type safety for validation
4. Use Python control flow vs graph edges

**Challenges:**

- Graph-based patterns need redesign
- Checkpointing features different
- Ecosystem tooling differences
- State management paradigm shift

**Best Approach:**

- Start with simple agents
- Gradually add complexity
- Leverage type hints
- Test extensively

---

## Key Takeaways

1. **Architecture > Prompts** - System design matters more than instruction wording
2. **Specialization Wins** - Focused specialists outperform generalists
3. **State is Foundation** - Explicit state management not optional
4. **Match Pattern to Problem** - No one-size-fits-all solution
5. **Type Safety Matters** - Validation prevents runtime failures
6. **Production Needs Governance** - Monitoring, security, observability required
7. **Start Simple, Scale Gradually** - Avoid premature complexity

---

## Sources

### Core Patterns & Architecture

- [Choose a design pattern for your agentic AI system | Google Cloud](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
- [Building AI Agents in 2026: Chatbots to Agentic Architectures](https://levelup.gitconnected.com/the-2026-roadmap-to-ai-agent-mastery-5e43756c0f26)
- [Agentic AI Design Patterns(2026 Edition) | Medium](https://medium.com/@dewasheesh.rana/agentic-ai-design-patterns-2026-ed-e3a5125162c5)
- [A practical guide to the architectures of agentic applications | Speakeasy](https://www.speakeasy.com/mcp/using-mcp/ai-agents/architecture-patterns)
- [From Prompts to Production: A Playbook for Agentic Development - InfoQ](https://www.infoq.com/articles/prompts-to-production-playbook-for-agentic-development/)
- [Designing Effective Multi-Agent Architectures | O'Reilly](https://www.oreilly.com/radar/designing-effective-multi-agent-architectures/)
- [Google's Eight Essential Multi-Agent Design Patterns - InfoQ](https://www.infoq.com/news/2026/01/multi-agent-design-patterns/)
- [4 Agentic AI Design Patterns & Real-World Examples [2026]](https://research.aimultiple.com/agentic-ai-design-patterns/)

### Orchestration Patterns

- [AI Agent Orchestration Patterns - Azure Architecture Center | Microsoft](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [Four Design Patterns for Event-Driven, Multi-Agent Systems](https://www.confluent.io/blog/event-driven-multi-agent-systems/)
- [Choosing the right orchestration pattern for multi agent systems](https://www.kore.ai/blog/choosing-the-right-orchestration-pattern-for-multi-agent-systems)
- [Design Patterns for Multi-Agent Orchestration | wethinkapp](https://www.wethinkapp.ai/blog/design-patterns-for-multi-agent-orchestration)

### LangGraph

- [LangGraph: Agent Orchestration Framework for Reliable AI Agents](https://www.langchain.com/langgraph)
- [Workflows and agents - LangChain Docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Building Agentic Workflows with LangGraph and Granite | IBM](https://www.ibm.com/think/tutorials/build-agentic-workflows-langgraph-granite)
- [Building AI Workflows with LangGraph: Practical Use Cases | Scalable Path](https://www.scalablepath.com/machine-learning/langgraph)
- [LangGraph AI Framework 2025: Complete Architecture Guide - Latenode](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-ai-framework-2025-complete-architecture-guide-multi-agent-orchestration-analysis)

### Supervisor Pattern

- [Use Agent Bricks: Supervisor Agent | Databricks](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/multi-agent-supervisor)
- [The Supervisor Pattern for Gen AI Agent Systems | Medium](https://medium.com/aitech/the-supervisor-pattern-for-gen-ai-agent-systems-d1920c0bdbbb)
- [Multi-Agent Supervisor Architecture | Databricks Blog](https://www.databricks.com/blog/multi-agent-supervisor-architecture-orchestrating-enterprise-ai-scale)
- [Building Multi-Agents Supervisor System with LangGraph | Medium](https://medium.com/@anuragmishra_27746/building-multi-agents-supervisor-system-from-scratch-with-langgraph-langsmith-b602e8c2c95d)
- [Benchmarking Multi-Agent Architectures | LangChain Blog](https://blog.langchain.com/benchmarking-multi-agent-architectures/)
- [Choosing the Right Multi-Agent Architecture | LangChain Blog](https://blog.langchain.com/choosing-the-right-multi-agent-architecture/)
- [GitHub - langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py)
- [Agent Supervisor Tutorial | LangGraph](https://langchain-ai.github.io/langgraphjs/tutorials/multi_agent/agent_supervisor/)
- [AI Agents Need a Boss: Supervisor Pattern in LangGraph | Medium](https://medium.com/@ashuashu20691/ai-agents-need-a-boss-building-with-the-supervisor-pattern-in-langgraph-mcp-9d8b7443e8fb)

### Coordination & Architecture

- [What Is Agentic Architecture? | IBM](https://www.ibm.com/think/topics/agentic-architecture)
- [Distinguishing Autonomous AI Agents from Collaborative Agentic Systems | arXiv](https://arxiv.org/html/2506.01438v1)
- [Agentic AI Architectures And Design Patterns | Medium](https://medium.com/@anil.jain.baba/agentic-ai-architectures-and-design-patterns-288ac589179a)
- [Agentic AI patterns and workflows on AWS - AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/introduction.html)
- [The Different Agentic Patterns | The AI Edge](https://newsletter.theaiedge.io/p/the-different-agentic-patterns)
- [AI Pipelines vs. Agentic AI | Stephen Collins](https://stephencollins.tech/newsletters/ai-pipelines-vs-agentic-ai-choosing-the-right-approach)
- [AI Agents vs. AI Workflows | IntuitionLabs](https://intuitionlabs.ai/articles/ai-agent-vs-ai-workflow)

### Pydantic AI

- [Multi-Agent Patterns - Pydantic AI](https://ai.pydantic.dev/multi-agent-applications/)
- [GitHub - pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai)
- [Agents - Pydantic AI](https://ai.pydantic.dev/agent/)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Building Intelligent Multi-Agent Systems with Pydantic AI | Medium](https://medium.com/@DataDo/building-intelligent-multi-agent-systems-with-pydantic-ai-f5c3d9526366)
- [Building Type-Safe AI Agents with Pydantic AI](https://blogs.justenougharchitecture.com/building-type-safe-ai-agents-with-pydantic-ai/)
- [Parallelization and orchestrator-workers workflows with Pydantic AI](https://dylancastillo.co/til/parallelization-orchestrator-workers-pydantic-ai.html)
- [Build AI Agents That Resume from Failure with Pydantic AI](https://www.prefect.io/blog/prefect-pydantic-integration)

### Framework Comparison

- [Pydantic AI vs LangGraph: Features, Integrations, and Pricing Compared - ZenML](https://www.zenml.io/blog/pydantic-ai-vs-langgraph)
- [Comparing Pydantic AI with Langgraph for Agent Development - Latenode](https://community.latenode.com/t/comparing-pydantic-ai-with-langgraph-for-agent-development/31002)
- [Pydantic AI vs LangGraph: The Ultimate Developer's Guide](https://atalupadhyay.wordpress.com/2025/07/10/pydantic-ai-vs-langgraph-the-ultimate-developers-guide/)
- [LangChain vs PydanticAI for building an AI Agent | Medium](https://medium.com/@finndersen/langchain-vs-pydanticai-for-building-an-ai-agent-e0a059435e9d)
- [Difference between pydanticAI and langgraph | BSWEN](https://docs.bswen.com/blog/2025-11-12-pydanticai-langgraph/)
- [Comparing Open-Source AI Agent Frameworks - Langfuse](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)
- [How to Choose Your AI Agent Framework](https://diamantai.substack.com/p/how-to-choose-your-ai-agent-framework)
- [Best AI Agent Frameworks in 2025](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more)
