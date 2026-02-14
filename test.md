# LLM Gateway

A production-grade LLM Gateway service that provides unified API access to multiple LLM providers (OpenAI, Anthropic, local models) with intelligent traffic management, graduated circuit breaking, and enterprise features.

## Features

- **Multi-Provider Routing** - Route requests to OpenAI, Anthropic, or local models
- **Graduated Traffic Shifting** - Intelligent circuit breaker with canary-based recovery
- **License Tiers** - Free, Pro, and Enterprise tiers with model/rate restrictions
- **Response Caching** - In-memory caching with TTL for identical requests
- **Session Memory** - Cross-model conversation continuity
- **Usage Tracking** - Per-user token and cost accounting
- **Health Checking** - Periodic provider health probes

## Architecture

```mermaid
graph TB
    subgraph Client
        A[HTTP Request]
    end

    subgraph Gateway
        B[Auth Service]
        C[License Check]
        D[Rate Limiter]
        E[Cache]
        F[Memory Store]
        G[Router]
        H[Traffic Shifting CB]
    end

    subgraph Providers
        I[OpenAI]
        J[Anthropic]
        K[Local vLLM]
    end

    subgraph Monitoring
        L[Usage Tracker]
        M[Health Checker]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E -->|Miss| F
    F --> G
    G --> H
    H -->|Primary| I
    H -->|Secondary| J
    H -->|Tertiary| K
    I --> L
    J --> L
    K --> L
    M -.->|Probes| I
    M -.->|Probes| J
    M -.->|Probes| K
```

## Request Processing Pipeline

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as Gateway
    participant Auth as AuthService
    participant RL as RateLimiter
    participant Cache as Cache
    participant Mem as MemoryStore
    participant Router as Router
    participant CB as TrafficShiftCB
    participant P1 as Primary Provider
    participant P2 as Secondary Provider
    participant Usage as UsageTracker

    C->>GW: POST /v1/chat/completions
    GW->>Auth: Authenticate(apiKey)
    Auth-->>GW: User

    GW->>Auth: CheckLicense(user, model)
    Auth-->>GW: OK

    GW->>RL: AllowRequest(userId, limits)
    RL-->>GW: OK

    GW->>Cache: Get(cacheKey)
    alt Cache Hit
        Cache-->>GW: Cached Response
        GW-->>C: Response
    else Cache Miss
        Cache-->>GW: nil

        GW->>Mem: GetContextMessages(sessionId)
        Mem-->>GW: History Messages

        GW->>Router: Route(context)
        Router-->>GW: [Primary, Secondary, ...]

        GW->>CB: ShouldRouteToPrimary()
        alt Route to Primary
            CB-->>GW: true
            GW->>P1: Complete(request)
            alt Success
                P1-->>GW: Response
                GW->>CB: RecordSuccess()
            else Failure
                P1-->>GW: Error
                GW->>CB: RecordFailure()
                GW->>P2: Complete(request)
                P2-->>GW: Response
            end
        else Route to Secondary
            CB-->>GW: false
            GW->>P2: Complete(request)
            P2-->>GW: Response
        end

        GW->>Cache: Set(cacheKey, response)
        GW->>Mem: AppendMessages(sessionId, messages)
        GW->>Usage: Record(requestLog)
        GW-->>C: Response
    end
```

## Traffic Shifting State Machine

The gateway uses a graduated circuit breaker instead of binary open/closed states:

```mermaid
stateDiagram-v2
    [*] --> HEALTHY

    HEALTHY --> DEGRADED: failures >= threshold

    DEGRADED --> RECOVERING: canary successes >= needed
    DEGRADED --> FULLY_OPEN: canary failures >= max

    RECOVERING --> HEALTHY: reached 100%
    RECOVERING --> DEGRADED: any failure during ramp

    FULLY_OPEN --> DEGRADED: cooldown elapsed

    note right of HEALTHY: 100% to primary
    note right of DEGRADED: 5% canary to primary\n95% to secondary
    note right of RECOVERING: Ramp: 25% → 50% → 75% → 100%
    note right of FULLY_OPEN: 0% to primary\n100% to secondary
```

### Traffic Distribution by State

| State | Primary % | Secondary % | Behavior |
|-------|-----------|-------------|----------|
| HEALTHY | 100% | 0% | All traffic to primary |
| DEGRADED | 5% | 95% | Canary probing primary |
| RECOVERING | 25→50→75→100% | 75→50→25→0% | Gradual ramp-up |
| FULLY_OPEN | 0% | 100% | Complete failover |

## Routing Strategies

```mermaid
graph LR
    subgraph Strategies
        A[Priority] --> B[Lower number wins]
        C[Weighted] --> D[Random by weight]
        E[Cost-Based] --> F[Cheapest first]
        G[Round-Robin] --> H[Cycle through]
    end
```

## Component Interactions

```mermaid
graph TB
    subgraph Core
        Gateway[Gateway]
        Handler[HTTP Handlers]
    end

    subgraph Authentication
        Auth[AuthService]
        Users[(Users DB)]
    end

    subgraph Traffic
        Router[Router]
        CB[TrafficShiftRegistry]
        TS[TrafficShiftingCB]
    end

    subgraph Providers
        Registry[ProviderRegistry]
        OpenAI[OpenAIProvider]
        Anthropic[AnthropicProvider]
        Mock[MockProvider]
    end

    subgraph State
        Cache[Cache]
        Memory[MemoryStore]
        RateLimiter[RateLimiter]
        Usage[UsageTracker]
    end

    subgraph Background
        HealthCheck[HealthChecker]
    end

    Gateway --> Handler
    Handler --> Auth
    Auth --> Users
    Handler --> Router
    Router --> CB
    CB --> TS
    Handler --> Registry
    Registry --> OpenAI
    Registry --> Anthropic
    Registry --> Mock
    Handler --> Cache
    Handler --> Memory
    Handler --> RateLimiter
    Handler --> Usage
    HealthCheck --> Registry
    HealthCheck --> CB
```

## License Tiers

```mermaid
graph TB
    subgraph Free Tier
        F1[gpt-3.5-turbo]
        F2[claude-haiku]
        F3[llama-3-8b]
        F4[mistral-7b]
        F5[10 RPM / 40K TPM]
    end

    subgraph Pro Tier
        P1[All Free models]
        P2[gpt-4o / gpt-4o-mini]
        P3[claude-sonnet]
        P4[60 RPM / 200K TPM]
    end

    subgraph Enterprise Tier
        E1[All Pro models]
        E2[500 RPM / 2M TPM]
        E3[Priority routing]
    end
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Main chat completion API (OpenAI-compatible) |
| GET | `/health` | Gateway health status |
| GET | `/v1/providers/status` | Traffic shift states for all providers |
| GET | `/v1/usage` | Usage and cost tracking |
| GET | `/v1/cache/stats` | Cache statistics |
| PUT | `/v1/providers/{id}/down` | Simulate provider failure |
| PUT | `/v1/providers/{id}/up` | Restore provider |

## Quick Start

```bash
# Run the gateway
go run .

# Test with a request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer gw-pro-key-456" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}'
```

## Demo API Keys

| Tier | API Key |
|------|---------|
| Free | `gw-free-key-123` |
| Pro | `gw-pro-key-456` |
| Enterprise | `gw-ent-key-789` |

## Demo Flow

```mermaid
sequenceDiagram
    participant User
    participant Gateway
    participant OpenAI as OpenAI (Primary)
    participant Anthropic as Anthropic (Secondary)

    Note over Gateway: State: HEALTHY (100% OpenAI)

    User->>Gateway: Send requests
    Gateway->>OpenAI: Route 100%
    OpenAI-->>Gateway: Success

    User->>Gateway: PUT /providers/openai/down
    Note over Gateway: State: DEGRADED (5%/95%)

    User->>Gateway: Send requests
    Gateway->>Anthropic: Route 95%
    Gateway->>OpenAI: Canary 5%

    User->>Gateway: PUT /providers/openai/up
    Note over Gateway: State: RECOVERING

    loop Gradual Ramp
        Gateway->>OpenAI: 25% → 50% → 75%
        OpenAI-->>Gateway: Success
    end

    Note over Gateway: State: HEALTHY (100% OpenAI)
```

## File Structure

```
├── main.go            # Entry point
├── handler.go         # Gateway struct and HTTP handlers
├── models.go          # Data structures
├── config.go          # Configuration
├── router.go          # Provider selection and ordering
├── circuitbreaker.go  # Graduated traffic-shifting CB
├── provider.go        # Provider adapters (OpenAI, Anthropic, Mock)
├── auth.go            # Authentication and license checking
├── cache.go           # Response caching
├── ratelimiter.go     # Per-user rate limiting
├── memory.go          # Session/conversation memory
├── usage.go           # Token and cost tracking
├── healthcheck.go     # Provider health probing
└── go.mod             # Go module file
```

## Configuration

Default configuration is set in `config.go`. Key settings:

- **Port**: 8080
- **Routing Strategy**: Priority-based
- **Cache TTL**: 3600 seconds
- **Health Check Interval**: 30 seconds
- **Circuit Breaker Thresholds**:
  - Failure threshold: 5
  - Canary successes needed: 3
  - Ramp successes needed: 5
  - Cooldown: 60 seconds
