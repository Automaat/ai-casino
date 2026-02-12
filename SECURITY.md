# Security

## Known Vulnerabilities

### CVE-2025-69872 (diskcache)

**Status:** Accepted risk (ignored in `mise audit`)

**Vulnerability:** Unsafe pickle deserialization in diskcache 5.6.3

**Risk Assessment:** LOW for this project

- We control cache directories (`data/cache/`)
- Only deserializing our own serialized data (not user input)
- Local application (not exposed web service)
- No external/untrusted data written to cache

**Mitigation:**

- Cache directories use restrictive permissions
- Only application-controlled data cached
- Monitor for upstream fix (5.6.3 is latest as of 2026-02-12)

**Action Required:** Update to patched version when available via Renovate

### Reporting Security Issues

Report vulnerabilities via GitHub Security Advisories or email project maintainer.
