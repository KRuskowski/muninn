# muninn

Embedded vector database and full-text search engine in Rust. Goal: as fast as possible without losing durability or recall. Read `docs/ARCHITECTURE.md` before changing anything; it is the contract between workstreams.

## Build and check
- `scripts/check.sh` runs fmt, clippy (`-D warnings`) and all tests. It must pass before every commit.
- This workstation is shared and short on RAM and disk: keep `CARGO_BUILD_JOBS` at 8 or lower (check.sh sets it), and keep test data under ~1 GB RAM. Large benchmarks go to a GCP VM, not here.
- Benchmarks: `cargo bench -p <crate>` (criterion). Put numbers in the PR with the machine they ran on.

## Style
- Google style adapted to Rust: 2-space indents and an 80-column limit, both enforced by `rustfmt.toml`.
- Public items get doc comments. Every `unsafe` block gets a `// SAFETY:` comment.
- Keep dependencies few. Ask in the PR before adding a heavy one.

## Rules
- Every SIMD or unsafe fast path has a test comparing it with a simple reference.
- Approximate-index changes report recall next to speed.
- `CHANGELOG.md` is customer-facing: describe what users can now do, not what the code does internally.
