# Contributing

Use a branch and a pull request. Describe the concrete failure, the resulting behavior, tests run and geometry limitations. See [development](docs/development.md).

Keep geometry in metres, preserve the board scale and separate inferred shape from independently measured accuracy. Do not silently replace rejected geometry with a global convex hull, fill a required cup/bowl cavity, relax quality gates to make a case pass, or report cache-resume timing as a cold benchmark.

Do not commit captures, API keys, model weights, output caches or private logs. Use procedural fixtures for tests. Paid provider calls are opt-in and must never run from tests or CI.

Contributions to this repository's own code are provided under its MIT license. Preserve third-party notices when using external code.
