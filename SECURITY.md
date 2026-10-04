# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability-reporting flow for issues that could
expose local files, permit non-local access, execute untrusted commands, bypass
component integrity or licence gates, or leak credentials. Do not place secrets,
private reference media or working exploit payloads in a public issue.

Reports can cover both the public localhost application and the downloadable
DMG. The native wrapper and release-assembly implementation are maintained
privately, but security reports about the shipped application are welcome.

## Local trust boundary

Dang Studio binds its browser service to `127.0.0.1`. Changes must preserve the
loopback-only listener, reject untrusted Host headers, validate uploaded file
types and sizes, constrain filesystem paths, and allow only verified manifests
to influence installation commands.

Model and runtime installers must use HTTPS, immutable revision information
and pinned hashes where available. Files must be verified before becoming
loader-visible.

## Credentials and external services

Some optional components may require a publisher login or access token. Tokens
must remain in the user's local credential store or the publisher's supported
client; they must never be written into repository files, logs, generated
media, issue reports or analytics. The Studio does not use a cloud inference
service.

## Repository hygiene

Model weights, generated media, character references, authentication files,
logs, caches and installation receipts do not belong in Git. The staged-content
checker rejects common credentials and large model/media file types. CI scans
all tracked files as well. These controls supplement review and hosted secret
scanning; they are not a guarantee.

Before reporting a release-file mismatch, compare the downloaded DMG with the
SHA-256 published on the matching GitHub release. Do not run a file whose hash
does not match.
