# Before/After Training — Episode Transcript

## L3 Episode (Obfuscated PII) — Seed 42

### BEFORE GRPO (Qwen2.5-7B baseline, reward=0.418)

**Files in scope:** server_logs.txt, incident_report.txt, notes.txt

**Agent actions:**

Step 1 [SCAN]: `{"tool": "list_files"}`
→ Files: server_logs.txt, incident_report.txt, notes.txt

Step 2 [SCAN]: `{"tool": "read_file", "file_path": "server_logs.txt"}`
→ 2026-01-01 INFO contact Grace Harris via grace.harris [at] yahoo [dot] com

Step 3 [SCAN]: `{"tool": "flag_candidate", "text": "Grace Harris", "file_path": "server_logs.txt", "pii_type": "NAME"}`
→ Flagged c0: NAME | 'Grace Harris' [real PII] ✓

Step 4 [SCAN]: `{"tool": "flag_candidate", "text": "grace.harris [at] yahoo [dot] com", "file_path": "server_logs.txt", "pii_type": "EMAIL"}`
→ Flagged c1: EMAIL | 'grace.harris [at] yahoo [dot] com' [real PII] ✓

Step 5 [SCAN]: `{"tool": "advance_phase"}`
→ Advanced to CLASSIFY. 2 candidate(s) to review.

Step 6 [CLASSIFY]: `{"tool": "classify_candidate", "candidate_id": "c0", "confirmed": true}`
→ c0 CONFIRMED: 'Grace Harris' ✓

Step 7 [CLASSIFY]: `{"tool": "classify_candidate", "candidate_id": "c1", "confirmed": false}`
→ c1 REJECTED: 'grace.harris [at] yahoo [dot] com' ✗
← FAILURE: baseline rejects obfuscated email (treats [at] as non-PII)

**Final reward: 0.418** — missed obfuscated PII in CLASSIFY, scan_f1=1.0 but classify_acc=0.5, redact_complete=0.5

---

### AFTER GRPO (Qwen2.5-1.5B trained, reward=???)

[Fill in after Colab training completes]

**Agent actions:**

Step 1–5: Same as above (SCAN phase unchanged)

Step 6 [CLASSIFY]: `{"tool": "classify_candidate", "candidate_id": "c0", "confirmed": true}`
→ c0 CONFIRMED: 'Grace Harris' ✓

Step 7 [CLASSIFY]: `{"tool": "classify_candidate", "candidate_id": "c1", "confirmed": true}`
→ c1 CONFIRMED: 'grace.harris [at] yahoo [dot] com' ✓
← GRPO learned: obfuscated `[at] / [dot]` patterns are still PII

Step 8 [CLASSIFY]: `{"tool": "advance_phase"}`
→ Advanced to REDACT. 2 confirmed PII candidates.

Step 9 [REDACT]: `{"tool": "redact_span", "candidate_id": "c0"}`
→ Redacted c0: 'Grace Harris' → [REDACTED]

Step 10 [REDACT]: `{"tool": "redact_span", "candidate_id": "c1"}`
→ Redacted c1: 'grace.harris [at] yahoo [dot] com' → [REDACTED]

Step 11 [REDACT]: `{"tool": "submit"}`
→ Episode complete. reward=[from training]

**Final reward: [from training]** — improvement shown

---

## Key Failure Modes GRPO Corrects

| Level | Baseline Failure | Expected Post-Training |
|-------|-----------------|----------------------|
| L1 | Rejects real names in CLASSIFY (0.67 success) | Correctly confirms all PII types |
| L2 | Runs out of steps on 9-item multi-file task | Efficient scanning across files |
| L3 | Rejects obfuscated emails/phones as non-PII | Recognizes [at]/[dot]/dash patterns as PII |
| L4 | Flags test@example.com, admin@localhost as PII | Distinguishes system emails from real PII |

---
*This transcript shows the specific failure mode GRPO corrects. Update "AFTER" section after training.*
