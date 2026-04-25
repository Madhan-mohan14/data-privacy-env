import random
import string


_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Maria", "Nathan", "Olivia", "Peter",
]
_LAST_NAMES = [
    "Johnson", "Martinez", "White", "Brown", "Davis", "Wilson",
    "Anderson", "Taylor", "Thomas", "Jackson", "Harris", "Clark",
]
_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "company.com", "corp.io", "work.net"]
_DIAGNOSES = [
    "Type 2 Diabetes", "Hypertension", "Asthma", "Arthritis",
    "Chronic Migraine", "Anemia", "Hypothyroidism", "GERD",
]
_MEDICATIONS = [
    "Metformin 500mg", "Lisinopril 10mg", "Albuterol 90mcg",
    "Levothyroxine 50mcg", "Omeprazole 20mg", "Atorvastatin 40mg",
]
_LOG_IPS = ["192.168.1.1", "10.0.0.5", "172.16.0.3"]


def _rand_name() -> tuple[str, str]:
    return random.choice(_FIRST_NAMES), random.choice(_LAST_NAMES)


def _rand_email(first: str, last: str) -> str:
    return f"{first.lower()}.{last.lower()}@{random.choice(_EMAIL_DOMAINS)}"


def _rand_phone() -> str:
    area = random.randint(200, 999)
    mid = random.randint(100, 999)
    end = random.randint(1000, 9999)
    return random.choice([
        f"({area}) {mid}-{end}",
        f"{area}.{mid}.{end}",
        f"+1-{area}-{mid}-{end}",
    ])


def _rand_ssn() -> str:
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"


def _obfuscate_email(email: str) -> str:
    user, domain = email.split("@")
    parts = domain.split(".")
    return f"{user} [at] {parts[0]} [dot] {parts[1]}"


def _obfuscate_phone(phone: str) -> str:
    return phone.replace("-", " dash ").replace(".", " dot ").replace("(", "").replace(")", "")


def _gen_l1() -> tuple[dict[str, str], list[str]]:
    """1 file, 3 clear PII: email, phone, name."""
    first, last = _rand_name()
    name = f"{first} {last}"
    email = _rand_email(first, last)
    phone = _rand_phone()
    ip = random.choice(_LOG_IPS)
    content = (
        f"2026-01-01 INFO user {name} logged in from {ip}\n"
        f"2026-01-01 INFO support contact: {email}\n"
        f"2026-01-02 WARN call us at {phone} for help\n"
        f"2026-01-02 INFO server restarted by admin\n"
    )
    return {"server_logs.txt": content}, [name, email, phone]


def _gen_l2() -> tuple[dict[str, str], list[str]]:
    """2 files, 6 clear PII."""
    people = [_rand_name() for _ in range(3)]
    names = [f"{f} {l}" for f, l in people]
    emails = [_rand_email(f, l) for f, l in people]
    phones = [_rand_phone() for _ in range(3)]
    ip = random.choice(_LOG_IPS)

    logs = (
        f"2026-01-01 ERROR user {names[0]} failed login from {ip}\n"
        f"2026-01-01 INFO support: {emails[0]}\n"
        f"2026-01-02 INFO {names[1]} called {phones[0]}\n"
    )
    report = (
        f"Incident report\n"
        f"Contact: {names[2]}\n"
        f"Email: {emails[1]}\n"
        f"Phone: {phones[1]}\n"
        f"Alt email: {emails[2]}\n"
        f"Alt phone: {phones[2]}\n"
    )
    pii = names + emails + phones
    return {"server_logs.txt": logs, "incident_report.txt": report}, pii


def _gen_l3() -> tuple[dict[str, str], list[str]]:
    """3 files, obfuscated PII. pii_list contains obfuscated forms matching file content."""
    people = [_rand_name() for _ in range(3)]
    names = [f"{f} {l}" for f, l in people]
    raw_emails = [_rand_email(f, l) for f, l in people]
    raw_phones = [_rand_phone() for _ in range(3)]
    obf_emails = [_obfuscate_email(e) for e in raw_emails]
    obf_phones = [_obfuscate_phone(p) for p in raw_phones]
    ip = random.choice(_LOG_IPS)

    logs = (
        f"2026-01-01 INFO contact {names[0]} via {obf_emails[0]}\n"
        f"2026-01-02 WARN reach {names[1]} at {obf_phones[0]}\n"
    )
    report = (
        f"User: {names[2]}\n"
        f"Reach at: {obf_emails[1]}\n"
        f"Phone: {obf_phones[1]}\n"
    )
    notes = (
        f"Backup contacts: {obf_emails[2]} / {obf_phones[2]}\n"
        f"Server IP: {ip}\n"
    )
    # pii_list uses OBFUSCATED forms so they match what's literally in the files.
    # This allows _compute_reward() to correctly match pii_list against file content
    # and against flagged_texts — fixing the scan_recall / redact_completeness mismatch.
    pii = names + obf_emails + obf_phones
    return {
        "server_logs.txt": logs,
        "incident_report.txt": report,
        "notes.txt": notes,
    }, pii


def _gen_l4() -> tuple[dict[str, str], list[str]]:
    """3 files + red herrings (fake emails/phones that are NOT PII)."""
    files, pii = _gen_l3()
    # Add red herrings — system/test addresses that look like PII but aren't
    red_herrings = [
        "test@example.com",
        "noreply@system.local",
        "admin@localhost",
        "000.000.0000",
    ]
    for fname in list(files.keys())[:2]:
        files[fname] += f"System alert sent to {red_herrings[0]}\n"
        files[fname] += f"Auto-reply from {red_herrings[1]}\n"
    files["system_log.txt"] = (
        f"Automated notification: {red_herrings[2]}\n"
        f"Test phone: {red_herrings[3]}\n"
    )
    return files, pii  # red herrings NOT in pii list — precision test


def generate_task_for_level(level: int) -> tuple[dict[str, str], list[str]]:
    """Return (files_dict, pii_list) for the given curriculum level."""
    generators = {1: _gen_l1, 2: _gen_l2, 3: _gen_l3, 4: _gen_l4}
    return generators.get(level, _gen_l1)()
