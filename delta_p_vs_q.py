import re
import matplotlib.pyplot as plt

NOTE_PATTERN = re.compile(r'([\d.]+)\s*LPM\s*\(?\s*(\d+\.?\d*)\.?\s*C?\)?')

def parse_flow_temps(filename):
    """Return one (flow_rate, delta_p) pair per distinct logging note, in file order."""
    rows = []
    seen_notes = set()
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[4:]:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 4:
            continue
        note = fields[3].strip()
        if not note or note in seen_notes:
            continue
        match = NOTE_PATTERN.search(note)
        if not match:
            continue
        seen_notes.add(note)
        flow = float(match.group(1))
        temp = float(match.group(2))
        rows.append((flow, temp))

    return rows

def flow_avg(rows):
    """Return the average flow rate and pressure drop from the parsed rows."""
    if not rows:
        return None, None
    total_flow = sum(flow for flow, delta_p in rows)
    total_delta_p = sum(delta_p for flow, delta_p in rows)
    avg_flow = total_flow / len(rows)
    avg_delta_p = total_delta_p / len(rows)
    return avg_flow, avg_delta_p