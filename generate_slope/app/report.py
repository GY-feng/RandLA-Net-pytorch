import numpy as np


def build_report():
    return {
        "files": [],
        "summary": {
            "total_files": 0,
            "total_defects": 0,
            "total_points": 0,
            "total_abnormal_points": 0,
        },
    }


def add_file_report(report, item):
    report["files"].append(item)
    report["summary"]["total_files"] += 1
    report["summary"]["total_defects"] += int(item.get("defect_count", 0))
    report["summary"]["total_points"] += int(item.get("total_points", 0))
    report["summary"]["total_abnormal_points"] += int(item.get("abnormal_points", 0))


def ratio_str(numer, denom):
    if denom <= 0:
        return "0.00%"
    return f"{100.0 * float(numer) / float(denom):.2f}%"
