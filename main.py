#!/usr/bin/env python3
"""
main.py — Ontology-Guided Counterfactual Affordance Reasoning
          for Actionable Robot Navigation Explanations

Entry point for running the demonstration scenarios.

Usage
-----
  python3 main.py               # run both hospital and warehouse scenarios
  python3 main.py hospital      # hospital scenario only (6 sub-scenarios)
  python3 main.py warehouse     # warehouse scenario only (4 sub-scenarios)

The hospital scenario demonstrates:
  • multi-robot capability differentiation (arm / no-arm, wide / narrow body)
  • door-opening affordance (delivery_bot opens ICU door; cargo_bot cannot)
  • width-based blocking (cargo_bot too wide for narrow corridor)
  • access restriction (staff corridor blocks robots without permission)
  • live world mutation (propping a door open reduces path cost by 2.5 units)
  • per-robot improvement roadmap (infrastructure changes that unlock routes)

The warehouse scenario demonstrates:
  • surface-type friction cost (wet floor adds 1.4 to edge weight)
  • illumination-based observation penalty (dark zone adds 1.8 uncertainty cost)
  • slope-angle blocking (18° ramp: wheeled blocked, tracked allowed)
  • width-based aisle restriction (1.4 m forklift cannot use 0.85 m narrow aisle)

No external dependencies — requires only Python 3.8+ stdlib.
"""

import sys


def main() -> None:
    # Parse which scenarios to run from the command line.
    # Default (no args) = run both.
    args = sys.argv[1:]
    run_hospital  = not args or "hospital"  in args
    run_warehouse = not args or "warehouse" in args

    if run_hospital:
        print("\n" + "=" * 72)
        print("  HOSPITAL NAVIGATION SCENARIO")
        print("=" * 72)
        from ontofact_nav.scenarios.hospital  import run as hospital_run
        hospital_run()

    if run_warehouse:
        print("\n" + "=" * 72)
        print("  WAREHOUSE NAVIGATION SCENARIO")
        print("=" * 72)
        from ontofact_nav.scenarios.warehouse import run as warehouse_run
        warehouse_run()


if __name__ == "__main__":
    main()
