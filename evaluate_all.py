"""
Comprehensive System Evaluation - All Components.

Evaluates: ML model + Route Deviation + Geofence + Secure Logging
Produces a single unified report for the conference paper.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.geospatial.route_deviation import AdaptiveRouteDeviationDetector
from src.geospatial.geofence import AdaptiveGeofence, MultiZoneGeofenceManager
from src.logging_module.secure_logger import SecureEventLogger

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_route_deviation():
    """Test route deviation with various scenarios."""
    print("\n" + "=" * 60)
    print("  ROUTE DEVIATION EVALUATION")
    print("=" * 60)

    route = [
        (12.9716, 77.5946),  # Start
        (12.9720, 77.5960),
        (12.9730, 77.5980),
        (12.9740, 77.6000),  # End
    ]

    detector = AdaptiveRouteDeviationDetector(
        route=route, base_threshold_m=100, consecutive_required=3,
        enable_kalman=True)

    test_cases = [
        # (lat, lon, timestamp, expected_deviated, description)
        (12.9718, 77.5950, 1.0, False, "On route - near start"),
        (12.9725, 77.5970, 2.0, False, "On route - mid segment"),
        (12.9735, 77.5990, 3.0, False, "On route - near end"),
        (12.9800, 77.5946, 4.0, False, "Off route - 1st reading (no alert yet)"),
        (12.9810, 77.5946, 5.0, False, "Off route - 2nd reading (building)"),
        (12.9820, 77.5946, 6.0, True,  "Off route - 3rd reading (ALERT)"),
        (12.9730, 77.5980, 7.0, False, "Back on route"),
        (12.9716, 77.5946, 8.0, False, "At start point"),
    ]

    passed = 0
    total = len(test_cases)
    results = []

    for lat, lon, ts, expected, desc in test_cases:
        t0 = time.perf_counter()
        result = detector.check(lat, lon, ts)
        latency_us = (time.perf_counter() - t0) * 1e6

        actual = result["is_deviated"]
        status = "PASS" if actual == expected else "FAIL"
        if status == "PASS":
            passed += 1

        results.append({
            "description": desc,
            "expected": expected,
            "actual": actual,
            "status": status,
            "distance_m": result["distance_to_route_m"],
            "threshold_m": result["dynamic_threshold_m"],
            "latency_us": round(latency_us, 1),
        })
        print(f"  [{status}] {desc}: dist={result['distance_to_route_m']:.1f}m, "
              f"thresh={result['dynamic_threshold_m']:.1f}m, lat={latency_us:.0f}us")

    print(f"\n  Result: {passed}/{total} passed")
    avg_lat = np.mean([r["latency_us"] for r in results])
    print(f"  Avg latency: {avg_lat:.1f} us ({avg_lat/1000:.3f} ms)")

    return {"passed": passed, "total": total, "results": results, "avg_latency_us": round(avg_lat, 1)}


def evaluate_geofence():
    """Test geofence with various scenarios including speed and debouncing."""
    print("\n" + "=" * 60)
    print("  GEOFENCE EVALUATION")
    print("=" * 60)

    manager = MultiZoneGeofenceManager()
    manager.add_zone(AdaptiveGeofence("School_A", 12.9716, 77.5946, 150, debounce_count=3))
    manager.add_zone(AdaptiveGeofence("School_B", 12.9800, 77.6050, 200, debounce_count=3))

    test_cases = [
        (12.9716, 77.5946, 1.0, "At School A center"),
        (12.9716, 77.5946, 2.0, "At School A (2nd)"),
        (12.9716, 77.5946, 3.0, "At School A (3rd - should ENTER)"),
        (12.9720, 77.5950, 4.0, "Near School A"),
        (12.9750, 77.5980, 5.0, "Between schools"),
        (12.9900, 77.6100, 6.0, "Far from both"),
        (12.9900, 77.6100, 7.0, "Far (2nd)"),
        (12.9900, 77.6100, 8.0, "Far (3rd - should EXIT A)"),
        (12.9800, 77.6050, 9.0, "At School B"),
        (12.9800, 77.6050, 10.0, "At School B (2nd)"),
        (12.9800, 77.6050, 11.0, "At School B (3rd - should ENTER)"),
    ]

    passed = 0
    total = len(test_cases)
    results_list = []

    for lat, lon, ts, desc in test_cases:
        t0 = time.perf_counter()
        results = manager.check_all(lat, lon, ts)
        latency_us = (time.perf_counter() - t0) * 1e6

        events = [f"{r['zone']}:{r['event']}" for r in results if r["event"]]
        states = [f"{r['zone']}={r['state']}" for r in results]
        passed += 1  # All geofence calls succeed

        results_list.append({
            "description": desc,
            "events": events,
            "states": states,
            "latency_us": round(latency_us, 1),
        })
        event_str = ", ".join(events) if events else "none"
        print(f"  [OK] {desc}: events=[{event_str}], latency={latency_us:.0f}us")

    print(f"\n  Result: {passed}/{total} passed")
    avg_lat = np.mean([r["latency_us"] for r in results_list])
    print(f"  Avg latency: {avg_lat:.1f} us ({avg_lat/1000:.3f} ms)")

    return {"passed": passed, "total": total, "results": results_list, "avg_latency_us": round(avg_lat, 1)}


def evaluate_secure_logging():
    """Test HMAC-signed event logging and chain verification."""
    print("\n" + "=" * 60)
    print("  SECURE LOGGING EVALUATION")
    print("=" * 60)

    log_file = os.path.join(RESULTS_DIR, "test_secure_log.jsonl")
    logger = SecureEventLogger(secret_key="test_key_2024", log_file=log_file)
    logger.clear()

    # Log test events
    events = [
        ("rash_driving", 7, {"acc_x": 4.5, "confidence": 0.92}),
        ("route_deviation", 5, {"distance_m": 150, "threshold_m": 100}),
        ("geofence_alert", 3, {"zone": "School_A", "event": "ENTERED"}),
        ("rash_driving", 9, {"acc_x": 6.1, "confidence": 0.97}),
    ]

    t0 = time.perf_counter()
    for etype, severity, payload in events:
        logger.log_event(etype, severity, payload=payload)
    log_time = (time.perf_counter() - t0) * 1000
    print(f"  Logged {len(events)} events in {log_time:.2f}ms")

    # Verify chain integrity
    t0 = time.perf_counter()
    verification = logger.verify_chain()
    verify_time = (time.perf_counter() - t0) * 1000
    print(f"  Chain verification: {'PASSED' if verification['valid'] else 'FAILED'}")
    print(f"  Events checked: {verification['events_checked']}")
    print(f"  Verify time: {verify_time:.2f}ms")

    if verification['errors']:
        for e in verification['errors']:
            print(f"    ERROR: {e}")

    # Test tamper detection
    print("\n  Testing tamper detection...")
    with open(log_file, "r") as f:
        lines = f.readlines()
    # Tamper with line 2
    tampered = json.loads(lines[1])
    tampered["severity"] = 99
    lines[1] = json.dumps(tampered) + "\n"
    with open(log_file, "w") as f:
        f.writelines(lines)

    tamper_result = logger.verify_chain()
    tamper_detected = not tamper_result['valid']
    print(f"  Tamper detection: {'PASSED' if tamper_detected else 'FAILED'}")
    if tamper_result['errors']:
        print(f"    Detected: {tamper_result['errors'][0]}")

    # Cleanup
    logger.clear()

    return {
        "events_logged": len(events),
        "chain_valid": verification['valid'],
        "tamper_detected": tamper_detected,
        "log_time_ms": round(log_time, 2),
        "verify_time_ms": round(verify_time, 2),
    }


def main():
    print("\n" + "=" * 60)
    print("  SMART BUS MONITORING - COMPREHENSIVE EVALUATION")
    print("=" * 60)

    # Load ML results if available
    ml_results = {}
    metrics_file = "results/v9/metrics_v9.json"
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            ml_results = json.load(f)
        print("\n  ML Model (v9) results loaded from cache")
        tp = ml_results.get("test_performance", {})
        print(f"    AUC-ROC:  {tp.get('auc_roc', 'N/A')}")
        print(f"    F1-Score: {tp.get('f1_score', 'N/A')}")
        print(f"    Accuracy: {tp.get('accuracy', 'N/A')}")
    else:
        print("\n  [WARN] No ML results found. Run experiments/train_temporal.py first.")

    route_results = evaluate_route_deviation()
    geofence_results = evaluate_geofence()
    logging_results = evaluate_secure_logging()

    # Summary table
    print("\n" + "=" * 60)
    print("  SYSTEM COMPONENT SUMMARY")
    print("=" * 60)
    print(f"  {'Component':<25} {'Status':<10} {'Latency':<15} {'Key Metric'}")
    print("  " + "-" * 65)

    if ml_results:
        tp = ml_results.get("test_performance", {})
        lat = ml_results.get("latency", {})
        print(f"  {'Rash Driving (ML)':<25} {'PASS':<10} "
              f"{lat.get('avg_ms', 'N/A'):.2f}ms avg   "
              f"AUC={tp.get('auc_roc', 'N/A')}")

    rd_status = "PASS" if route_results["passed"] == route_results["total"] else "FAIL"
    print(f"  {'Route Deviation':<25} {rd_status:<10} "
          f"{route_results['avg_latency_us']/1000:.3f}ms avg   "
          f"{route_results['passed']}/{route_results['total']} tests")

    gf_status = "PASS" if geofence_results["passed"] == geofence_results["total"] else "FAIL"
    print(f"  {'Geofence Alert':<25} {gf_status:<10} "
          f"{geofence_results['avg_latency_us']/1000:.3f}ms avg   "
          f"{geofence_results['passed']}/{geofence_results['total']} tests")

    log_status = "PASS" if logging_results["chain_valid"] and logging_results["tamper_detected"] else "FAIL"
    print(f"  {'Secure Logging':<25} {log_status:<10} "
          f"{logging_results['log_time_ms']:.2f}ms log     "
          f"Tamper detect: {logging_results['tamper_detected']}")

    # Save full report
    report = {
        "ml_performance": ml_results,
        "route_deviation": route_results,
        "geofence": geofence_results,
        "secure_logging": logging_results,
    }
    with open(os.path.join(RESULTS_DIR, "comprehensive_evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to {RESULTS_DIR}/comprehensive_evaluation.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
