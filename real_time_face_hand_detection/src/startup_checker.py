"""Startup validation checks for the real-time detection app."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class CheckResult:
    """Single startup check result."""

    name: str
    passed: bool
    details: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _emit(result: CheckResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name}: {result.details}")


def _check_project_files() -> CheckResult:
    root = _project_root()
    required_paths = [
        root / "config",
        root / "config" / "settings.py",
        root / "src",
        root / "src" / "main.py",
        root / "src" / "detectors",
        root / "src" / "detectors" / "face_detector.py",
        root / "src" / "detectors" / "hand_detector.py",
        root / "src" / "utils",
        root / "requirements.txt",
    ]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        return CheckResult(
            name="Project structure",
            passed=False,
            details=f"Missing required paths: {', '.join(missing)}",
        )

    return CheckResult(name="Project structure", passed=True, details="All required folders/files exist")


def _check_import(name: str) -> Tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, f"Import failed: {exc}"

    module_path = getattr(module, "__file__", "built-in")
    return True, f"Imported successfully from {module_path}"


def _check_python_and_deps() -> List[CheckResult]:
    results: List[CheckResult] = []

    ok_cv2, msg_cv2 = _check_import("cv2")
    results.append(CheckResult(name="Dependency import: cv2", passed=ok_cv2, details=msg_cv2))

    ok_mp, msg_mp = _check_import("mediapipe")
    results.append(CheckResult(name="Dependency import: mediapipe", passed=ok_mp, details=msg_mp))

    return results


def _check_mediapipe_api() -> List[CheckResult]:
    results: List[CheckResult] = []

    spec = importlib.util.find_spec("mediapipe")
    if spec is None:
        results.append(
            CheckResult(
                name="MediaPipe package discovery",
                passed=False,
                details="Package 'mediapipe' is not installed in the active interpreter",
            )
        )
        return results

    try:
        import mediapipe as mp
    except Exception as exc:
        results.append(CheckResult(name="MediaPipe import", passed=False, details=f"Import failed: {exc}"))
        return results

    mp_file = Path(getattr(mp, "__file__", ""))
    project_root = _project_root()
    if mp_file and project_root in mp_file.parents:
        results.append(
            CheckResult(
                name="MediaPipe module shadowing",
                passed=False,
                details=f"Local module is shadowing pip package: {mp_file}",
            )
        )
    else:
        results.append(
            CheckResult(
                name="MediaPipe module shadowing",
                passed=True,
                details=f"No local shadowing detected (module path: {mp_file})",
            )
        )

    root_solutions = getattr(mp, "solutions", None)
    has_root = root_solutions is not None
    results.append(
        CheckResult(
            name="MediaPipe root API (mediapipe.solutions)",
            passed=has_root,
            details="Available" if has_root else "Not exposed at package root",
        )
    )

    fallback_available = False
    try:
        from mediapipe.python import solutions as mp_solutions

        fallback_available = True
    except Exception as exc:
        results.append(
            CheckResult(
                name="MediaPipe fallback API (mediapipe.python.solutions)",
                passed=False,
                details=f"Unavailable: {exc}",
            )
        )
    else:
        results.append(
            CheckResult(
                name="MediaPipe fallback API (mediapipe.python.solutions)",
                passed=True,
                details="Available",
            )
        )

        results.append(
            CheckResult(
                name="MediaPipe face_detection solution",
                passed=hasattr(mp_solutions, "face_detection"),
                details="Available" if hasattr(mp_solutions, "face_detection") else "Missing",
            )
        )
        results.append(
            CheckResult(
                name="MediaPipe hands solution",
                passed=hasattr(mp_solutions, "hands"),
                details="Available" if hasattr(mp_solutions, "hands") else "Missing",
            )
        )

    if has_root:
        results.append(
            CheckResult(
                name="MediaPipe root face_detection",
                passed=hasattr(root_solutions, "face_detection"),
                details="Available" if hasattr(root_solutions, "face_detection") else "Missing",
            )
        )
        results.append(
            CheckResult(
                name="MediaPipe root hands",
                passed=hasattr(root_solutions, "hands"),
                details="Available" if hasattr(root_solutions, "hands") else "Missing",
            )
        )

    if not has_root and not fallback_available:
        results.append(
            CheckResult(
                name="MediaPipe solutions availability",
                passed=False,
                details="Neither root nor fallback solutions API is available",
            )
        )

    return results


def _check_camera() -> CheckResult:
    try:
        import cv2
        from config import settings
    except Exception as exc:
        return CheckResult(name="Camera check", passed=False, details=f"Prerequisite import failed: {exc}")

    cap = cv2.VideoCapture(settings.CAMERA_ID)
    try:
        if not cap.isOpened():
            return CheckResult(
                name="Camera check",
                passed=False,
                details=f"Could not open CAMERA_ID={settings.CAMERA_ID}",
            )

        ok, _ = cap.read()
        if not ok:
            return CheckResult(
                name="Camera frame read",
                passed=False,
                details="Camera opened but failed to read a frame",
            )

        return CheckResult(
            name="Camera check",
            passed=True,
            details=f"Camera opened and frame read succeeded (CAMERA_ID={settings.CAMERA_ID})",
        )
    finally:
        cap.release()


def _check_detector_imports() -> List[CheckResult]:
    results: List[CheckResult] = []
    for module_name in ("detectors.face_detector", "detectors.hand_detector"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            results.append(
                CheckResult(name=f"Detector import: {module_name}", passed=False, details=f"Import failed: {exc}")
            )
        else:
            results.append(
                CheckResult(name=f"Detector import: {module_name}", passed=True, details="Imported successfully")
            )
    return results


def run_startup_checks() -> bool:
    """Run startup checks one-by-one and print PASS/FAIL for each."""
    print("Running startup checks...\n")

    all_results: List[CheckResult] = []
    all_results.append(_check_project_files())
    all_results.extend(_check_python_and_deps())
    all_results.extend(_check_mediapipe_api())
    all_results.append(_check_camera())
    all_results.extend(_check_detector_imports())

    for result in all_results:
        _emit(result)

    failed = [result for result in all_results if not result.passed]

    print("\nStartup check summary:")
    print(f"- Total checks: {len(all_results)}")
    print(f"- Passed: {len(all_results) - len(failed)}")
    print(f"- Failed: {len(failed)}")

    if failed:
        print("\nBlocking failures detected. Fix failed checks before running detection.")
        return False

    print("\nAll startup checks passed. Launching real-time detection.")
    return True
