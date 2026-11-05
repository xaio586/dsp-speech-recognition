"""Check that the first-week Python dependencies can be imported."""

import importlib
import platform

PACKAGES = ["numpy", "scipy", "matplotlib", "soundfile", "pandas", "sounddevice"]


def main() -> None:
    print(f"Python: {platform.python_version()}")
    failed = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
        except Exception as error:  # pragma: no cover - used for environment diagnosis
            failed.append((package, str(error)))
            print(f"[FAIL] {package}: {error}")
        else:
            version = getattr(module, "__version__", "installed")
            print(f"[ OK ] {package}: {version}")
    if failed:
        raise SystemExit(1)
    print("Environment check passed.")


if __name__ == "__main__":
    main()

