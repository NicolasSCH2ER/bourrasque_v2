"""test_lib.py — verification autonome de `extension/lib.py`.

Executable sans Blender : `python extension/tests/test_lib.py`.

Se concentre sur le garde-fou de compatibilite ABI (`_check_abi_compat`),
teste avec des objets factices, sans dependre d'une DLL native reelle.
"""

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import lib  # noqa: E402

_FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append((name, exc))
        print(f"[FAIL] {name}: {exc!r}")
    else:
        print(f"[ OK ] {name}")


def test_abi_ok_does_not_raise():
    fake = types.SimpleNamespace()
    fake.bq_abi_version = lambda: lib.BQ_ABI_VERSION
    fake.bq_config_size = lambda: __import__("ctypes").sizeof(lib.BqConfig)
    lib._check_abi_compat(fake)  # ne doit rien lever


def test_missing_abi_version_function_raises():
    fake = types.SimpleNamespace()  # pas de bq_abi_version -> AttributeError

    raised = False
    try:
        lib._check_abi_compat(fake)
    except lib.BourrasqueError as exc:
        raised = True
        msg = str(exc)
        assert "perimee" in msg, msg
        assert "bq_abi_version" in msg, msg
        assert "REDEMARREZ BLENDER" in msg, msg
        assert str(lib.BQ_ABI_VERSION) in msg, msg
    assert raised, "_check_abi_compat aurait du lever BourrasqueError"


def test_abi_version_mismatch_raises():
    fake = types.SimpleNamespace()
    fake.bq_abi_version = lambda: lib.BQ_ABI_VERSION + 1
    fake.bq_config_size = lambda: __import__("ctypes").sizeof(lib.BqConfig)

    raised = False
    try:
        lib._check_abi_compat(fake)
    except lib.BourrasqueError as exc:
        raised = True
        msg = str(exc)
        assert "incompatible" in msg, msg
        assert str(lib.BQ_ABI_VERSION + 1) in msg, msg
        assert str(lib.BQ_ABI_VERSION) in msg, msg
        assert "REDEMARREZ BLENDER" in msg, msg
    assert raised, "_check_abi_compat aurait du lever BourrasqueError"


def test_config_size_mismatch_raises():
    import ctypes

    fake = types.SimpleNamespace()
    fake.bq_abi_version = lambda: lib.BQ_ABI_VERSION
    fake.bq_config_size = lambda: ctypes.sizeof(lib.BqConfig) - 8  # taille perimee

    raised = False
    try:
        lib._check_abi_compat(fake)
    except lib.BourrasqueError as exc:
        raised = True
        msg = str(exc)
        assert "BqConfig" in msg, msg
        assert str(ctypes.sizeof(lib.BqConfig)) in msg, msg
        assert "REDEMARREZ BLENDER" in msg, msg
    assert raised, "_check_abi_compat aurait du lever BourrasqueError"


def test_missing_config_size_function_raises():
    fake = types.SimpleNamespace()
    fake.bq_abi_version = lambda: lib.BQ_ABI_VERSION
    # pas de bq_config_size -> AttributeError

    raised = False
    try:
        lib._check_abi_compat(fake)
    except lib.BourrasqueError as exc:
        raised = True
        msg = str(exc)
        assert "perimee" in msg, msg
        assert "bq_config_size" in msg, msg
        assert "REDEMARREZ BLENDER" in msg, msg
    assert raised, "_check_abi_compat aurait du lever BourrasqueError"


def main():
    check("abi_ok_does_not_raise", test_abi_ok_does_not_raise)
    check("missing_abi_version_function_raises", test_missing_abi_version_function_raises)
    check("abi_version_mismatch_raises", test_abi_version_mismatch_raises)
    check("config_size_mismatch_raises", test_config_size_mismatch_raises)
    check("missing_config_size_function_raises", test_missing_config_size_function_raises)

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) en echec.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
