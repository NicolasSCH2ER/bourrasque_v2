"""test_lib.py — verification autonome de `extension/lib.py`.

Executable sans Blender : `python extension/tests/test_lib.py`.

Se concentre sur le garde-fou de compatibilite ABI (`_check_abi_compat`),
teste avec des objets factices, sans dependre d'une DLL native reelle.
"""

import ctypes
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
    fake.bq_mesher_config_size = lambda: __import__("ctypes").sizeof(lib.BqMesherConfig)
    fake.bq_whitewater_config_size = lambda: __import__("ctypes").sizeof(
        lib.BqWhitewaterConfig
    )
    fake.bq_rigid_body_size = lambda: __import__("ctypes").sizeof(lib.BqRigidBody)
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


def test_whitewater_config_has_expected_size():
    import ctypes

    # Meme motif que BqMesherConfig : la taille cote Python n'a pas
    # d'oracle sans DLL native, ce test fige au moins que la struct
    # ctypes se construit et a une taille non nulle, coherente avec le
    # nombre de champs declares.
    size = ctypes.sizeof(lib.BqWhitewaterConfig)
    n_fields = len(lib.BqWhitewaterConfig._fields_)
    assert size >= n_fields * 4, (size, n_fields)


class _FakeWhitewaterDll:
    """DLL factice pour tester `Whitewater` sans bibliotheque native."""

    def __init__(self):
        self.destroy_calls = 0

    def bq_whitewater_create(self, cfg_ptr):
        return ctypes.cast(1, lib.BqWhitewaterPtr)

    def bq_whitewater_destroy(self, handle):
        self.destroy_calls += 1

    def bq_last_error(self):
        return b"erreur factice"


def test_whitewater_create_and_close_is_idempotent():
    import ctypes as _ctypes

    fake_dll = _FakeWhitewaterDll()
    ww = lib.Whitewater.__new__(lib.Whitewater)
    ww._dll = fake_dll
    cfg = lib.BqWhitewaterConfig()
    ww._handle = fake_dll.bq_whitewater_create(_ctypes.byref(cfg))

    assert ww._handle
    ww.close()
    assert fake_dll.destroy_calls == 1
    ww.close()  # doit etre un no-op, pas un second appel natif
    assert fake_dll.destroy_calls == 1


def test_whitewater_create_raises_on_null_handle():
    class _NullDll:
        def bq_whitewater_create(self, cfg_ptr):
            return None

        def bq_last_error(self):
            return b"VRAM insuffisante"

    dll = _NullDll()
    cfg = lib.BqWhitewaterConfig()

    raised = False
    try:
        ww = lib.Whitewater.__new__(lib.Whitewater)
        ww._dll = dll
        handle = dll.bq_whitewater_create(ctypes.byref(cfg))
        if not handle:
            raise lib.BourrasqueError(
                f"Impossible de creer le whitewater Bourrasque : "
                f"{lib._last_error(dll)}"
            )
    except lib.BourrasqueError as exc:
        raised = True
        assert "VRAM insuffisante" in str(exc)
    assert raised, "la creation aurait du lever BourrasqueError"


def main():
    check("abi_ok_does_not_raise", test_abi_ok_does_not_raise)
    check("missing_abi_version_function_raises", test_missing_abi_version_function_raises)
    check("abi_version_mismatch_raises", test_abi_version_mismatch_raises)
    check("config_size_mismatch_raises", test_config_size_mismatch_raises)
    check("missing_config_size_function_raises", test_missing_config_size_function_raises)
    check("whitewater_config_has_expected_size", test_whitewater_config_has_expected_size)
    check(
        "whitewater_create_and_close_is_idempotent",
        test_whitewater_create_and_close_is_idempotent,
    )
    check(
        "whitewater_create_raises_on_null_handle",
        test_whitewater_create_raises_on_null_handle,
    )

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) en echec.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
