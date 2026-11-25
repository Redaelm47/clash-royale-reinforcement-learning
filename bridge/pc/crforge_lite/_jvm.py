"""Starting the JVM with the env module's jars on its class path.

The jars come from one place: the env module's installed distribution, which
``./gradlew :env:installDist`` lays out under ``env/build/install/env`` (its ``lib`` folder holds
the env module, the battle core and the libraries they use). Every entry point of the package takes
that folder as its ``install`` argument; nothing is read from environment variables and there is no
default folder. The JVM starts once per process (JPype cannot restart it), so every later call must
name the same folder.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jpype

_started_with: Path | None = None


def jars(install: str | Path) -> list[str]:
    """The jars of an installed env distribution, sorted by name.

    :param install: the folder ``./gradlew :env:installDist`` makes (``env/build/install/env``)
    :raises FileNotFoundError: when the folder holds no jars under ``lib``
    """
    folder = Path(install).expanduser().resolve()
    found = sorted(str(jar) for jar in (folder / "lib").glob("*.jar"))
    if not found:
        raise FileNotFoundError(
            f"no jars under {folder / 'lib'}: install must name the env module's installed "
            "distribution, made by ./gradlew :env:installDist (env/build/install/env)"
        )
    return found


def start(install: str | Path, jvm_args: Sequence[str] = ()) -> None:
    """Starts the JVM once with the jars of ``install``; a later call with the same folder does
    nothing, one with another folder is refused.

    :param install: the env module's installed distribution (``env/build/install/env``)
    :param jvm_args: options for the JVM, for example ``["-Xmx4g"]``; only the first start uses
        them
    """
    global _started_with
    folder = Path(install).expanduser().resolve()
    if jpype.isJVMStarted():
        if _started_with is None:
            raise RuntimeError("the JVM was started outside crforge_env: its jars are unknown")
        if folder != _started_with:
            raise RuntimeError(
                f"the JVM runs the jars of {_started_with}; it cannot be restarted with {folder}"
            )
        return
    jpype.startJVM(find_jvm(), *jvm_args, classpath=jars(folder), convertStrings=False)
    _started_with = folder


def find_jvm() -> str:
    """The JVM path (added for the PC kit): JPype's, else JAVA_HOME, else the java on the PATH,
    else the usual Windows install folders (Temurin, Oracle, Microsoft)."""
    import glob
    import os
    import shutil
    try:
        return jpype.getDefaultJVMPath()
    except Exception:
        pass
    names = ["bin/server/jvm.dll", "jre/bin/server/jvm.dll", "lib/server/libjvm.so", "lib/server/libjvm.dylib"]
    homes = []
    if os.environ.get("JAVA_HOME"):
        homes.append(Path(os.environ["JAVA_HOME"]))
    java = shutil.which("java")
    if java:
        homes.append(Path(java).resolve().parent.parent)
    for pattern in (r"C:\Program Files\Eclipse Adoptium\*", r"C:\Program Files\Java\*",
                    r"C:\Program Files\Microsoft\jdk*", r"C:\Program Files\Zulu\*"):
        homes += [Path(h) for h in sorted(glob.glob(pattern), reverse=True)]
    for home in homes:
        for n in names:
            if (home / n).exists():
                return str(home / n)
    raise FileNotFoundError("Java introuvable : installe Java 21 (voir LISEZMOI)")


def java_class(name: str):
    """A Java class by its binary name; the JVM must have been started."""
    if not jpype.isJVMStarted():
        raise RuntimeError("the JVM is not started: give the installed env distribution first")
    return jpype.JClass(name)
