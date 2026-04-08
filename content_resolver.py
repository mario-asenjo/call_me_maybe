#!/usr/bin/env python3
"""
content_resolver.py

Recorre todos los ficheros dentro de una carpeta indicada con --folder
y muestra por stdout el contenido de cada fichero con este formato:

[ruta/relativa/al/folder]:
Contenido del fichero
---

Permite excluir carpetas o ficheros por nombre usando --exclude.

Ejemplos:
    python content_resolver.py --folder=src
    python content_resolver.py --folder=src --exclude=__pycache__
    python content_resolver.py --folder=src --exclude=__pycache__,.git
    python content_resolver.py --folder=src --exclude=__pycache__ --exclude=.venv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hace cat de todos los ficheros dentro de una carpeta y muestra "
            "su contenido con la ruta relativa como cabecera."
        )
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Ruta base desde la que recorrer los ficheros.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Nombre a excluir. Puede repetirse varias veces o pasarse separado "
            "por comas. Ejemplo: --exclude=__pycache__,.git"
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding usado para leer los archivos. Por defecto: utf-8",
    )
    parser.add_argument(
        "--errors",
        default="replace",
        choices=["strict", "ignore", "replace"],
        help=(
            "Modo de gestión de errores de decodificación al leer archivos. "
            "Por defecto: replace"
        ),
    )
    return parser.parse_args()


def normalize_excludes(raw_excludes: Iterable[str]) -> Set[str]:
    excludes: Set[str] = set()

    for item in raw_excludes:
        if not item:
            continue
        for part in item.split(","):
            part = part.strip()
            if part:
                excludes.add(part)

    return excludes


def should_exclude(path: Path, excludes: Set[str]) -> bool:
    return path.name in excludes


def iter_files(base_folder: Path, excludes: Set[str]) -> Iterable[Path]:
    """
    Recorre recursivamente la carpeta base devolviendo solo archivos,
    excluyendo cualquier archivo o carpeta cuyo nombre esté en excludes.
    """
    for entry in sorted(base_folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if should_exclude(entry, excludes):
            continue

        if entry.is_dir():
            yield from iter_files(entry, excludes)
        elif entry.is_file():
            yield entry


def print_file_content(file_path: Path, base_folder: Path, encoding: str, errors: str) -> None:
    relative_path = file_path.relative_to(base_folder)

    print(f"[{relative_path.as_posix()}]:")
    try:
        content = file_path.read_text(encoding=encoding, errors=errors)
        # Evita añadir líneas extra innecesarias si ya termina en salto de línea
        if content:
            print(content, end="" if content.endswith("\n") else "\n")
    except Exception as exc:
        print(f"[ERROR leyendo archivo: {exc}]")
    print("---")


def main() -> int:
    args = parse_args()

    base_folder = Path(args.folder).expanduser().resolve()
    excludes = normalize_excludes(args.exclude)

    if not base_folder.exists():
        print(f"Error: la carpeta no existe: {base_folder}", file=sys.stderr)
        return 1

    if not base_folder.is_dir():
        print(f"Error: la ruta indicada no es una carpeta: {base_folder}", file=sys.stderr)
        return 1

    try:
        files_found = False
        for file_path in iter_files(base_folder, excludes):
            files_found = True
            print_file_content(
                file_path=file_path,
                base_folder=base_folder,
                encoding=args.encoding,
                errors=args.errors,
            )

        if not files_found:
            print("No se encontraron archivos para mostrar.", file=sys.stderr)

        return 0

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error inesperado: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())