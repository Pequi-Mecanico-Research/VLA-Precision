"""Garante que o dataset LeRobot esteja no formato v3.0 antes de o VLA-Precision ler.

O VLA-Precision fixa uma versão do LeRobot que só lê datasets v3.0. O OpenPI usa uma versão que
lê v2.1, e por isso os datasets da equipe ficam em v2.1. Este módulo faz a ponte:

1. acha o dataset (pasta local de ``data.lerobot_root`` ou o Hugging Face Hub);
2. lê a versão em ``meta/info.json``;
3. se for v3.0, usa direto;
4. se for v2.1, converte para uma **cópia local separada** em
   ``<paths.cache_root>/lerobot_v30/<repo_id>`` e usa a cópia. Nas próximas execuções, a cópia é
   reaproveitada enquanto o original não mudar.

Garantias: o dataset original **nunca** é modificado e **nada** é enviado para o Hub. A conversão
usa as funções do conversor oficial do próprio LeRobot fixado no projeto, então o resultado é
garantidamente legível por ele.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from vla_precision.config.schema import RootConfig, Stage1Config
from vla_precision.data.paths import _safe_name, source_lerobot_root

LOGGER = logging.getLogger(__name__)

READABLE_VERSION = "v3.0"
CONVERTIBLE_VERSION = "v2.1"
MARKER_NAME = "vla_precision_conversion.json"
_FINGERPRINT_FILES = ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", "meta/episodes_stats.jsonl")


def dataset_codebase_version(root: Path) -> str:
    """Versão do formato gravada em ``meta/info.json`` (por exemplo ``"v2.1"`` ou ``"v3.0"``)."""
    info = json.loads((root / "meta" / "info.json").read_text())
    return str(info.get("codebase_version", "unknown"))


def converted_dataset_dir(config: RootConfig | Stage1Config) -> Path:
    """Onde fica a cópia convertida para v3.0."""
    return Path(config.paths.cache_root).expanduser() / "lerobot_v30" / _safe_name(config.data.lerobot_repo_id)


def downloaded_v21_dir(config: RootConfig | Stage1Config) -> Path:
    """Onde fica o original v2.1 baixado do Hub, quando não há pasta local."""
    return Path(config.paths.cache_root).expanduser() / "lerobot_v21" / _safe_name(config.data.lerobot_repo_id)


def lerobot_dataset_root(config: RootConfig | Stage1Config) -> Path | None:
    """Raiz do dataset pronta para o LeRobot do VLA-Precision (v3.0).

    Substitui ``source_lerobot_root`` nos pontos que abrem o dataset. Retorna ``None`` quando o
    próprio LeRobot pode baixar o dataset do Hub sem conversão (o Hub já tem v3.0).
    """
    root = source_lerobot_root(config)
    if root is None:
        root = _hub_source_root(config)
        if root is None:
            return None
    elif not (root / "meta" / "info.json").exists():
        # Pasta configurada, mas ainda sem dataset: mantém o comportamento original do LeRobot.
        return root

    version = dataset_codebase_version(root)
    if version == READABLE_VERSION:
        LOGGER.info("Dataset %s já está em %s: %s", config.data.lerobot_repo_id, version, root)
        return root
    if version != CONVERTIBLE_VERSION:
        raise ValueError(
            f"Dataset {config.data.lerobot_repo_id} em {root} tem formato {version!r}. O VLA-Precision lê "
            f"{READABLE_VERSION} e só converte automaticamente a partir de {CONVERTIBLE_VERSION}."
        )
    if not config.data.lerobot_auto_convert:
        raise ValueError(
            f"Dataset {config.data.lerobot_repo_id} em {root} está em {CONVERTIBLE_VERSION}, mas o "
            f"VLA-Precision lê {READABLE_VERSION} e data.lerobot_auto_convert=false. Ligue a opção ou "
            f"converta à mão para uma pasta separada e aponte data.lerobot_root para ela."
        )
    return _ensure_converted_copy(root, converted_dataset_dir(config), config.data.lerobot_repo_id)


def _hub_source_root(config: RootConfig | Stage1Config) -> Path | None:
    """Sem pasta local: consulta a versão no Hub e, se for v2.1, baixa o original para o cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = config.data.lerobot_repo_id
    local = downloaded_v21_dir(config)
    if (local / "meta" / "info.json").exists():
        return local

    info_path = Path(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset"))
    version = str(json.loads(info_path.read_text()).get("codebase_version", "unknown"))
    if version == READABLE_VERSION:
        return None
    LOGGER.warning("Dataset %s no Hub está em %s; baixando o original para %s", repo_id, version, local)
    snapshot_download(repo_id, repo_type="dataset", local_dir=local)
    return local


def _source_fingerprint(root: Path) -> str:
    """Hash dos metadados do original: muda se o dataset original for regravado."""
    digest = hashlib.sha256()
    for name in _FINGERPRINT_FILES:
        path = root / name
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_marker(path: Path) -> dict | None:
    marker = path / MARKER_NAME
    if not marker.exists():
        return None
    return json.loads(marker.read_text())


def _ensure_converted_copy(source: Path, target: Path, repo_id: str) -> Path:
    fingerprint = _source_fingerprint(source)
    marker = _read_marker(target)
    if (
        marker is not None
        and marker.get("source_fingerprint") == fingerprint
        and dataset_codebase_version(target) == READABLE_VERSION
    ):
        LOGGER.info("Usando a cópia v3.0 já convertida de %s: %s", repo_id, target)
        return target

    if target.exists() and marker is None:
        # Nunca apaga uma pasta que não foi criada por este módulo.
        raise FileExistsError(
            f"{target} já existe e não tem {MARKER_NAME}; não foi criada pelo VLA-Precision. "
            f"Mova ou apague essa pasta à mão antes de converter."
        )

    LOGGER.warning(
        "Dataset %s está em %s; convertendo uma cópia para %s em %s (o original não é alterado)",
        repo_id,
        CONVERTIBLE_VERSION,
        READABLE_VERSION,
        target,
    )
    work = target.with_name(target.name + ".tmp")
    if work.exists():
        shutil.rmtree(work)  # sobra de uma conversão interrompida deste módulo
    work.mkdir(parents=True)
    _convert_v21_to_v30(source, work)
    (work / MARKER_NAME).write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "source_root": str(source),
                "source_version": CONVERTIBLE_VERSION,
                "source_fingerprint": fingerprint,
                "converted_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    if target.exists():
        shutil.rmtree(target)  # cópia antiga deste módulo (tem o marcador), desatualizada
    work.rename(target)
    LOGGER.warning("Conversão concluída: %s", target)
    return target


def _convert_v21_to_v30(source: Path, target: Path) -> None:
    """Etapas do conversor oficial do LeRobot, lendo de ``source`` e escrevendo em ``target``.

    Não usa ``convert_dataset`` do LeRobot de propósito: aquela função trabalha no lugar (renomeia
    e apaga pastas) e, por padrão, envia o resultado para o Hub.
    """
    from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB
    from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as official

    official.convert_info(source, target, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    official.convert_tasks(source, target)
    episodes_metadata = official.convert_data(source, target, DEFAULT_DATA_FILE_SIZE_IN_MB)
    episodes_videos_metadata = official.convert_videos(source, target, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    official.convert_episodes_metadata(source, target, episodes_metadata, episodes_videos_metadata)
