"""B3 — teste do modelo do WidowX dentro do VLA-Precision, sem robô.

Roda na máquina com GPU (a 5080), no ambiente do VLA-Precision:

    git submodule update --init third_party/openpi
    uv sync --frozen --group stage2
    uv run --no-sync python tools/widowx/check_stage1_model.py

Etapas (cada uma imprime PASSOU / FALHOU / ATENÇÃO):

1. Receitas: as duas receitas do WidowX (B2) existem e usam as transformações do fork.
2. Checkpoint: baixa só ``params/`` e ``assets/`` do Hugging Face (se ainda não estiver no disco)
   e acha os pesos e o ``norm_stats.json`` com as funções do próprio VLA-Precision.
3. Chaves: compara, sem carregar os pesos, as chaves do checkpoint com as do modelo. O OpenPI
   descarta sem aviso pesos que o modelo não tem; aqui isso vira erro explícito.
4. Dataset × norm stats: abre o dataset pelo VLA-Precision (convertendo para v3.0 se preciso) e
   compara média e desvio de estado e ação com o ``norm_stats.json`` do checkpoint. Se baterem, o
   VLA-Precision lê as mesmas colunas, na mesma ordem e escala do treino.
5. Inferência e memória: carrega o modelo na GPU, prevê ações para frames do dataset, compara com
   as ações gravadas e mede a memória de GPU usada.

Detalhes e interpretação: docs/widowx.md, passo B3.
"""

from __future__ import annotations

import os

# Sem isso o JAX reserva 75% da GPU de saída e a medição de memória não diz nada.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import dataclasses
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_HF_REPO = "rafaeljose/pi05_widowxai_organize_table_inf_b32_w16_20k"
DEFAULT_DATASET = "rafaeljose/table-inf_v21"
FULL_CONFIG = "pi05_full_finetune_widowx"
ACOB_CONFIG = "pi05_acob_widowx"

RESULTS: list[tuple[str, str, str]] = []


def report(step: str, status: str, detail: str) -> None:
    RESULTS.append((step, status, detail))
    print(f"  [{status}] {detail}")


def header(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def gib(num_bytes: float) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB"


def flatten(tree, prefix=()) -> dict[tuple[str, ...], object]:
    """Achata dicionários aninhados (inclusive os de metadados do Orbax) em {caminho: folha}."""
    out = {}
    for key, value in tree.items():
        path = (*prefix, str(key))
        if isinstance(value, Mapping):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


# ---------------------------------------------------------------------------------------------
# 1. Receitas
# ---------------------------------------------------------------------------------------------
def step_configs():
    header("[1/5] Receitas do WidowX no VLA-Precision (B2)")
    import jax
    from flax import nnx

    from vla_precision.integrations.openpi.configs import get_config

    full_cfg = get_config(FULL_CONFIG)
    acob_cfg = get_config(ACOB_CONFIG)
    for cfg in (full_cfg, acob_cfg):
        data_cls = type(cfg.data)
        tf = cfg.data.create(Path("/nonexistent"), cfg.model).data_transforms
        names = [f"{type(t).__module__}.{type(t).__name__}" for t in (*tf.inputs, *tf.outputs)]
        ok = data_cls.__module__.startswith("vla_precision.") and all(
            n.startswith("openpi.policies.widowx_policy.") for n in names
        )
        report(
            "1. receitas",
            "PASSOU" if ok else "FALHOU",
            f"{cfg.name}: classe de dados {data_cls.__module__}.{data_cls.__name__}; transformações {names}",
        )

    def count(cfg, filt=None):
        shape = nnx.eval_shape(cfg.model.create, jax.random.key(0))
        state = nnx.state(shape, filt) if filt is not None else nnx.state(shape)
        return sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(state.to_pure_dict()))

    total = count(acob_cfg)
    trainable = count(acob_cfg, acob_cfg.trainable_filter)
    report(
        "1. receitas",
        "INFO",
        f"{ACOB_CONFIG}: {total / 1e9:.2f}B parâmetros no total, {trainable / 1e6:.1f}M treináveis no Stage 2 "
        f"({100 * trainable / total:.2f}%: só os adaptadores LoRA do action expert)",
    )
    return full_cfg, acob_cfg


# ---------------------------------------------------------------------------------------------
# 2. Checkpoint
# ---------------------------------------------------------------------------------------------
def step_checkpoint(checkpoint_dir: Path, hf_repo: str, dataset: str):
    header("[2/5] Checkpoint")
    from vla_precision.integrations.openpi.checkpoints import resolve_openpi_assets, resolve_openpi_checkpoint

    if not (checkpoint_dir / "params").exists():
        from huggingface_hub import snapshot_download

        print(f"  Baixando só params/ e assets/ de {hf_repo} para {checkpoint_dir} (~12,4 GB)...")
        snapshot_download(
            hf_repo,
            local_dir=checkpoint_dir,
            allow_patterns=["params/*", "assets/*", "_CHECKPOINT_METADATA"],
        )
    checkpoint = resolve_openpi_checkpoint(str(checkpoint_dir))
    assets = resolve_openpi_assets(str(checkpoint_dir), default_asset_id=dataset)
    report("2. checkpoint", "PASSOU", f"pesos: {checkpoint.params_path}")
    if assets is None:
        report("2. checkpoint", "FALHOU", "nenhum norm_stats.json dentro do checkpoint")
        raise SystemExit(1)
    norm_stats = json.loads(Path(assets.norm_stats_path).read_text())["norm_stats"]
    dims = {k: len(v["mean"]) for k, v in norm_stats.items()}
    ok = dims.get("state") == 7 and dims.get("actions") == 7
    report(
        "2. checkpoint",
        "PASSOU" if ok else "FALHOU",
        f"norm stats: {assets.norm_stats_path} (asset_id={assets.asset_id}); dimensões {dims}",
    )
    return checkpoint, assets, norm_stats


# ---------------------------------------------------------------------------------------------
# 3. Chaves do checkpoint x modelo
# ---------------------------------------------------------------------------------------------
def step_keys(params_path: str, full_cfg, acob_cfg):
    header("[3/5] Chaves do checkpoint × modelo (sem carregar os pesos)")
    import jax
    import orbax.checkpoint as ocp
    from flax import nnx

    with ocp.PyTreeCheckpointer() as ckptr:
        tree = ckptr.metadata(Path(params_path).resolve())["params"]
        meta = flatten(getattr(tree, "tree", tree))  # algumas versões do Orbax embrulham a árvore
    if all(path[-1] == "value" for path in meta):
        meta = {path[:-1]: leaf for path, leaf in meta.items()}
    ckpt_shapes = {path: tuple(getattr(leaf, "shape", ())) for path, leaf in meta.items()}
    print(f"  Checkpoint: {len(ckpt_shapes)} tensores")

    lora = re.compile(".*lora.*")
    for cfg, missing_allowed in ((full_cfg, None), (acob_cfg, lora)):
        shape = nnx.eval_shape(cfg.model.create, jax.random.key(0))
        model = {path: tuple(leaf.shape) for path, leaf in flatten(nnx.state(shape).to_pure_dict()).items()}
        missing = sorted(set(model) - set(ckpt_shapes))
        unexpected = sorted(set(ckpt_shapes) - set(model))
        wrong = sorted(p for p in set(model) & set(ckpt_shapes) if model[p] != ckpt_shapes[p])
        bad_missing = [p for p in missing if not (missing_allowed and missing_allowed.fullmatch("/".join(p)))]
        ok = not bad_missing and not unexpected and not wrong
        detail = (
            f"{cfg.name}: {len(model)} tensores no modelo; {len(missing)} só no modelo"
            f"{' (todos LoRA, começam do zero: esperado)' if missing and not bad_missing else ''}; "
            f"{len(unexpected)} só no checkpoint; {len(wrong)} com formato diferente"
        )
        report("3. chaves", "PASSOU" if ok else "FALHOU", detail)
        for label, items in (("faltando (não-LoRA)", bad_missing), ("só no checkpoint", unexpected), ("formato", wrong)):
            for p in items[:5]:
                print(f"      {label}: {'/'.join(p)}")


# ---------------------------------------------------------------------------------------------
# 4. Dataset x norm stats
# ---------------------------------------------------------------------------------------------
def step_dataset(dataset: str, dataset_root: str | None, cache_root: str, norm_stats: dict):
    header("[4/5] Dataset × norm stats do checkpoint")
    from vla_precision.data.lerobot_version import dataset_codebase_version, lerobot_dataset_root
    from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

    install_lerobot_import_compat()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = SimpleNamespace(
        data=SimpleNamespace(lerobot_repo_id=dataset, lerobot_root=dataset_root, lerobot_auto_convert=True),
        paths=SimpleNamespace(cache_root=cache_root),
    )
    root = lerobot_dataset_root(cfg)
    ds = LeRobotDataset(dataset, root=root)
    version = dataset_codebase_version(root) if root is not None else "v3.0 (Hub)"
    report("4. dataset", "PASSOU", f"{dataset} aberto pelo VLA-Precision ({version}): {len(ds)} frames, raiz {root}")

    getattr(ds, "_ensure_hf_dataset_loaded", lambda: None)()  # o LeRobot do VLA-Precision carrega sob demanda
    hf = ds.hf_dataset.with_format("numpy")
    columns = {
        "state": np.stack(hf["observation.state"]).astype(np.float64),
        "actions": np.stack(hf["action"]).astype(np.float64),
    }
    # O estado é exato. As ações do OpenPI são contadas por chunk (50 passos à frente), o que muda
    # um pouco o peso das pontas dos episódios: por isso a tolerância maior.
    tolerance = {"state": 0.05, "actions": 0.10}
    for key, values in columns.items():
        ns_mean = np.asarray(norm_stats[key]["mean"])
        ns_std = np.asarray(norm_stats[key]["std"])
        mean_err = np.max(np.abs(values.mean(0) - ns_mean) / np.maximum(ns_std, 1e-6))
        std_err = np.max(np.abs(values.std(0) / np.maximum(ns_std, 1e-6) - 1))
        ok = values.shape[1] == ns_mean.shape[0] and mean_err < tolerance[key] and std_err < tolerance[key]
        report(
            "4. norm stats",
            "PASSOU" if ok else "FALHOU",
            f"{key}: {values.shape[1]}D; maior diferença de média = {mean_err:.3f} desvios, "
            f"de desvio = {100 * std_err:.1f}% (tolerância {tolerance[key]:.2f})",
        )
    episodes = np.asarray(hf["episode_index"]).reshape(-1)
    return ds, columns, episodes


# ---------------------------------------------------------------------------------------------
# 5. Inferência e memória
# ---------------------------------------------------------------------------------------------
def step_inference(full_cfg, checkpoint_dir: Path, dataset: str, ds, columns, episodes, num_frames: int):
    header("[5/5] Inferência na GPU e memória")
    import jax
    from openpi.policies import policy_config

    device = jax.devices()[0]
    print(f"  Dispositivo: {device.device_kind} ({device.platform})")
    if device.platform != "gpu":
        report("5. inferência", "ATENÇÃO", "sem GPU: a medição de memória não vale para a 5080")

    def memory() -> dict:
        try:
            return device.memory_stats() or {}
        except Exception:
            return {}

    cfg = dataclasses.replace(full_cfg, data=dataclasses.replace(full_cfg.data, repo_id=dataset))
    t0 = time.time()
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir)
    after_load = memory()
    print(f"  Modelo carregado em bf16 em {time.time() - t0:.0f} s")

    horizon = 10
    action_std = np.maximum(np.asarray(columns["actions"]).std(0), 1e-6)
    indices = np.linspace(0, len(ds) - horizon - 1, num_frames).astype(int)
    model_err, copy_err, latencies = [], [], []
    for n, i in enumerate(indices):
        if episodes[i] != episodes[i + horizon]:
            continue  # o trecho atravessa o fim do episódio
        frame = ds[int(i)]
        obs = {
            "observation/image": frame["observation.images.env"].numpy(),
            "observation/wrist_image": frame["observation.images.wrist"].numpy(),
            "observation/state": frame["observation.state"].numpy(),
            "prompt": frame["task"],
        }
        t = time.time()
        actions = np.asarray(policy.infer(obs)["actions"])
        if n > 0:
            latencies.append(time.time() - t)  # a primeira chamada inclui a compilação
        truth = columns["actions"][i : i + horizon]
        model_err.append(np.abs(actions[:horizon, :7] - truth) / action_std)
        copy_err.append(np.abs(columns["state"][i][None, :] - truth) / action_std)
    peak = memory()

    model_mean = float(np.mean(model_err))
    copy_mean = float(np.mean(copy_err))
    ok = actions.shape[-1] == 7 and model_mean < copy_mean
    report(
        "5. inferência",
        "PASSOU" if ok else "FALHOU",
        f"saída {actions.shape}; erro médio nos {horizon} primeiros passos = {model_mean:.3f} desvios "
        f"(referência 'ficar parado' = {copy_mean:.3f}); {len(model_err)} frames",
    )
    per_joint = np.mean(np.concatenate(model_err), axis=0)
    print("      erro por junta (desvios): " + ", ".join(f"j{k}={v:.2f}" for k, v in enumerate(per_joint)))
    if latencies:
        report("5. inferência", "INFO", f"latência por inferência: {1000 * np.median(latencies):.0f} ms (mediana)")
    if after_load and peak:
        limit = peak.get("bytes_limit", 0)
        report(
            "5. memória",
            "INFO",
            f"depois de carregar: {gib(after_load.get('bytes_in_use', 0))}; pico com inferência: "
            f"{gib(peak.get('peak_bytes_in_use', 0))}; GPU disponível: {gib(limit)}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/stage1") / DEFAULT_HF_REPO.split("/")[1])
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="usado só se o checkpoint não estiver no disco")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-root", default=None, help="pasta local do dataset (senão, Hugging Face)")
    parser.add_argument("--cache-root", default="./.cache", help="onde fica a cópia convertida para v3.0")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--skip-inference", action="store_true", help="pula a etapa 5 (sem GPU)")
    args = parser.parse_args()

    full_cfg, acob_cfg = step_configs()
    checkpoint, _, norm_stats = step_checkpoint(args.checkpoint_dir, args.hf_repo, args.dataset)
    step_keys(checkpoint.params_path, full_cfg, acob_cfg)
    ds, columns, episodes = step_dataset(args.dataset, args.dataset_root, args.cache_root, norm_stats)
    if not args.skip_inference:
        step_inference(full_cfg, args.checkpoint_dir, args.dataset, ds, columns, episodes, args.num_frames)

    header("Resumo")
    for step, status, detail in RESULTS:
        if status != "INFO":
            print(f"  [{status}] {step}: {detail[:110]}")
    failed = [r for r in RESULTS if r[1] == "FALHOU"]
    print(f"\n  {'TUDO CERTO' if not failed else f'{len(failed)} FALHA(S)'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
