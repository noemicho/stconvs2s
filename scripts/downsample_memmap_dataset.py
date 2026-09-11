from pathlib import Path
import argparse
import json
import shutil

import numpy as np


TARGET_METADATA_FILES = {
    "websirene": "targets_metadata.json",
    "alertario": "targets_alertario_metadata.json",
}


DEFAULT_TARGET_FILES = {
    "websirene": ("Y_all.dat", "M_all.dat"),
    "alertario": ("Y_alertario.dat", "M_alertario.dat"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera uma versão com menor resolução espacial de um dataset "
            "memmap já existente, preservando timestamps e remapeando "
            "radar, targets e máscaras para uma nova grade."
        )
    )

    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--target-source", choices=["alertario", "websirene", "both"], default="alertario")
    parser.add_argument("--chunk-size", type=int, default=256)

    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def nearest_indices(old_size, new_size):
    if old_size == new_size:
        return np.arange(old_size, dtype=np.int64)

    return np.floor(np.arange(new_size) * old_size / new_size).astype(np.int64)


def remap_indices(old_indices, old_size, new_size):
    new_indices = np.floor(old_indices * new_size / old_size).astype(np.int64)
    return np.clip(new_indices, 0, new_size - 1)


def downsample_radar(source_year_dir, output_year_dir, height, width, chunk_size):
    metadata_path = source_year_dir / "metadata.json"
    metadata = load_json(metadata_path)

    old_shape = tuple(metadata["shape"])
    n_frames, old_height, old_width, channels = old_shape

    if channels != 3:
        raise ValueError(f"Esperado radar com 3 canais, recebido: {channels}")

    source_frames = np.memmap(
        source_year_dir / metadata.get("frames_file", "radar_frames.dat"),
        dtype=np.dtype(metadata.get("dtype", "uint8")),
        mode="r",
        shape=old_shape,
    )

    output_shape = (n_frames, height, width, channels)
    output_frames = np.memmap(
        output_year_dir / "radar_frames.dat",
        dtype=np.uint8,
        mode="w+",
        shape=output_shape,
    )

    row_idx = nearest_indices(old_height, height)
    col_idx = nearest_indices(old_width, width)

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk = np.asarray(source_frames[start:end])
        output_frames[start:end] = chunk[:, row_idx][:, :, col_idx, :]

        print(
            f"  radar frames {start}:{end}/{n_frames}",
            flush=True,
        )

    output_frames.flush()

    shutil.copy2(
        source_year_dir / metadata.get("timestamps_file", "radar_timestamps.npy"),
        output_year_dir / "radar_timestamps.npy",
    )

    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "height": height,
            "width": width,
            "channels": channels,
            "dtype": "uint8",
            "shape": list(output_shape),
            "frames_file": "radar_frames.dat",
            "timestamps_file": "radar_timestamps.npy",
            "downsampled_from_shape": list(old_shape),
            "downsample_method": "nearest_neighbor_from_memmap",
        }
    )

    save_json(output_year_dir / "metadata.json", output_metadata)

    return old_height, old_width, n_frames


def target_sources_to_process(target_source):
    if target_source == "both":
        return ["websirene", "alertario"]

    return [target_source]


def downsample_targets(
    source_year_dir,
    output_year_dir,
    source,
    old_height,
    old_width,
    height,
    width,
    chunk_size,
):
    metadata_name = TARGET_METADATA_FILES[source]
    metadata_path = source_year_dir / metadata_name

    if not metadata_path.exists():
        print(f"  {metadata_name} não encontrado. Pulando {source}.", flush=True)
        return

    metadata = load_json(metadata_path)
    default_y_file, default_m_file = DEFAULT_TARGET_FILES[source]
    y_file = metadata.get("Y_file", default_y_file)
    m_file = metadata.get("M_file", default_m_file)

    source_y_path = source_year_dir / y_file
    source_m_path = source_year_dir / m_file

    if not source_y_path.exists() or not source_m_path.exists():
        print(f"  Arquivos Y/M de {source} não encontrados. Pulando.", flush=True)
        return

    old_shape = tuple(metadata["shape"])
    n_frames, _, _, channels = old_shape

    if channels != 1:
        raise ValueError(f"Esperado target com 1 canal, recebido: {channels}")

    source_y = np.memmap(
        source_y_path,
        dtype=np.float32,
        mode="r",
        shape=old_shape,
    )
    source_m = np.memmap(
        source_m_path,
        dtype=np.uint8,
        mode="r",
        shape=old_shape,
    )

    output_shape = (n_frames, height, width, channels)
    output_y = np.memmap(
        output_year_dir / y_file,
        dtype=np.float32,
        mode="w+",
        shape=output_shape,
    )
    output_m = np.memmap(
        output_year_dir / m_file,
        dtype=np.uint8,
        mode="w+",
        shape=output_shape,
    )

    output_y[:] = 0.0
    output_m[:] = 0

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        y_chunk = np.asarray(source_y[start:end, :, :, 0])
        m_chunk = np.asarray(source_m[start:end, :, :, 0])

        t_idx, row_idx, col_idx = np.nonzero(m_chunk > 0)

        if len(t_idx) > 0:
            new_rows = remap_indices(row_idx, old_height, height)
            new_cols = remap_indices(col_idx, old_width, width)
            rain_log = y_chunk[t_idx, row_idx, col_idx]

            np.maximum.at(
                output_y,
                (
                    t_idx + start,
                    new_rows,
                    new_cols,
                    np.zeros(len(t_idx), dtype=np.int64),
                ),
                rain_log,
            )
            output_m[
                t_idx + start,
                new_rows,
                new_cols,
                0,
            ] = 1

        print(
            f"  {source} targets {start}:{end}/{n_frames}",
            flush=True,
        )

    output_y.flush()
    output_m.flush()

    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "height": height,
            "width": width,
            "channels": channels,
            "shape": list(output_shape),
            "Y_file": y_file,
            "M_file": m_file,
            "downsampled_from_shape": list(old_shape),
            "downsample_method": "station_pixel_remap_max_log_target",
        }
    )

    save_json(output_year_dir / metadata_name, output_metadata)


def process_year(args, year):
    source_year_dir = args.source_root / f"year={year}"
    output_year_dir = args.output_root / f"year={year}"

    if not source_year_dir.exists():
        print(f"[{year}] Diretório de origem não encontrado. Pulando.", flush=True)
        return

    output_year_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60, flush=True)
    print(f"REDUZINDO RESOLUÇÃO DO ANO {year}", flush=True)
    print("=" * 60, flush=True)

    old_height, old_width, _ = downsample_radar(
        source_year_dir=source_year_dir,
        output_year_dir=output_year_dir,
        height=args.height,
        width=args.width,
        chunk_size=args.chunk_size,
    )

    for source in target_sources_to_process(args.target_source):
        downsample_targets(
            source_year_dir=source_year_dir,
            output_year_dir=output_year_dir,
            source=source,
            old_height=old_height,
            old_width=old_width,
            height=args.height,
            width=args.width,
            chunk_size=args.chunk_size,
        )

    print(f"[{year}] salvo em: {output_year_dir}", flush=True)


def main():
    args = parse_args()

    if args.source_root.resolve() == args.output_root.resolve():
        raise ValueError("--output-root deve ser diferente de --source-root")

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("=== CONFIGURAÇÃO GERAL ===", flush=True)
    print("source_root:", args.source_root, flush=True)
    print("output_root:", args.output_root, flush=True)
    print("year_start:", args.year_start, flush=True)
    print("year_end:", args.year_end, flush=True)
    print("output_resolution:", (args.height, args.width), flush=True)
    print("target_source:", args.target_source, flush=True)
    print("chunk_size:", args.chunk_size, flush=True)

    for year in range(args.year_start, args.year_end + 1):
        process_year(args, year)

    print("\nTODOS OS ANOS FINALIZADOS.", flush=True)


if __name__ == "__main__":
    main()
