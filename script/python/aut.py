"""Feasibility categorization orchestrator using modular pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import (
    HF_API_NAME,
    HF_SPACE_ID,
    LABEL_COLUMN,
    LLM_MODEL,
    LOCAL_MODEL_NAME,
    METRICS_FILE,
    MODEL_BACKEND,
    OUTPUT_FILE,
)
from .local_model_utils import classify_dataframe_with_local_model
from .email_utils import send_result_email
from .hf_api_utils import classify_dataframe_with_hf_api
from .io_utils import ensure_directories, resolve_input_file
from .llm_utils import classify_dataframe_with_llm
from .preprocessing_utils import (
    ensure_nltk_resources,
    load_dataset,
    prepare_dataframe,
    validate_training_data,
)
from .reporting_utils import save_classified_data, save_metrics_summary
from .visualization_utils import (
    generate_wordclouds,
    plot_category_distribution,
)


def run_pipeline() -> None:
    """Execute the full NLP workflow end-to-end."""
    ensure_directories()
    ensure_nltk_resources()

    dataset_path = resolve_input_file()
    print(f'Reading input file: {dataset_path}')
    df_raw = load_dataset(dataset_path)
    df_prepared = prepare_dataframe(df_raw)

    artifacts: dict | None = None
    feature_importance = None
    metrics_summary: dict | None = None
    run_metadata = {'backend': MODEL_BACKEND}

    if MODEL_BACKEND == 'llm':
        print('Using LLM backend; skipping traditional training.')
        df_classified = classify_dataframe_with_llm(df_prepared)
        metrics_summary = {'backend': 'llm', 'llm_model': LLM_MODEL}
        run_metadata.update(
            {
                'llm_model': LLM_MODEL,
                'records_classified': len(df_classified),
            }
        )
    elif MODEL_BACKEND == 'bert':
        from .modeling_utils import (  # Local import to avoid torch dependency when unused
            apply_bert_model,
            train_and_evaluate_bert,
        )

        training_df = validate_training_data(df_prepared)
        artifacts = train_and_evaluate_bert(
            training_df['texto_bruto'],
            training_df[LABEL_COLUMN],
        )
        metrics_summary = artifacts['metrics_summary']
        print(
            f"BERT ({artifacts['model_name']}) accuracy: "
            f"{metrics_summary['accuracy']:.3f}"
        )
        df_classified = apply_bert_model(
            df_prepared,
            artifacts,
            text_column='texto_bruto',
        )
        run_metadata.update(
            {
                'bert_model': artifacts['model_name'],
                'records_classified': len(df_classified),
            }
        )
    elif MODEL_BACKEND == 'hf':
        df_classified = classify_dataframe_with_hf_api(
            df_prepared,
            text_column='texto_bruto',
        )
        metrics_summary = {
            'backend': 'hf',
            'hf_space': HF_SPACE_ID,
            'hf_api_name': HF_API_NAME,
        }
        run_metadata.update(
            {
                'hf_space': HF_SPACE_ID,
                'hf_api_name': HF_API_NAME,
                'records_classified': len(df_classified),
            }
        )
    elif MODEL_BACKEND == 'local':
        print('Using local transformers pipeline backend.')
        df_classified = classify_dataframe_with_local_model(
            df_prepared,
            text_column='texto_bruto',
        )
        metrics_summary = {
            'backend': 'local',
            'local_model': LOCAL_MODEL_NAME,
        }
        run_metadata.update(
            {
                'local_model': LOCAL_MODEL_NAME,
                'records_classified': len(df_classified),
            }
        )
    else:
        from .modeling_utils import (
            apply_best_model,
            extract_feature_importance,
            train_and_evaluate,
        )

        training_df = validate_training_data(df_prepared)
        artifacts = train_and_evaluate(
            training_df['texto_limpo'],
            training_df[LABEL_COLUMN],
        )
        metrics_summary = artifacts
        best_pipeline = artifacts['best_pipeline']
        feature_importance = extract_feature_importance(best_pipeline)

        print(
            'Accuracies -> '
            + ', '.join(
                f'{name.upper()}: {result["accuracy"]:.3f}'
                for name, result in artifacts['results'].items()
            )
        )
        print('Best model:', artifacts['best_key'].upper())

        df_classified = apply_best_model(df_prepared, best_pipeline)
        run_metadata.update(
            {
                'best_model_key': artifacts['best_key'],
                'records_classified': len(df_classified),
            }
        )

    save_classified_data(df_classified, dataset_path)
    save_metrics_summary(metrics_summary, feature_importance, run_metadata)

    wordcloud_paths = generate_wordclouds(df_classified, 'texto_bruto')
    plot_category_distribution(df_classified)

    if wordcloud_paths:
        print('Word clouds saved at:')
        for path in wordcloud_paths:
            print(f'  - {path}')
    print(f'Metrics report saved to: {METRICS_FILE}')
    send_result_email(OUTPUT_FILE, df_classified, dataset_path)


def clear_hf_cache(cache_dir: str | Path) -> None:
    """Remove Hugging Face cache directory after the run finishes."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f'Cache path not found, nothing to remove: {cache_path}')
        return

    try:
        shutil.rmtree(cache_path)
        print(f'Removed Hugging Face cache at: {cache_path}')
    except OSError as exc:
        print(f'Failed to remove Hugging Face cache ({cache_path}): {exc}')


if __name__ == '__main__':
    run_pipeline()
    clear_hf_cache(Path.home() / '.cache' / 'huggingface')