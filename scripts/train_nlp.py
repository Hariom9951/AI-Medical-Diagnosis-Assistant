import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow

from src.components.nlp_model_trainer import NLPClassifierConfig, SymptomClassifier
from src.pipeline.nlp_training_pipeline import NLPTrainingPipeline


def main():
    print("Initializing NLP Training Pipeline runner...")
    pipeline = NLPTrainingPipeline(Path("configs/nlp_training_config.yaml"))

    # Get configuration and map classes
    config = pipeline.config

    print("=" * 60)
    print("PHASE 15 NLP BRANCH RUN INITIALIZATION")
    print("=" * 60)
    print(f"Base Model: {config.model_name}")
    print(f"Tokenizer Name: {config.tokenizer_name}")
    print(f"Max Sequence Length: {config.max_length}")
    print(f"Dropout: {config.dropout}")

    # Run 1 epoch
    print("\nExecuting training run...")
    results = pipeline.run(max_epochs=1)

    # Get model summary parameters
    classifier = SymptomClassifier(
        config.model_name, num_classes=results["num_classes"], dropout=config.dropout
    )
    summary = classifier.model_summary()

    # Print requested outputs
    print("\n" + "=" * 60)
    print("EXECUTION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model Summary: {summary['model_name']}")
    print(f"Number of Classes: {results['num_classes']}")
    print(f"Tokenizer Vocabulary Size: {results['vocab_size']}")
    print(f"Total Parameters: {summary['total_parameters']}")
    print(f"Trainable Parameters: {summary['trainable_parameters']}")
    print(f"Frozen Parameters: {summary['frozen_parameters']}")

    history = results["training_history"]
    if history:
        epoch_stats = history[0]
        print(f"\nOne Successful Training Epoch Metrics:")
        print(f"  - Train Loss: {epoch_stats['train_loss']}")
        print(f"  - Train Accuracy: {epoch_stats['train_acc']:.4f}")
        print(f"  - Validation Loss: {epoch_stats['val_loss']}")
        print(f"  - Validation Accuracy: {epoch_stats['val_acc']:.4f}")
        print(f"  - Learning Rate: {epoch_stats['learning_rate']}")
        print(f"  - Training Time: {epoch_stats['epoch_time_s']}s")

    # Fetch active mlflow run or run id
    # Since the run completes, we search for the latest run in the experiment
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
        )
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            print(f"\nMLflow Run ID: {run_id}")
            print(f"MLflow Tracking URI: {config.mlflow_tracking_uri}")
            print(f"MLflow Experiment: {config.mlflow_experiment_name}")

    print(f"\nCheckpoint Location (Best Model): {results['best_model_path']}")
    print(f"Tokenizer Location: {results['tokenizer_path']}")
    print(f"Curves Plot Location: {results['curves_path']}")
    print(f"Configuration Snapshot: {results['config_snapshot_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
