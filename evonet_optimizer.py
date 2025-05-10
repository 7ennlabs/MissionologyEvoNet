import os
import subprocess
import sys
import argparse
import random
import logging
from datetime import datetime
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# --- Constants ---
DEFAULT_SEQ_LENGTH = 10
DEFAULT_POP_SIZE = 50
DEFAULT_GENERATIONS = 50
DEFAULT_MUTATION_RATE = 0.4       # Probability of applying any mutation to an individual
DEFAULT_WEIGHT_MUT_RATE = 0.8     # If mutation occurs, probability of weight perturbation
DEFAULT_ACTIVATION_MUT_RATE = 0.2 # If mutation occurs, probability of activation change
DEFAULT_MUTATION_STRENGTH = 0.1 # Magnitude of weight perturbation
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_ELITISM_COUNT = 2         # Keep top N individuals directly
DEFAULT_EPOCHS_FINAL_TRAIN = 100
DEFAULT_BATCH_SIZE = 64

# --- Logging Setup ---
def setup_logging(log_dir: str, log_level=logging.INFO) -> None:
    """Configures logging to file and console."""
    log_filename = os.path.join(log_dir, 'evolution.log')
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout) # Also print to console
        ]
    )

# --- GPU Check ---
def check_gpu() -> bool:
    """Checks for GPU availability and sets memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found.")
            logging.info(f"Using GPU: {gpus[0].name}")
            return True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.error(f"Error setting memory growth: {e}")
            return False
    else:
        logging.warning("GPU not found. Using CPU.")
        return False

# --- Data Generation ---
def generate_data(num_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates random sequences and their sorted versions."""
    logging.info(f"Generating {num_samples} samples with sequence length {seq_length}...")
    X = np.random.rand(num_samples, seq_length) * 100
    y = np.sort(X, axis=1)
    logging.info("Data generation complete.")
    return X, y

# --- Neuroevolution Core ---
def create_individual(seq_length: int) -> Sequential:
    """Creates a Keras Sequential model with random architecture."""
    model = Sequential(name=f"model_random_{random.randint(1000, 9999)}")
    num_hidden_layers = random.randint(1, 4) # Reduced max layers for simplicity
    neurons_per_layer = [random.randint(8, 64) for _ in range(num_hidden_layers)]
    activations = [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(num_hidden_layers)]

    # Input Layer
    model.add(Input(shape=(seq_length,)))

    # Hidden Layers
    for i in range(num_hidden_layers):
        model.add(Dense(neurons_per_layer[i], activation=activations[i]))

    # Output Layer - must match sequence length for sorting
    model.add(Dense(seq_length, activation='linear')) # Linear activation for regression output

    # Compile the model immediately for weight manipulation capabilities
    # Use a standard optimizer; learning rate might be adjusted during final training
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

@tf.function # Potentially speeds up prediction
def get_predictions(model: Sequential, X: np.ndarray, batch_size: int) -> tf.Tensor:
    """Gets model predictions using tf.function."""
    return model(X, training=False) # Use __call__ inside tf.function

def calculate_fitness(individual: Sequential, X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    """Calculates fitness based on inverse MSE. Handles potential errors."""
    try:
        # Ensure data is float32 for TensorFlow
        X_tf = tf.cast(X, tf.float32)
        y_tf = tf.cast(y, tf.float32)

        # Use the tf.function decorated prediction function
        y_pred_tf = get_predictions(individual, X_tf, batch_size)

        # Calculate MSE using TensorFlow operations for potential GPU acceleration
        mse = tf.reduce_mean(tf.square(y_tf - y_pred_tf))
        mse_val = mse.numpy() # Get the numpy value

        # Fitness: Inverse MSE (add small epsilon to avoid division by zero)
        fitness_score = 1.0 / (mse_val + 1e-8)

        # Handle potential NaN or Inf values in fitness
        if not np.isfinite(fitness_score):
            logging.warning(f"Non-finite fitness detected ({fitness_score}) for model {individual.name}. Assigning low fitness.")
            return 1e-8 # Assign a very low fitness

        return float(fitness_score)

    except Exception as e:
        logging.error(f"Error during fitness calculation for model {individual.name}: {e}", exc_info=True)
        return 1e-8 # Return minimal fitness on error


def mutate_individual(individual: Sequential, weight_mut_rate: float, act_mut_rate: float, mut_strength: float) -> Sequential:
    """Applies mutations (weight perturbation, activation change) to an individual."""
    mutated_model = clone_model(individual)
    mutated_model.set_weights(individual.get_weights()) # Crucial: Copy weights

    mutated = False
    # 1. Weight Mutation
    if random.random() < weight_mut_rate:
        mutated = True
        for layer in mutated_model.layers:
            if isinstance(layer, Dense):
                weights_biases = layer.get_weights()
                new_weights_biases = []
                for wb in weights_biases:
                    noise = np.random.normal(0, mut_strength, wb.shape)
                    new_weights_biases.append(wb + noise)
                if new_weights_biases: # Ensure layer had weights
                    layer.set_weights(new_weights_biases)
        # logging.debug(f"Applied weight mutation to {mutated_model.name}")

    # 2. Activation Mutation (Applied independently)
    if random.random() < act_mut_rate:
         # Find Dense layers eligible for activation change (not the output layer)
        dense_layers = [layer for layer in mutated_model.layers if isinstance(layer, Dense)]
        if len(dense_layers) > 1: # Ensure there's at least one hidden layer
            mutated = True
            layer_to_mutate = random.choice(dense_layers[:-1]) # Exclude output layer
            current_activation = layer_to_mutate.get_config().get('activation', 'linear')
            possible_activations = ['relu', 'tanh', 'sigmoid']
            if current_activation in possible_activations:
                 possible_activations.remove(current_activation)
            new_activation = random.choice(possible_activations)

            # Rebuild the model config with the new activation
            # This is safer than trying to modify layer activation in-place
            config = mutated_model.get_config()
            for layer_config in config['layers']:
                 if layer_config['config']['name'] == layer_to_mutate.name:
                      layer_config['config']['activation'] = new_activation
                      # logging.debug(f"Changed activation of layer {layer_to_mutate.name} to {new_activation} in {mutated_model.name}")
                      break # Found the layer

            # Create a new model from the modified config
            # Important: Need to re-compile after structural changes from config
            try:
                mutated_model_new_act = Sequential.from_config(config)
                mutated_model_new_act.compile(optimizer=Adam(learning_rate=0.001), loss='mse') # Re-compile
                mutated_model = mutated_model_new_act # Replace the old model
            except Exception as e:
                 logging.error(f"Error rebuilding model after activation mutation for {mutated_model.name}: {e}")
                 # Revert mutation if rebuilding fails


    # Re-compile the final mutated model to ensure optimizer state is fresh
    if mutated:
        mutated_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        mutated_model._name = f"mutated_{individual.name}" # Rename

    return mutated_model


def tournament_selection(population: List[Sequential], fitness_scores: List[float], k: int) -> Sequential:
    """Selects the best individual from a randomly chosen tournament group."""
    tournament_indices = random.sample(range(len(population)), k)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index_in_tournament = np.argmax(tournament_fitness)
    winner_original_index = tournament_indices[winner_index_in_tournament]
    return population[winner_original_index]

def evolve_population(population: List[Sequential], X: np.ndarray, y: np.ndarray, generations: int,
                      mutation_rate: float, weight_mut_rate: float, act_mut_rate: float, mut_strength: float,
                      tournament_size: int, elitism_count: int, batch_size: int) -> Tuple[Sequential, List[float], List[float]]:
    """Runs the evolutionary process."""
    best_fitness_history = []
    avg_fitness_history = []
    best_model_overall = None
    best_fitness_overall = -1.0

    for gen in range(generations):
        # 1. Evaluate Fitness
        fitness_scores = [calculate_fitness(ind, X, y, batch_size) for ind in population]

        # Track overall best
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            # Keep a copy of the best model structure and weights
            best_model_overall = clone_model(population[current_best_idx])
            best_model_overall.set_weights(population[current_best_idx].get_weights())
            best_model_overall.compile(optimizer=Adam(), loss='mse') # Re-compile just in case
            logging.info(f"Generation {gen+1}: New overall best fitness: {best_fitness_overall:.4f}")


        avg_fitness = np.mean(fitness_scores)
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)

        logging.info(f"Generation {gen+1}/{generations} - Best Fitness: {current_best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}")

        new_population = []

        # 2. Elitism: Carry over the best individuals
        if elitism_count > 0:
            elite_indices = np.argsort(fitness_scores)[-elitism_count:]
            for idx in elite_indices:
                 # Clone elite models to avoid modifications affecting originals if selected again
                 elite_clone = clone_model(population[idx])
                 elite_clone.set_weights(population[idx].get_weights())
                 elite_clone.compile(optimizer=Adam(), loss='mse') # Ensure compiled
                 new_population.append(elite_clone)


        # 3. Selection & Reproduction for the rest of the population
        while len(new_population) < len(population):
            # Select parent(s) using tournament selection
            parent = tournament_selection(population, fitness_scores, tournament_size)

            # Create child through mutation (crossover could be added here)
            child = parent # Start with the parent
            if random.random() < mutation_rate:
                # Clone parent before mutation to avoid modifying the original selected parent
                parent_clone = clone_model(parent)
                parent_clone.set_weights(parent.get_weights())
                parent_clone.compile(optimizer=Adam(), loss='mse') # Ensure compiled
                child = mutate_individual(parent_clone, weight_mut_rate, act_mut_rate, mut_strength)
            else:
                 # If no mutation, still clone the parent to ensure new population has distinct objects
                 child = clone_model(parent)
                 child.set_weights(parent.get_weights())
                 child.compile(optimizer=Adam(), loss='mse') # Ensure compiled


            new_population.append(child)

        population = new_population[:len(population)] # Ensure population size is maintained

    if best_model_overall is None: # Handle case where no improvement was ever found
         best_idx = np.argmax([calculate_fitness(ind, X, y, batch_size) for ind in population])
         best_model_overall = population[best_idx]

    return best_model_overall, best_fitness_history, avg_fitness_history


# --- Plotting ---
def plot_fitness_history(history_best: List[float], history_avg: List[float], output_dir: str) -> None:
    """Plots and saves the fitness history."""
    plt.figure(figsize=(12, 6))
    plt.plot(history_best, label="Best Fitness per Generation", marker='o', linestyle='-')
    plt.plot(history_avg, label="Average Fitness per Generation", marker='x', linestyle='--')
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score (1 / MSE)")
    plt.title("Evolutionary Process Fitness History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fitness_history.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Fitness history plot saved to {plot_path}")

# --- Evaluation ---
def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, batch_size: int) -> Dict[str, float]:
    """Evaluates the final model on the test set."""
    logging.info("Evaluating final model on test data...")
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
    test_mse = np.mean(np.square(y_test - y_pred))
    logging.info(f"Final Test MSE: {test_mse:.6f}")

    # Calculate Kendall's Tau for a sample (can be slow for large datasets)
    sample_size = min(100, X_test.shape[0])
    taus = []
    indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
    for i in indices:
        tau, _ = kendalltau(y_test[i], y_pred[i])
        if not np.isnan(tau): # Handle potential NaN if predictions are constant
            taus.append(tau)
    avg_kendall_tau = np.mean(taus) if taus else 0.0
    logging.info(f"Average Kendall's Tau (on {sample_size} samples): {avg_kendall_tau:.4f}")

    return {
        "test_mse": float(test_mse),
        "avg_kendall_tau": float(avg_kendall_tau)
    }

# --- Main Pipeline ---
def run_pipeline(args: argparse.Namespace):
    """Executes the complete neuroevolution pipeline."""

    # Create unique output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_base_dir, f"evorun_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging for this run
    setup_logging(output_dir)
    logging.info(f"Starting EvoNet Pipeline Run: {timestamp}")
    logging.info(f"Output directory: {output_dir}")

    # Log arguments/configuration
    logging.info("Configuration:")
    args_dict = vars(args)
    for k, v in args_dict.items():
        logging.info(f"  {k}: {v}")
    # Save config to file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    logging.info(f"Configuration saved to {config_path}")


    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    logging.info(f"Using random seed: {args.seed}")

    # Check GPU
    check_gpu()

    # Generate Data
    X_train, y_train = generate_data(args.train_samples, args.seq_length)
    X_test, y_test = generate_data(args.test_samples, args.seq_length)

    # Initialize Population
    logging.info(f"Initializing population of {args.pop_size} individuals...")
    population = [create_individual(args.seq_length) for _ in range(args.pop_size)]
    logging.info("Population initialized.")

    # Run Evolution
    logging.info(f"Starting evolution for {args.generations} generations...")
    best_model_unevolved, best_fitness_hist, avg_fitness_hist = evolve_population(
        population, X_train, y_train, args.generations,
        args.mutation_rate, args.weight_mut_rate, args.activation_mut_rate, args.mutation_strength,
        args.tournament_size, args.elitism_count, args.batch_size
    )
    logging.info("Evolution complete.")

    # Save fitness history data
    history_path = os.path.join(output_dir, "fitness_history.csv")
    history_data = np.array([best_fitness_hist, avg_fitness_hist]).T
    np.savetxt(history_path, history_data, delimiter=',', header='BestFitness,AvgFitness', comments='')
    logging.info(f"Fitness history data saved to {history_path}")

    # Plot fitness history
    plot_fitness_history(best_fitness_hist, avg_fitness_hist, output_dir)

    # Final Training of the Best Model
    logging.info("Starting final training of the best evolved model...")
    # Clone the best model again to ensure we don't modify the original reference unintentionally
    final_model = clone_model(best_model_unevolved)
    final_model.set_weights(best_model_unevolved.get_weights())
    # Use a fresh optimizer instance for final training
    final_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Callbacks for efficient training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

    # Use a portion of training data for validation during final training
    history = final_model.fit(
        X_train, y_train,
        epochs=args.epochs_final_train,
        batch_size=args.batch_size,
        validation_split=0.2, # Use 20% of training data for validation
        callbacks=[early_stopping, reduce_lr],
        verbose=2 # Show one line per epoch
    )
    logging.info("Final training complete.")

    # Evaluate the TRAINED final model
    final_metrics = evaluate_model(final_model, X_test, y_test, args.batch_size)

    # Save the TRAINED final model
    model_path = os.path.join(output_dir, "best_evolved_model_trained.keras") # Use .keras format
    final_model.save(model_path)
    logging.info(f"Final trained model saved to {model_path}")

    # Save final results
    results = {
        "config": args_dict,
        "final_evaluation": final_metrics,
        "evolution_summary": {
            "best_fitness_overall": best_fitness_hist[-1] if best_fitness_hist else None,
            "avg_fitness_final_gen": avg_fitness_hist[-1] if avg_fitness_hist else None,
        },
        "training_history": history.history # Include loss/val_loss history from final training
    }
    results_path = os.path.join(output_dir, "final_results.json")
    # Convert numpy types in history to native Python types for JSON serialization
    for key in results['training_history']:
        results['training_history'][key] = [float(v) for v in results['training_history'][key]]

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Final results saved to {results_path}")
    logging.info("Pipeline finished successfully!")


# --- Argument Parser ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoNet: Neuroevolution for Sorting Task")

    # --- Directory ---
    parser.add_argument('--output_base_dir', type=str, default=os.path.join(os.getcwd(), "evonet_runs"),
                        help='Base directory to store run results.')

    # --- Data ---
    parser.add_argument('--seq_length', type=int, default=DEFAULT_SEQ_LENGTH,
                        help='Length of the sequences to sort.')
    parser.add_argument('--train_samples', type=int, default=5000, help='Number of training samples.')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples.')

    # --- Evolution Parameters ---
    parser.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE, help='Population size.')
    parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS, help='Number of generations.')
    parser.add_argument('--mutation_rate', type=float, default=DEFAULT_MUTATION_RATE,
                        help='Overall probability of mutating an individual.')
    parser.add_argument('--weight_mut_rate', type=float, default=DEFAULT_WEIGHT_MUT_RATE,
                        help='Probability of weight perturbation if mutation occurs.')
    parser.add_argument('--activation_mut_rate', type=float, default=DEFAULT_ACTIVATION_MUT_RATE,
                        help='Probability of activation change if mutation occurs.')
    parser.add_argument('--mutation_strength', type=float, default=DEFAULT_MUTATION_STRENGTH,
                        help='Standard deviation of Gaussian noise for weight mutation.')
    parser.add_argument('--tournament_size', type=int, default=DEFAULT_TOURNAMENT_SIZE,
                        help='Number of individuals participating in tournament selection.')
    parser.add_argument('--elitism_count', type=int, default=DEFAULT_ELITISM_COUNT,
                        help='Number of best individuals to carry over directly.')

    # --- Training & Evaluation ---
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for predictions and training.')
    parser.add_argument('--epochs_final_train', type=int, default=DEFAULT_EPOCHS_FINAL_TRAIN,
                        help='Max epochs for final training of the best model.')

    # --- Reproducibility ---
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: random).')

    args = parser.parse_args()

    # If seed is not provided, generate one
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    return args


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Parse Command Line Arguments
    cli_args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(cli_args.output_base_dir, exist_ok=True)

    # 2. Run the Pipeline
    try:
        run_pipeline(cli_args)
    except Exception as e:
        # Log any uncaught exceptions during the pipeline execution
        # The logger might not be set up if error is early, so print as fallback
        print(f"FATAL ERROR in pipeline execution: {e}", file=sys.stderr)
        # Attempt to log if logger was initialized
        if logging.getLogger().hasHandlers():
             logging.critical("FATAL ERROR in pipeline execution:", exc_info=True)
        else:
             import traceback
             print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1) # Exit with error code