# ==============================================================================
# EvoNet Optimizer - v3 - Daha İleri İyileştirmeler
# Açıklama: Çaprazlama, Kontrol Noktası eklenmiş, Adaptif Mutasyon ve
# Gelişmiş Fitness için kavramsal öneriler içeren versiyon.
# ==============================================================================

import os
import subprocess
import sys
import argparse
import random
import logging
from datetime import datetime
import json
import pickle # Checkpointing için
import time   # Checkpointing için
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# --- Sabitler ve Varsayılan Değerler ---
DEFAULT_SEQ_LENGTH = 10
DEFAULT_POP_SIZE = 50
DEFAULT_GENERATIONS = 50
DEFAULT_CROSSOVER_RATE = 0.6      # Çaprazlama uygulama olasılığı
DEFAULT_MUTATION_RATE = 0.4       # Mutasyon uygulama olasılığı (eğer çaprazlama olmazsa)
DEFAULT_WEIGHT_MUT_RATE = 0.8
DEFAULT_ACTIVATION_MUT_RATE = 0.2 # Aktivasyon mutasyonu hala deneysel
DEFAULT_MUTATION_STRENGTH = 0.1
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_ELITISM_COUNT = 2
DEFAULT_EPOCHS_FINAL_TRAIN = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "evonet_runs_v3")
DEFAULT_CHECKPOINT_INTERVAL = 10  # Kaç nesilde bir checkpoint alınacağı (0 = kapalı)

# --- Loglama Ayarları ---
# (setup_logging fonksiyonu öncekiyle aynı, tekrar eklemiyorum)
def setup_logging(log_dir: str, log_level=logging.INFO) -> None:
    log_filename = os.path.join(log_dir, 'evolution_run.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'), # 'a' mode append for resuming
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging setup complete.")

# --- GPU Kontrolü ---
# (check_gpu fonksiyonu öncekiyle aynı, tekrar eklemiyorum)
def check_gpu() -> bool:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found.")
            if logical_gpus: logging.info(f"Using GPU: {tf.config.experimental.get_device_details(gpus[0])['device_name']}")
            return True
        except RuntimeError as e:
            logging.error(f"Error setting memory growth for GPU: {e}", exc_info=True)
            return False
    else:
        logging.warning("GPU not found. Using CPU.")
        return False

# --- Veri Üretimi ---
# (generate_data fonksiyonu öncekiyle aynı, tekrar eklemiyorum)
def generate_data(num_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    logging.info(f"Generating {num_samples} samples with sequence length {seq_length}...")
    try:
        X = np.random.rand(num_samples, seq_length).astype(np.float32) * 100
        y = np.sort(X, axis=1).astype(np.float32)
        logging.info("Data generation successful.")
        return X, y
    except Exception as e:
        logging.error(f"Error during data generation: {e}", exc_info=True)
        raise

# --- Neuroevolution Çekirdeği ---

def create_individual(seq_length: int, input_shape: Tuple) -> Sequential:
    """Rastgele mimariye sahip bir Keras Sequential modeli oluşturur ve derler."""
    # (Fonksiyon öncekiyle büyük ölçüde aynı, isim revize edildi)
    try:
        model = Sequential(name=f"model_rnd_{random.randint(10000, 99999)}")
        num_hidden_layers = random.randint(1, 4)
        neurons_per_layer = [random.randint(8, 64) for _ in range(num_hidden_layers)]
        activations = [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(num_hidden_layers)]
        model.add(Input(shape=input_shape))
        for i in range(num_hidden_layers):
            model.add(Dense(neurons_per_layer[i], activation=activations[i]))
        model.add(Dense(seq_length, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    except Exception as e:
        logging.error(f"Error creating individual model: {e}", exc_info=True)
        raise

@tf.function
def get_predictions(model: Sequential, X: tf.Tensor) -> tf.Tensor:
    """Model tahminlerini tf.function kullanarak alır."""
    return model(X, training=False)

def calculate_fitness(individual: Sequential, X: np.ndarray, y: np.ndarray, batch_size: int, fitness_params: Dict = None) -> float:
    """Bir bireyin fitness değerini hesaplar. Gelişmiş fitness için öneri içerir."""
    # --- KAVRAMSAL: Gelişmiş Fitness Fonksiyonu ---
    # Burada sadece MSE kullanılıyor. Daha gelişmiş bir fitness için:
    # 1. Diğer metrikleri hesapla (örn: Kendall Tau).
    # 2. Model karmaşıklığını hesapla (örn: parametre sayısı).
    # 3. Bu değerleri ağırlıklı bir formülle birleştir.
    # fitness_params = fitness_params or {}
    # w_mse = fitness_params.get('w_mse', 1.0)
    # w_tau = fitness_params.get('w_tau', 0.1)
    # w_comp = fitness_params.get('w_comp', 0.0001)
    # --------------------------------------------
    if not isinstance(X, tf.Tensor): X = tf.cast(X, tf.float32)
    if not isinstance(y, tf.Tensor): y = tf.cast(y, tf.float32)
    try:
        y_pred_tf = get_predictions(individual, X)
        mse = tf.reduce_mean(tf.square(y - y_pred_tf))
        mse_val = mse.numpy()
        fitness_score = 1.0 / (mse_val + 1e-8) # Temel fitness

        # --- KAVRAMSAL: Gelişmiş Fitness Hesabı ---
        # if w_tau > 0 or w_comp > 0:
        #     # Kendall Tau hesapla (maliyetli olabilir, örneklem gerekebilir)
        #     tau_val = calculate_avg_kendall_tau(y.numpy(), y_pred_tf.numpy(), sample_size=100) # Örnek bir fonksiyon
        #     # Karmaşıklık hesapla
        #     complexity = individual.count_params()
        #     # Birleştirilmiş fitness
        #     fitness_score = w_mse * fitness_score + w_tau * tau_val - w_comp * complexity
        # --------------------------------------------

        if not np.isfinite(fitness_score) or fitness_score < -1e6: # Negatif olabilen fitness için kontrol
            logging.warning(f"Non-finite or very low fitness ({fitness_score:.4g}) for model {individual.name}. Assigning minimal fitness.")
            return -1e7 # Gelişmiş fitness negatif olabileceği için daha düşük sınır
        return float(fitness_score)
    except Exception as e:
        logging.error(f"Error during fitness calculation for model {individual.name}: {e}", exc_info=True)
        return -1e7

# (Aktivasyon mutasyonu hala deneysel, ana odak ağırlık mutasyonunda)
def mutate_individual(individual: Sequential, weight_mut_rate: float, mut_strength: float) -> Sequential:
    """Bir bireye ağırlık bozulması mutasyonu uygular."""
    try:
        mutated_model = clone_model(individual)
        mutated_model.set_weights(individual.get_weights())
        mutated = False
        if random.random() < weight_mut_rate: # Ağırlık mutasyon olasılığı (dışarıdan gelen genel rate ile birleştirilebilir)
            mutated = True
            for layer in mutated_model.layers:
                if isinstance(layer, Dense) and layer.get_weights():
                    weights_biases = layer.get_weights()
                    new_weights_biases = [wb + np.random.normal(0, mut_strength, wb.shape).astype(np.float32) for wb in weights_biases]
                    layer.set_weights(new_weights_biases)

        if mutated:
            mutated_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            mutated_model._name = f"mutated_{individual.name}_{random.randint(1000,9999)}"
        return mutated_model
    except Exception as e:
        logging.error(f"Error during mutation of model {individual.name}: {e}", exc_info=True)
        return individual


def check_architecture_compatibility(model1: Sequential, model2: Sequential) -> bool:
    """İki modelin basit çaprazlama için uyumlu olup olmadığını kontrol eder (katman sayısı ve tipleri)."""
    if len(model1.layers) != len(model2.layers):
        return False
    for l1, l2 in zip(model1.layers, model2.layers):
        if type(l1) != type(l2):
            return False
        # Daha detaylı kontrol (nöron sayısı vb.) eklenebilir, ancak basit tutalım.
    return True

def crossover_individuals(parent1: Sequential, parent2: Sequential) -> Tuple[Optional[Sequential], Optional[Sequential]]:
    """İki ebeveynden basit ağırlık ortalaması/karıştırması ile çocuklar oluşturur."""
    # Mimari uyumluluğunu kontrol et (basit versiyon)
    if not check_architecture_compatibility(parent1, parent2):
        logging.debug("Skipping crossover due to incompatible architectures.")
        return None, None # Uyumsuzsa çaprazlama yapma

    try:
        # Çocukları ebeveynleri klonlayarak başlat
        child1 = clone_model(parent1)
        child2 = clone_model(parent2)
        child1.set_weights(parent1.get_weights()) # Başlangıç ağırlıklarını ata
        child2.set_weights(parent2.get_weights())

        p1_weights = parent1.get_weights()
        p2_weights = parent2.get_weights()
        child1_new_weights = []
        child2_new_weights = []

        # Katman katman ağırlıkları çaprazla
        for i in range(len(p1_weights)): # Ağırlık matrisleri/bias vektörleri üzerinde döngü
            w1 = p1_weights[i]
            w2 = p2_weights[i]
            # Basit ortalama veya rastgele seçim (örnek: rastgele seçim)
            mask = np.random.rand(*w1.shape) < 0.5
            cw1 = np.where(mask, w1, w2)
            cw2 = np.where(mask, w2, w1) # Ters maske ile
            # Veya basit ortalama: cw1 = (w1 + w2) / 2.0; cw2 = cw1
            child1_new_weights.append(cw1.astype(np.float32))
            child2_new_weights.append(cw2.astype(np.float32))


        child1.set_weights(child1_new_weights)
        child2.set_weights(child2_new_weights)

        # Çocukları derle
        child1.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        child2.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        child1._name = f"xover_{parent1.name[:10]}_{parent2.name[:10]}_c1_{random.randint(1000,9999)}"
        child2._name = f"xover_{parent1.name[:10]}_{parent2.name[:10]}_c2_{random.randint(1000,9999)}"
        #logging.debug(f"Crossover performed between {parent1.name} and {parent2.name}")
        return child1, child2

    except Exception as e:
        logging.error(f"Error during crossover between {parent1.name} and {parent2.name}: {e}", exc_info=True)
        return None, None # Hata olursa çocuk üretme

# (tournament_selection fonksiyonu öncekiyle aynı)
def tournament_selection(population: List[Sequential], fitness_scores: List[float], k: int) -> Sequential:
    if not population: raise ValueError("Population cannot be empty.")
    if len(population) < k: k = len(population)
    try:
        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_local_idx = np.argmax(tournament_fitness)
        winner_global_idx = tournament_indices[winner_local_idx]
        return population[winner_global_idx]
    except Exception as e:
        logging.error(f"Error during tournament selection: {e}", exc_info=True)
        return random.choice(population)

# --- Checkpointing ---
def save_checkpoint(output_dir: str, generation: int, population: List[Sequential], rnd_state: Tuple, np_rnd_state: Tuple, tf_rnd_state: Any):
    """Evrim durumunu kaydeder."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"evo_gen_{generation}.pkl")
    logging.info(f"Saving checkpoint for generation {generation} to {checkpoint_file}...")
    try:
        # Modelleri kaydetmek için ağırlıkları ve konfigürasyonları al
        population_state = []
        for model in population:
            try:
                 # Önce modeli diske kaydetmeyi dene (daha sağlam olabilir ama yavaş)
                 # model_path = os.path.join(checkpoint_dir, f"model_gen{generation}_{model.name}.keras")
                 # model.save(model_path)
                 # population_state.append({"config": model.get_config(), "saved_path": model_path})

                 # Alternatif: Ağırlık ve config'i pickle içine göm (daha riskli)
                 population_state.append({
                     "name": model.name,
                     "config": model.get_config(),
                     "weights": model.get_weights()
                 })
            except Exception as e:
                 logging.error(f"Could not serialize model {model.name} for checkpoint: {e}")
                 population_state.append(None) # Hata durumunda None ekle

        state = {
            "generation": generation,
            "population_state": [p for p in population_state if p is not None], # Başarısız olanları çıkarma
            "random_state": rnd_state,
            "numpy_random_state": np_rnd_state,
            "tensorflow_random_state": tf_rnd_state, # TensorFlow state'i pickle ile kaydetmek sorunlu olabilir
            "timestamp": datetime.now().isoformat()
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"Checkpoint saved successfully for generation {generation}.")
    except Exception as e:
        logging.error(f"Failed to save checkpoint for generation {generation}: {e}", exc_info=True)


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """Kaydedilmiş evrim durumunu yükler."""
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        return None
    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    try:
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)

        population = []
        for model_state in state["population_state"]:
            try:
                # Eğer model ayrı kaydedildiyse:
                # model = load_model(model_state["saved_path"])
                # population.append(model)

                # Pickle içine gömüldüyse:
                model = Sequential.from_config(model_state["config"])
                model.set_weights(model_state["weights"])
                # Modelin yeniden derlenmesi GEREKİR!
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                model._name = model_state.get("name", f"model_loaded_{random.randint(1000,9999)}") # İsmi geri yükle
                population.append(model)
            except Exception as e:
                logging.error(f"Failed to load model state from checkpoint for model {model_state.get('name', 'UNKNOWN')}: {e}")

        # Sadece başarıyla yüklenen modelleri al
        state["population"] = population
        if not population:
             logging.error("Failed to load any model from the checkpoint population state.")
             return None # Hiç model yüklenemediyse checkpoint geçersiz

        logging.info(f"Checkpoint loaded successfully. Resuming from generation {state['generation'] + 1}.")
        return state
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}", exc_info=True)
        return None

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
     """Verilen klasördeki en son checkpoint dosyasını bulur."""
     checkpoint_dir = os.path.join(output_dir, "checkpoints")
     if not os.path.isdir(checkpoint_dir):
          return None
     checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("evo_gen_") and f.endswith(".pkl")]
     if not checkpoints:
          return None
     # Dosya adından nesil numarasını çıkar ve en yükseğini bul
     latest_gen = -1
     latest_file = None
     for cp in checkpoints:
          try:
               gen_num = int(cp.split('_')[2].split('.')[0])
               if gen_num > latest_gen:
                    latest_gen = gen_num
                    latest_file = os.path.join(checkpoint_dir, cp)
          except (IndexError, ValueError):
               logging.warning(f"Could not parse generation number from checkpoint file: {cp}")
               continue
     return latest_file


# --- Ana Evrim Döngüsü (Checkpoint ve Crossover ile) ---
def evolve_population_v3(population: List[Sequential], X: np.ndarray, y: np.ndarray, start_generation: int, total_generations: int,
                      crossover_rate: float, mutation_rate: float, weight_mut_rate: float, mut_strength: float,
                      tournament_size: int, elitism_count: int, batch_size: int,
                      output_dir: str, checkpoint_interval: int) -> Tuple[Optional[Sequential], List[float], List[float]]:
    """Evrimsel süreci çalıştırır (Checkpoint ve Crossover içerir)."""
    best_fitness_history = []
    avg_fitness_history = []
    best_model_overall = None
    best_fitness_overall = -np.inf

    X_tf = tf.cast(X, tf.float32)
    y_tf = tf.cast(y, tf.float32)

    # --- KAVRAMSAL: Uyarlanabilir Mutasyon Oranı ---
    # current_mutation_rate = mutation_rate # Başlangıç değeri
    # stagnation_counter = 0
    # --------------------------------------------

    for gen in range(start_generation, total_generations):
        generation_start_time = datetime.now()
        # 1. Fitness Değerlendirme
        try:
            fitness_scores = [calculate_fitness(ind, X_tf, y_tf, batch_size) for ind in population]
        except Exception as e:
            logging.critical(f"Error calculating fitness for population in Generation {gen+1}: {e}", exc_info=True)
            if best_model_overall: return best_model_overall, best_fitness_history, avg_fitness_history
            else: raise

        # 2. İstatistikler ve En İyiyi Takip
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        avg_fitness = np.mean(fitness_scores)
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)

        new_best_found = False
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            new_best_found = True
            try:
                best_model_overall = clone_model(population[current_best_idx])
                best_model_overall.set_weights(population[current_best_idx].get_weights())
                best_model_overall.compile(optimizer=Adam(), loss='mse')
                logging.info(f"Generation {gen+1}: *** New overall best fitness found: {best_fitness_overall:.6f} ***")
            except Exception as e:
                 logging.error(f"Could not clone new best model: {e}", exc_info=True)
                 best_fitness_overall = current_best_fitness # Sadece fitness'ı güncelle

        generation_time = (datetime.now() - generation_start_time).total_seconds()
        logging.info(f"Generation {gen+1}/{total_generations} | Best Fitness: {current_best_fitness:.6f} | Avg Fitness: {avg_fitness:.6f} | Time: {generation_time:.2f}s")

        # --- KAVRAMSAL: Uyarlanabilir Mutasyon Oranı Güncelleme ---
        # if new_best_found:
        #     stagnation_counter = 0
        #     # current_mutation_rate = max(min_mutation_rate, current_mutation_rate * 0.98) # Azalt
        # else:
        #     stagnation_counter += 1
        # if stagnation_counter > stagnation_limit:
        #     # current_mutation_rate = min(max_mutation_rate, current_mutation_rate * 1.1) # Artır
        #     stagnation_counter = 0 # Sayacı sıfırla
        # logging.debug(f"Current mutation rate: {current_mutation_rate:.4f}")
        # --------------------------------------------

        # 3. Yeni Popülasyon Oluşturma
        new_population = []

        # 3a. Elitizm
        if elitism_count > 0 and len(population) >= elitism_count:
            try:
                elite_indices = np.argsort(fitness_scores)[-elitism_count:]
                for idx in elite_indices:
                    elite_clone = clone_model(population[idx])
                    elite_clone.set_weights(population[idx].get_weights())
                    elite_clone.compile(optimizer=Adam(), loss='mse')
                    new_population.append(elite_clone)
            except Exception as e:
                 logging.error(f"Error during elitism: {e}", exc_info=True)


        # 3b. Seçilim, Çaprazlama ve Mutasyon
        num_to_generate = len(population) - len(new_population)
        generated_count = 0
        while generated_count < num_to_generate:
            try:
                # İki ebeveyn seç
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)

                child1, child2 = None, None # Çocukları başlat

                # Çaprazlama uygula (belirli bir olasılıkla)
                if random.random() < crossover_rate and parent1 is not parent2:
                    child1, child2 = crossover_individuals(parent1, parent2)

                # Eğer çaprazlama yapılmadıysa veya başarısız olduysa, mutasyonla devam et
                if child1 is None: # İlk çocuk oluşmadıysa
                    # Ebeveynlerden birini mutasyona uğrat
                     parent_to_mutate = parent1 # Veya parent2 veya rastgele biri
                     if random.random() < mutation_rate: # Genel mutasyon oranı kontrolü
                           child1 = mutate_individual(parent_to_mutate, weight_mut_rate, mut_strength)
                     else: # Mutasyon da olmazsa, ebeveyni klonla
                           child1 = clone_model(parent_to_mutate); child1.set_weights(parent_to_mutate.get_weights())
                           child1.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                           child1._name = f"cloned_{parent_to_mutate.name}_{random.randint(1000,9999)}"

                    # Yeni popülasyona ekle
                    if child1:
                         new_population.append(child1)
                         generated_count += 1
                         if generated_count >= num_to_generate: break # Gerekli sayıya ulaşıldıysa çık

                else: # Çaprazlama başarılı olduysa (child1 ve child2 var)
                     # Çaprazlama sonrası çocuklara ayrıca mutasyon uygulama seçeneği eklenebilir
                     # if random.random() < post_crossover_mutation_rate: child1 = mutate(...)
                     # if random.random() < post_crossover_mutation_rate: child2 = mutate(...)

                     new_population.append(child1)
                     generated_count += 1
                     if generated_count >= num_to_generate: break

                     if child2: # İkinci çocuk da varsa ekle
                          new_population.append(child2)
                          generated_count += 1
                          if generated_count >= num_to_generate: break

            except Exception as e:
                logging.error(f"Error during selection/reproduction cycle: {e}", exc_info=True)
                if generated_count < num_to_generate: # Eksik kalırsa rastgele doldur
                    logging.warning("Adding random individual due to reproduction error.")
                    new_population.append(create_individual(y.shape[1], X.shape[1:]))
                    generated_count += 1

        population = new_population[:len(population)] # Popülasyon boyutunu garantile

        # 4. Checkpoint Alma
        if checkpoint_interval > 0 and (gen + 1) % checkpoint_interval == 0:
            try:
                # Rastgele durumları al
                rnd_state = random.getstate()
                np_rnd_state = np.random.get_state()
                # tf_rnd_state = tf.random.get_global_generator().state # TF state kaydetmek zor olabilir
                tf_rnd_state = None # Şimdilik None
                save_checkpoint(output_dir, gen + 1, population, rnd_state, np_rnd_state, tf_rnd_state)
            except Exception as e:
                 logging.error(f"Failed to execute checkpoint saving for generation {gen+1}: {e}", exc_info=True)


    # Döngü sonu
    if best_model_overall is None and population:
         logging.warning("No overall best model tracked. Returning best from final population.")
         final_fitness_scores = [calculate_fitness(ind, X_tf, y_tf, batch_size) for ind in population]
         best_idx_final = np.argmax(final_fitness_scores)
         best_model_overall = population[best_idx_final]
    elif not population:
         logging.error("Evolution finished with an empty population!")
         return None, best_fitness_history, avg_fitness_history

    logging.info(f"Evolution finished. Best fitness achieved: {best_fitness_overall:.6f}")
    return best_model_overall, best_fitness_history, avg_fitness_history

# --- Grafik Çizimi (Öncekiyle aynı) ---
def plot_fitness_history(history_best: List[float], history_avg: List[float], output_dir: str) -> None:
    if not history_best or not history_avg:
        logging.warning("Fitness history is empty, cannot plot.")
        return
    try:
        plt.figure(figsize=(12, 7)); plt.plot(history_best, label="Best Fitness", marker='o', linestyle='-', linewidth=2)
        plt.plot(history_avg, label="Average Fitness", marker='x', linestyle='--', alpha=0.7); plt.xlabel("Generation")
        plt.ylabel("Fitness Score"); plt.title("Evolutionary Fitness History"); plt.legend(); plt.grid(True); plt.tight_layout()
        plot_path = os.path.join(output_dir, "fitness_history.png"); plt.savefig(plot_path); plt.close()
        logging.info(f"Fitness history plot saved to {plot_path}")
    except Exception as e: logging.error(f"Error plotting fitness history: {e}", exc_info=True)

# --- Değerlendirme (Öncekiyle aynı) ---
def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, batch_size: int) -> Dict[str, float]:
    if model is None: return {"test_mse": np.inf, "avg_kendall_tau": 0.0}
    logging.info("Evaluating final model on test data...")
    try:
        y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
        test_mse = np.mean(np.square(y_test - y_pred))
        logging.info(f"Final Test MSE: {test_mse:.6f}")
        sample_size = min(500, X_test.shape[0]); taus = []; indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
        for i in indices:
            try: tau, _ = kendalltau(y_test[i], y_pred[i]);
            if not np.isnan(tau): taus.append(tau)
            except ValueError: pass # Handle constant prediction case
        avg_kendall_tau = np.mean(taus) if taus else 0.0
        logging.info(f"Average Kendall's Tau (on {sample_size} samples): {avg_kendall_tau:.4f}")
        return {"test_mse": float(test_mse), "avg_kendall_tau": float(avg_kendall_tau)}
    except Exception as e:
        logging.error(f"Error during final model evaluation: {e}", exc_info=True)
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}

# --- Ana İş Akışı (Checkpoint Yükleme ile) ---
def run_pipeline_v3(args: argparse.Namespace):
    """Checkpoint ve Crossover içeren ana iş akışı."""

    # Çalıştırma adı ve çıktı klasörü
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evorun_{timestamp}_gen{args.generations}_pop{args.pop_size}"
    # Eğer resume path verilmişse, o klasörü kullan
    output_dir = args.resume_from if args.resume_from else os.path.join(args.output_base_dir, run_name)
    resume_run = bool(args.resume_from)
    if resume_run:
         run_name = os.path.basename(output_dir) # Klasör adını kullan
         logging.info(f"Attempting to resume run from: {output_dir}")
    else:
         try: os.makedirs(output_dir, exist_ok=True)
         except OSError as e: print(f"FATAL: Could not create output directory: {output_dir}. Error: {e}", file=sys.stderr); sys.exit(1)

    # Loglamayı ayarla ('a' modu ile devam etmeye uygun)
    setup_logging(output_dir)
    logging.info(f"========== Starting/Resuming EvoNet Pipeline Run: {run_name} ==========")
    logging.info(f"Output directory: {output_dir}")

    # --- Checkpoint Yükleme ---
    start_generation = 0
    population = []
    initial_state_loaded = False
    latest_checkpoint_path = find_latest_checkpoint(output_dir) if resume_run else None

    if latest_checkpoint_path:
        loaded_state = load_checkpoint(latest_checkpoint_path)
        if loaded_state:
            start_generation = loaded_state['generation'] # Kaldığı nesilden başla
            population = loaded_state['population']
            # Rastgele durumları geri yükle
            try:
                random.setstate(loaded_state['random_state'])
                np.random.set_state(loaded_state['numpy_random_state'])
                # tf.random.set_global_generator(tf.random.Generator.from_state(loaded_state['tensorflow_random_state'])) # TF state sorunlu olabilir
                logging.info(f"Random states restored from checkpoint.")
            except Exception as e:
                 logging.warning(f"Could not fully restore random states from checkpoint: {e}")
            initial_state_loaded = True
            logging.info(f"Resuming from Generation {start_generation + 1} with {len(population)} individuals.")
        else:
             logging.error("Failed to load checkpoint. Starting from scratch.")
             resume_run = False # Checkpoint yüklenemediyse sıfırdan başla
    elif resume_run:
         logging.warning(f"Resume requested but no valid checkpoint found in {output_dir}. Starting from scratch.")
         resume_run = False # Checkpoint yoksa sıfırdan başla


    # --- Sıfırdan Başlama veya Devam Etme Ayarları ---
    if not initial_state_loaded:
        # Argümanları logla ve kaydet (sadece sıfırdan başlarken)
        logging.info("--- Configuration ---")
        args_dict = vars(args)
        for k, v in args_dict.items(): logging.info(f"  {k:<20}: {v}")
        logging.info("---------------------")
        config_path = os.path.join(output_dir, "config.json")
        try:
            with open(config_path, 'w') as f: json.dump(args_dict, f, indent=4, sort_keys=True)
            logging.info(f"Configuration saved to {config_path}")
        except Exception as e: logging.error(f"Failed to save configuration: {e}", exc_info=True)

        # Rastgele tohumları ayarla
        try:
            random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
            logging.info(f"Using random seed: {args.seed}")
        except Exception as e: logging.warning(f"Could not set all random seeds: {e}")

        # GPU kontrolü
        is_gpu_available = check_gpu()

        # Veri Üretimi
        try:
            X_train, y_train = generate_data(args.train_samples, args.seq_length)
            X_test, y_test = generate_data(args.test_samples, args.seq_length)
            input_shape = X_train.shape[1:]
        except Exception: logging.critical("Failed to generate data. Exiting."); sys.exit(1)

        # Popülasyon Başlatma
        logging.info(f"--- Initializing Population (Size: {args.pop_size}) ---")
        try:
            population = [create_individual(args.seq_length, input_shape) for _ in range(args.pop_size)]
            logging.info("Population initialized successfully.")
        except Exception: logging.critical("Failed to initialize population. Exiting."); sys.exit(1)
    else:
         # Checkpoint'ten devam ediliyorsa, veriyi yeniden üretmemiz gerekebilir
         # veya checkpoint'e veriyi de dahil edebiliriz (büyük olabilir).
         # Şimdilik veriyi yeniden üretelim.
         logging.info("Reloading data for resumed run...")
         is_gpu_available = check_gpu() # GPU durumunu tekrar kontrol et
         try:
            X_train, y_train = generate_data(args.train_samples, args.seq_length)
            X_test, y_test = generate_data(args.test_samples, args.seq_length)
         except Exception: logging.critical("Failed to reload data for resumed run. Exiting."); sys.exit(1)
         # Config dosyasını tekrar okuyup loglayabiliriz
         config_path = os.path.join(output_dir, "config.json")
         try:
              with open(config_path, 'r') as f: args_dict = json.load(f)
              logging.info("--- Loaded Configuration (from resumed run) ---")
              for k, v in args_dict.items(): logging.info(f"  {k:<20}: {v}")
              logging.info("-----------------------------------------------")
         except Exception as e:
              logging.warning(f"Could not reload config.json: {e}")
              args_dict = vars(args) # Argümanları kullan


    # Evrim Süreci
    logging.info(f"--- Starting/Resuming Evolution ({args.generations} Total Generations) ---")
    if start_generation >= args.generations:
         logging.warning(f"Loaded checkpoint generation ({start_generation}) is already >= total generations ({args.generations}). Skipping evolution.")
         best_model_unevolved = population[0] if population else None # En iyi modeli checkpoint'ten almaya çalışmak lazım
         best_fitness_hist, avg_fitness_hist = [], [] # Geçmişi de yüklemek lazım
         # TODO: Checkpoint'ten en iyi modeli ve geçmişi de yükle
         # Şimdilik basitleştirilmiş - evrim atlanıyor
    else:
        try:
            best_model_unevolved, best_fitness_hist, avg_fitness_hist = evolve_population_v3(
                population, X_train, y_train, start_generation, args.generations,
                args.crossover_rate, args.mutation_rate, args.weight_mut_rate, args.mutation_strength,
                args.tournament_size, args.elitism_count, args.batch_size,
                output_dir, args.checkpoint_interval
            )
        except Exception as e:
            logging.critical(f"Fatal error during evolution process: {e}", exc_info=True)
            sys.exit(1)
    logging.info("--- Evolution Complete ---")

    # (Fitness geçmişini kaydetme ve çizdirme - öncekiyle aynı)
    if best_fitness_hist or avg_fitness_hist: # Sadece listeler boş değilse
        # Geçmişi de checkpoint'ten yükleyip birleştirmek gerekebilir.
        # Şimdilik sadece bu çalıştırmadaki kısmı kaydediyoruz/çizdiriyoruz.
        # TODO: Checkpoint'ten yüklenen geçmişle birleştir.
        plot_fitness_history(best_fitness_hist, avg_fitness_hist, output_dir)
        history_path = os.path.join(output_dir, "fitness_history_run.csv") # Farklı isim?
        try:
            history_data = np.array([np.arange(start_generation + 1, start_generation + len(best_fitness_hist) + 1), best_fitness_hist, avg_fitness_hist]).T
            np.savetxt(history_path, history_data, delimiter=',', header='Generation,BestFitness,AvgFitness', comments='', fmt=['%d', '%.8f', '%.8f'])
            logging.info(f"Fitness history (this run) saved to {history_path}")
        except Exception as e: logging.error(f"Could not save fitness history data: {e}")
    else: logging.warning("Fitness history is empty, skipping saving/plotting.")

    # (En iyi modelin son eğitimi, değerlendirme ve sonuç kaydı - öncekiyle aynı)
    if best_model_unevolved is None:
        logging.error("Evolution did not yield a best model. Skipping final training and evaluation.")
        final_metrics = {"test_mse": np.inf, "avg_kendall_tau": 0.0}; final_model_path = None; training_summary = {}
    else:
        logging.info("--- Starting Final Training of Best Evolved Model ---")
        try:
            final_model = clone_model(best_model_unevolved); final_model.set_weights(best_model_unevolved.get_weights())
            final_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            logging.info("Model Summary of Best Evolved (Untrained):"); final_model.summary(print_fn=logging.info)
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, min_lr=1e-7, verbose=1)
            history = final_model.fit(X_train, y_train, epochs=args.epochs_final_train, batch_size=args.batch_size, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=2)
            logging.info("Final training complete.")
            training_summary = {"epochs_run": len(history.history['loss']), "final_train_loss": history.history['loss'][-1], "final_val_loss": history.history['val_loss'][-1]}
            final_metrics = evaluate_model(final_model, X_test, y_test, args.batch_size)
            final_model_path = os.path.join(output_dir, "best_evolved_model_trained.keras")
            final_model.save(final_model_path); logging.info(f"Final trained model saved to {final_model_path}")
        except Exception as e:
             logging.error(f"Error during final training or evaluation: {e}", exc_info=True)
             final_metrics = {"test_mse": np.inf, "avg_kendall_tau": 0.0}; final_model_path = None; training_summary = {"error": str(e)}

    logging.info("--- Saving Final Results ---")
    final_results = { # ... (öncekiyle aynı sonuç yapısı) ...
        "run_info": {"run_name": run_name, "timestamp": timestamp, "output_directory": output_dir, "gpu_used": is_gpu_available, "resumed": resume_run},
        "config": args_dict,
        "evolution_summary": { # TODO: Checkpoint'ten yüklenen geçmişle birleştirilmeli
            "generations_run_this_session": len(best_fitness_hist) if best_fitness_hist else 0,
            "best_fitness_achieved_overall": best_fitness_overall if best_fitness_overall > -np.inf else None,
            "best_fitness_final_gen": best_fitness_hist[-1] if best_fitness_hist else None,
            "avg_fitness_final_gen": avg_fitness_hist[-1] if avg_fitness_hist else None, },
        "final_training_summary": training_summary, "final_evaluation_on_test": final_metrics, "saved_model_path": final_model_path }
    results_path = os.path.join(output_dir, "final_results.json")
    try:
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        with open(results_path, 'w') as f: json.dump(final_results, f, indent=4, default=convert_numpy_types)
        logging.info(f"Final results summary saved to {results_path}")
    except Exception as e: logging.error(f"Failed to save final results JSON: {e}", exc_info=True)

    logging.info(f"========== Pipeline Run {run_name} Finished ==========")


# --- Argüman Ayrıştırıcı (Yeni Argümanlar Eklendi) ---
def parse_arguments_v3() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoNet v3: Neuroevolution with Crossover & Checkpointing")

    # --- Dizinler ve Kontrol ---
    parser.add_argument('--output_base_dir', type=str, default=DEFAULT_OUTPUT_BASE_DIR, help='Base directory for new runs.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a previous run directory to resume from.')
    parser.add_argument('--checkpoint_interval', type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help='Save checkpoint every N generations (0 to disable).')

    # --- Veri Ayarları ---
    parser.add_argument('--seq_length', type=int, default=DEFAULT_SEQ_LENGTH, help='Length of sequences.')
    parser.add_argument('--train_samples', type=int, default=5000, help='Number of training samples.')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples.')

    # --- Evrim Parametreleri ---
    parser.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE, help='Population size.')
    parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS, help='Total number of generations.')
    parser.add_argument('--crossover_rate', type=float, default=DEFAULT_CROSSOVER_RATE, help='Probability of applying crossover.')
    parser.add_argument('--mutation_rate', type=float, default=DEFAULT_MUTATION_RATE, help='Probability of applying mutation (if crossover is not applied).')
    parser.add_argument('--weight_mut_rate', type=float, default=DEFAULT_WEIGHT_MUT_RATE, help='Weight mutation probability within mutation.')
    # parser.add_argument('--activation_mut_rate', type=float, default=DEFAULT_ACTIVATION_MUT_RATE, help='Activation mutation probability (experimental).')
    parser.add_argument('--mutation_strength', type=float, default=DEFAULT_MUTATION_STRENGTH, help='Std dev for weight mutation noise.')
    parser.add_argument('--tournament_size', type=int, default=DEFAULT_TOURNAMENT_SIZE, help='Tournament selection size.')
    parser.add_argument('--elitism_count', type=int, default=DEFAULT_ELITISM_COUNT, help='Number of elite individuals.')

    # --- Eğitim ve Değerlendirme ---
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size.')
    parser.add_argument('--epochs_final_train', type=int, default=DEFAULT_EPOCHS_FINAL_TRAIN, help='Max epochs for final training.')

    # --- Tekrarlanabilirlik ---
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: random).')

    args = parser.parse_args()
    if args.seed is None: args.seed = random.randint(0, 2**32 - 1); print(f"Generated random seed: {args.seed}")
    # Basit kontrol: Crossover + Mutation oranı > 1 olmamalı (teknik olarak olabilir ama mantık gereği biri seçilmeli)
    # if args.crossover_rate + args.mutation_rate > 1.0: logging.warning("Sum of crossover and mutation rates exceeds 1.0")
    return args


# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    cli_args = parse_arguments_v3()
    try:
        run_pipeline_v3(cli_args)
    except SystemExit: pass
    except Exception as e:
        print(f"\nFATAL UNHANDLED ERROR in main execution block: {e}", file=sys.stderr)
        if logging.getLogger().hasHandlers(): logging.critical("FATAL UNHANDLED ERROR:", exc_info=True)
        else: import traceback; print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)